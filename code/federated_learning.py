import os, argparse, json, copy, time
from tqdm import tqdm
import torch, torchvision
import numpy as np
from torch.utils.data import DataLoader

import data, models 
import experiment_manager as xpm
from fl_devices import Client, Server

np.set_printoptions(precision=4, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("--schedule", default="main", type=str)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=None, type=int)
parser.add_argument("--reverse_order", default=False, type=bool)
parser.add_argument("--hp", default=None, type=str)

parser.add_argument("--DATA_PATH", default=None, type=str)
parser.add_argument("--RESULTS_PATH", default=None, type=str)
parser.add_argument("--CHECKPOINT_PATH", default=None, type=str)

args = parser.parse_args()



def run_experiment(xp, xp_count, n_experiments):

  print(xp)
  hp = xp.hyperparameters
  
  model_fn, optimizer, optimizer_hp = models.get_model(hp["net"])
  optimizer_fn = lambda x : optimizer(x, **{k : hp[k] if k in hp else v for k, v in optimizer_hp.items()}) 
  train_data, test_data = data.get_data(hp["dataset"], args.DATA_PATH)
  all_distill_data = data.get_data(hp["distill_dataset"], args.DATA_PATH)

  np.random.seed(0)
  # What fraction of the unlabeled data should be used for training the anomaly detector
  distill_data = data.IdxSubset(all_distill_data, np.random.permutation(len(all_distill_data))[:hp["n_distill"]]) # data used for distillation

  client_data, label_counts = data.get_client_data(train_data, n_clients=hp["n_clients"], 
        classes_per_client=hp["classes_per_client"])


  client_loaders = [data.DataMerger({'base': local_data}, mixture_coefficients={'base':1}, **hp) for local_data in client_data]

  test_loader = data.DataMerger({'base': data.IdxSubset(test_data, list(range(len(test_data))))}, mixture_coefficients={'base':1}, batch_size=100)
  distill_loader = DataLoader(distill_data, batch_size=128, shuffle=True)

  clients = [Client(model_fn, optimizer_fn, loader, idnum=i, counts=counts, distill_loader=distill_loader) for i, (loader , counts) in enumerate(zip(client_loaders, label_counts))]
  server = Server(model_fn, lambda x : torch.optim.Adam(x, lr=1e-3), test_loader, distill_loader)



  models.print_model(server.model)

  # Start Distributed Training Process
  print("Start Distributed Training..\n")
  t1 = time.time()

  xp.log({"server_val_{}".format(key) : value for key, value in server.evaluate().items()})
  for c_round in range(1, hp["communication_rounds"]+1):

    participating_clients = server.select_clients(clients, hp["participation_rate"])
    xp.log({"participating_clients" : np.array([client.id for client in participating_clients])})

    for client in tqdm(participating_clients):
      client.synchronize_with_server(server, c_round)

      train_stats = client.compute_weight_update(hp["local_epochs"], train_oulier_model=hp["aggregation_mode"] in ["FAD+S", "FAD+P+S"], c_round=c_round,
                max_c_round=hp["communication_rounds"], **hp) 
      print(train_stats)

      if hp["save_softlabels"]:
        predictions = client.compute_prediction_matrix(server.distill_loader)
        xp.log({"client_{}_predictions".format(client.id) : predictions})

    if hp["aggregation_mode"] in ["FA"]:
      server.aggregate_weight_updates(participating_clients)

    if hp["aggregation_mode"] in ["FD", "FDcup", "FDsample", "FDcupdown", "FDer", "FDquant"]:
      distill_mode = {"FD" : "mean_probs", "FDcup" : "pate_up", "FDsample" : "sample", "FDcupdown" : "pate", "FDer" : "mean_logits_er", "FDquant" : ["quantized", hp["quantization_bits"]]}[hp["aggregation_mode"]]
      distill_stats = server.distill(participating_clients, hp["distill_iter"], mode=distill_mode, reset_model=hp["reset_model"])

      if hp["co_distill"]:
        server.co_distill(hp["co_distill_iter"])


    # Logging
    if xp.is_log_round(c_round):
      print("Experiment: {} ({}/{})".format(args.schedule, xp_count+1, n_experiments))   
      
      xp.log({'communication_round' : c_round, 'epochs' : c_round*hp['local_epochs']})
      xp.log({key : clients[0].optimizer.__dict__['param_groups'][0][key] for key in optimizer_hp})
      
      # Evaluate  
      xp.log({"server_val_{}".format(key) : value for key, value in server.evaluate().items()})

      # Save results to Disk
      try:
        xp.save_to_disc(path=args.RESULTS_PATH, name=hp['log_path'])
      except:
        print("Saving results Failed!")

      # Timing
      e = int((time.time()-t1)/c_round*(hp['communication_rounds']-c_round))
      print("Remaining Time (approx.):", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60), 
                "[{:.2f}%]\n".format(c_round/hp['communication_rounds']*100))

  # Save model to disk
  server.save_model(path=args.CHECKPOINT_PATH, name=hp["save_model"])
    
  # Delete objects to free up GPU memory
  del server; clients.clear()
  torch.cuda.empty_cache()


def run():


  experiments_raw = json.loads(args.hp)


  hp_dicts = [hp for x in experiments_raw for hp in xpm.get_all_hp_combinations(x)][args.start:args.end]
  if args.reverse_order:
    hp_dicts = hp_dicts[::-1]
  experiments = [xpm.Experiment(hyperparameters=hp) for hp in hp_dicts]

  print("Running {} Experiments..\n".format(len(experiments)))
  for xp_count, experiment in enumerate(experiments):
    run_experiment(experiment, xp_count, len(experiments))


if __name__ == "__main__":
  run()
    