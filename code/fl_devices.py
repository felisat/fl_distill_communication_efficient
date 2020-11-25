import random
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np 
from models import outlier_net




device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

class Device(object):
  def __init__(self, model_fn, optimizer_fn, loader,client_data=None, init=None, **kwargs):
    self.model = model_fn().to(device)
    self.loader = loader

    self.W = {key : value for key, value in self.model.named_parameters()}
    self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
    self.W_old = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}

    self.optimizer_fn = optimizer_fn
    self.optimizer = optimizer_fn(self.model.parameters())   
    self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.96)  

    self.c_round = 0
    
  def evaluate(self, loader=None):
    return eval_op(self.model, self.loader if not loader else loader)

  def save_model(self, path=None, name=None, verbose=True):
    if name:
      torch.save(self.model.state_dict(), path+name)
      if verbose: print("Saved model to", path+name)

  def load_model(self, path=None, name=None, verbose=True):
    if name:
      self.model.load_state_dict(torch.load(path+name))
      if verbose: print("Loaded model from", path+name)


  def predict_logit(self, x):
    """Logit prediction on input"""
    self.model.train()
    with torch.no_grad():
      y_ = self.model(x)
    return y_

  def predict(self, x):
    """Softmax prediction on input"""
    y_ = nn.Softmax(1)(self.predict_logit(x))
    return y_

  def predict_max(self, x):
    """Onehot Argmax prediction on input"""
    y_ = self.predict(x)
    amax = torch.argmax(y_, dim=1).detach()
    t = torch.zeros_like(y_)
    t[torch.arange(y_.shape[0]),amax] = 1

    return t.detach()

  def predict_sampled(self, x):
    """Onehot Argmax prediction on input"""
    y_ = self.predict(x)

    prob_dist = torch.distributions.Categorical(y_) # probs should be of size batch x classes
    samples = prob_dist.sample().detach()


    t = torch.zeros_like(y_)
    t[torch.arange(y_.shape[0]),samples] = 1

    return t.detach()

  def predict_random(self, x):
    """Random prediction on input"""
    adv = torch.zeros(size=[x.shape[0],10], device="cuda")
    adv[:,0] = 1.0
    return adv


  def compute_prediction_matrix(self, distill_loader, argmax=True):
    predictions = []
    idcs = []
    for x, _, idx in distill_loader:
      x = x.to(device)
      s_predict = self.predict(x).detach()
      predictions += [s_predict]
      idcs += [idx]

    argidx = torch.argsort(torch.cat(idcs, dim=0))

    predictions =  torch.cat(predictions, dim=0)[argidx].detach().cpu().numpy()

    if argmax:
      return np.argmax(predictions, axis=-1).astype("uint8")
    else:
      return predictions



      
class Client(Device):
  def __init__(self, model_fn, optimizer_fn, loader, counts, distill_loader, init=None, idnum=None):
    super().__init__(model_fn, optimizer_fn, loader, init)
    self.id = idnum
    self.feature_extractor = None
    self.distill_loader = distill_loader

  def synchronize_with_server(self, server, c_round):
    if server.co_model is None:
      server_state = server.model.state_dict()
    else:
      server_state = server.co_model.state_dict()
      print("Using co model")
    server_state = {k : v for k, v in server_state.items() if "binary" not in k}
    self.model.load_state_dict(server_state, strict=False)
    self.c_round = c_round

    # update data loader subbatch distribution
    self.loader.update(c_round=c_round)
    #copy(target=self.W, source=server.W)
    
  def compute_weight_update(self, epochs=1, loader=None, reset_optimizer=False, **kwargs):
    if reset_optimizer:
      self.optimizer = self.optimizer_fn(self.model.parameters())  
 
    train_stats = train_op(self.model, self.loader if not loader else loader, self.optimizer, self.scheduler, epochs, **kwargs)
    #print(self.label_counts)
    #eval_scores(self.model, self.distill_loader)

    return train_stats




  def train_shallow_outlier_detector(self, model, distill_loader, **kw_args):
    from sklearn.decomposition import PCA
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import IsolationForest
    from sklearn.neural_network import MLPRegressor
    from  sklearn.neighbors import KernelDensity


    if self.feature_extractor is not None:
      X_train = torch.cat([self.feature_extractor.f(x[0].cuda()).detach().cpu() for x in self.loader], dim=0).reshape(-1,512).numpy()
      X_distill = torch.cat([self.feature_extractor.f(x[0][0].cuda()).detach().cpu() for x in distill_loader], dim=0).reshape(-1,512).numpy()
    else:
      X_train = torch.cat([x[0] for x in self.loader], dim=0).reshape(-1,1*32*32).numpy()
      X_distill = torch.cat([x[0][0] for x in distill_loader], dim=0).reshape(-1,1*32*32).numpy()

    pca = PCA(n_components=0.95, svd_solver="full", whiten=True)
    pca.fit(X_train)
    X_train_ = pca.transform(X_train)
    X_distill_ = pca.transform(X_distill)


    if model=="ocsvm":
      self.outlier_model = OneClassSVM(**kw_args).fit(X_train_)
      self.outlier_model.score = lambda x : self.outlier_model.score_samples(x)
    
    elif model=="isolation_forest":
      self.outlier_model = IsolationForest(**kw_args).fit(X_train_)

    elif model=="kde":
      self.outlier_model = KernelDensity(**kw_args).fit(X_train_)
      self.outlier_model.score = lambda x : np.exp(self.outlier_model.score_samples(x))

    scores = self.outlier_model.score(X_distill_)

    norm_scores = (scores-np.min(scores))/(np.max(scores)-np.min(scores))
    self.scores = torch.Tensor(norm_scores)



  def predict_deep_outlier_score(self, x):
    self.model.eval()
    s = torch.nn.Softmax(1)(self.model.forward_binary(x.cuda()))[:,0]
    self.model.train()
    return s



    
 
class Server(Device):
  def __init__(self, model_fn, optimizer_fn, loader, unlabeled_loader, init=None):
    super().__init__(model_fn, optimizer_fn, loader, init)
    self.distill_loader = unlabeled_loader

    self.model_fn = model_fn
    self.co_model = None
    
  def select_clients(self, clients, frac=1.0):
    return random.sample(clients, int(len(clients)*frac)) 
    
  def aggregate_weight_updates(self, clients):
    reduce_average(target=self.W, sources=[client.W for client in clients])

  def co_distill(self, distill_iter, quantization_bits=None):
    self.co_model = self.model_fn().to(device)

    self.co_optimizer = self.optimizer_fn(self.co_model.parameters())   
    self.co_model.train()  
    self.model.train()

    acc = 0
    itr = 0
    #for ep in range(epochs):
    while True:
      running_loss, samples = 0.0, 0
      for x,_, idx in tqdm(self.distill_loader):   
        x = x.to(device) 
        itr += 1    

        y = nn.Softmax(1)(self.model(x))

        self.co_optimizer.zero_grad()
        y_ = nn.Softmax(1)(self.co_model(x))


        if quantization_bits:
          y = quantize_probs(y, quantization_bits)


        loss = kulbach_leibler_divergence(y_,y.detach())

        running_loss += loss.item()*y.shape[0]
        samples += y.shape[0]

        loss.backward()
        self.co_optimizer.step()  

      print(running_loss / samples)
      if itr >= distill_iter:
        acc_new = eval_op(self.co_model, self.loader)["accuracy"]
        print(acc_new)

        #self.model.load_state_dict(self.co_model.state_dict(), strict=False)

        return {"loss" : running_loss / samples, "acc" : acc_new}


  def distill(self, clients, distill_iter, mode, reset_optimizer=False, reset_model=False):
    print("Distilling...")
    if reset_model:
      self.model = self.model_fn().to(device)
      self.optimizer = self.optimizer_fn(self.model.parameters()) 

    if reset_optimizer:
      self.optimizer = self.optimizer_fn(self.model.parameters())   
    self.model.train()  

    acc = 0
    itr = 0
    #for ep in range(epochs):
    while True:
      running_loss, samples = 0.0, 0
      for x,_, idx in self.distill_loader:   
        x = x.to(device) 
        itr += 1    

        if mode == "mean_probs":
          y = torch.zeros([x.shape[0], 10], device="cuda")
          for i, client in enumerate(clients):
            y_p = client.predict(x)
            y += (y_p/len(clients)).detach()

        if mode == "mean_logits":
          y = torch.zeros([x.shape[0], 10], device="cuda")
          for i, client in enumerate(clients):
            y_p = client.predict_logit(x)
            y += (y_p/len(clients)).detach()
          y = nn.Softmax(1)(y)         

        if mode == "sample":
          y = torch.mean(torch.stack([client.predict_sampled(x) for client in clients]), dim=0)

        if mode == "pate":
          hist = torch.sum(torch.stack([client.predict_max(x) for client in clients]), dim=0)
          amax = torch.argmax(hist, dim=1)
          y = torch.zeros_like(hist)
          y[torch.arange(hist.shape[0]),amax] = 1

        if mode == "pate_up":
          y = torch.mean(torch.stack([client.predict_max(x) for client in clients]), dim=0)

        if mode == "mean_logits_er":
          y = torch.zeros([x.shape[0], 10], device="cuda")
          for i, client in enumerate(clients):
            y_p = client.predict_logit(x)
            y += (y_p/len(clients)).detach()
          y = nn.Softmax(1)(y/0.1)  

        if isinstance(mode, list) and mode[0] == "quantized":
          bits = mode[1]

          y = torch.zeros([x.shape[0], 10], device="cuda")
          for i, client in enumerate(clients):
            y_p = client.predict(x)
            y_quant = quantize_probs(y_p, bits)

            y += (y_quant/len(clients)).detach()


        if isinstance(mode, list) and mode[0] == "quantized_down":
          bits = mode[1]
          bits_down = mode[2]

          y = torch.zeros([x.shape[0], 10], device="cuda")
          for i, client in enumerate(clients):
            y_p = client.predict(x)
            y_quant = quantize_probs(y_p, bits)

            y += (y_quant/len(clients)).detach()


          y = quantize_probs(y, bits_down)


        self.optimizer.zero_grad()
        y_ = nn.Softmax(1)(self.model(x))

        loss = kulbach_leibler_divergence(y_,y.detach())

        running_loss += loss.item()*y.shape[0]
        samples += y.shape[0]

        loss.backward()
        self.optimizer.step()  

   
      if itr >= distill_iter:
        acc_new = eval_op(self.model, self.loader)["accuracy"]
        print(acc_new)
        return {"loss" : running_loss / samples, "acc" : acc_new}



def train_op(model, loader, optimizer, scheduler, epochs, lambda_fedprox=0.0, **kw_args):
    model.train()  
    running_loss, samples = 0.0, 0

    W0 = {k : v.detach().clone() for k, v in model.named_parameters()}

    for ep in range(epochs):
      for x, y, source, index in loader:   
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        loss = nn.CrossEntropyLoss()(model(x), y)

        if lambda_fedprox != 0.0:
          loss += lambda_fedprox * torch.sum((flatten(W0).cuda()-flatten(dict(model.named_parameters())).cuda())**2)

        running_loss += loss.item()*y.shape[0]
        samples += y.shape[0]

        loss.backward()
        optimizer.step()  
      #scheduler.step()

    return {"loss" : running_loss / samples}





def train_op_with_score(model, loader, optimizer, scheduler, epochs, lambda_outlier=0.1, lambda_ent=0.0, lambda_fedprox=0.0, only_train_final_outlier_layer=False, **kwargs):
    model.train()  

    W0 = {k : v.detach().clone() for k, v in model.named_parameters()}

    running_loss, running_class, running_ent, running_binary, samples = 0.0, 0.0, 0.0, 0.0, 0
    for ep in range(epochs):
      for x, y, source, index in loader:   
        x, y, source = x.to(device), y.to(device), source.to(device).long()

        optimizer.zero_grad()

        classification_loss = nn.CrossEntropyLoss()(model(x[source == 0]), y[source == 0])
       
        outlier_loss = nn.CrossEntropyLoss()(model.forward_binary(x, only_train_final_outlier_layer), source)

        loss =  lambda_outlier * kwargs["distill_weight"] * warmup(kwargs["c_round"], kwargs["max_c_round"], type=kwargs["warmup_type"]) * outlier_loss + classification_loss 

        if lambda_fedprox > 0.0:
          loss += lambda_fedprox * torch.sum((flatten(W0).cuda()-flatten(dict(model.named_parameters())).cuda())**2)

        if lambda_ent > 0.0:
          p = torch.nn.Softmax(1)(model.forward_binary(x))
          ent = -torch.mean(torch.sum(p * torch.log(p.clamp_min(1e-7)), dim=1)) 
          loss += lambda_ent * ent


        running_loss += loss.item()*y.shape[0]
        #running_ent += ent.item()*y.shape[0]
        running_binary += outlier_loss.item()*y.shape[0]
        running_class += classification_loss.item()*y.shape[0]

        samples += y.shape[0]

        loss.backward()
        optimizer.step()  
      #scheduler.step()

    return {"loss" : running_loss / samples, "outlier_loss" : running_binary / samples, "classification_loss" : running_class / samples}


def eval_scores(model, eval_loader):
    preds = []
    ys = []
    for (x, y), idx in eval_loader:
        preds += [torch.nn.Softmax(1)(model.forward_binary(x.cuda()))[:,0].cpu().detach().numpy()] 
        ys += [y.numpy()]
        
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)

    for i in range(10):
      print(" -",i, np.mean(preds[ys==i]), "+-", np.std(preds[ys==i]))


def eval_op(model, loader):
    model.train()
    samples, correct = 0, 0

    with torch.no_grad():
      for i, (x, y, source, index) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        y_ = model(x)
        _, predicted = torch.max(y_.detach(), 1)
        
        samples += y.shape[0]
        correct += (predicted == y).sum().item()

    return {"accuracy" : correct/samples}

def warmup(curr, max, type="constant"):
    if type == "constant":
        return 1
    elif type == "tanh":
        return np.tanh(curr / (0.5 * max))

def flatten(source):
  return torch.cat([value.flatten() for value in source.values()])

def copy(target, source):
  for name in target:
    target[name].data = source[name].detach().clone()
    
def reduce_average(target, sources):
  for name in target:
      target[name].data = torch.mean(torch.stack([source[name].detach() for source in sources]), dim=0).clone()

def subtract_(target, minuend, subtrahend):
  for name in target:
    target[name].data = minuend[name].detach().clone()-subtrahend[name].detach().clone()

def sigmoid(x):
  return np.exp(x)/(1+np.exp(x))

def kulbach_leibler_divergence(predicted, target):
  return -(target * torch.log(predicted.clamp_min(1e-7))).sum(dim=-1).mean() 
  #+ 1*(target.clamp(min=1e-7) * torch.log(target.clamp(min=1e-7))).sum(dim=-1).mean()



def quant(x, bits):
    m = 2**bits-1
    return torch.round(x * m)/m

def quantize_probs(x, bits):
    q = quant(x, bits)
    
    delt = torch.round((1-q.sum(dim=1))*(2**bits-1))
    absdelt = torch.abs(delt).long()
    
    V = torch.argsort(-(x-q)*delt.view(-1,1),dim=1)
    
    for i, (k, s) in enumerate(zip(absdelt, torch.sign(delt))):
        pos = V[i,:k]

        for j in pos:
            q[i, j] += s / (2**bits-1)
            
    return q