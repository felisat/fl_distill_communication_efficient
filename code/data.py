import torch, torchvision
import numpy as np
import copy

from torch.utils.data import DataLoader

from itertools import cycle

def get_mnist(path):
  transforms = torchvision.transforms.Compose([ torchvision.transforms.Resize((32,32)),
                                                torchvision.transforms.ToTensor(),    
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                ])
  train_data = torchvision.datasets.MNIST(root=path+"MNIST", train=True, download=True, transform=transforms)
  test_data = torchvision.datasets.MNIST(root=path+"MNIST", train=False, download=True, transform=transforms)

  return train_data, test_data


def get_emnist(path):
  transforms = torchvision.transforms.Compose([ torchvision.transforms.Resize((32,32)),
                                                torchvision.transforms.ToTensor(),    
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                ])
  data = torchvision.datasets.EMNIST(root=path, split="byclass", download=True, transform=transforms)
  #test_data = torchvision.datasets.MNIST(root=path+"EMNIST", train=False, download=True, transform=transforms)

  return data

def get_cifar10(path):
  transforms = torchvision.transforms.Compose([
                                          #torchvision.transforms.RandomCrop(32, padding=4),
                                          #torchvision.transforms.RandomHorizontalFlip(),
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                               (0.2023, 0.1994, 0.2010))
                                          ])
  train_data = torchvision.datasets.CIFAR10(root=path+"CIFAR", train=True, download=True, transform=transforms)
  test_data = torchvision.datasets.CIFAR10(root=path+"CIFAR", train=False, download=True, transform=transforms)

  return train_data, test_data


def get_cifar_distill(path):
  transforms = torchvision.transforms.Compose([
                                          #torchvision.transforms.RandomCrop(32, padding=4),
                                          #torchvision.transforms.RandomHorizontalFlip(),
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                               (0.2023, 0.1994, 0.2010))
                                          ])
  test_data = torchvision.datasets.CIFAR10(root=path+"CIFAR", train=True, download=True, transform=transforms)

  return test_data


def get_cifar100(path):
  transforms = torchvision.transforms.Compose([
                                          #torchvision.transforms.RandomCrop(32, padding=4),
                                          #torchvision.transforms.RandomHorizontalFlip(),
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                               (0.2023, 0.1994, 0.2010))
                                          ])
  train_data = torchvision.datasets.CIFAR100(root=path+"CIFAR100", train=True, download=True, transform=transforms)
  test_data = torchvision.datasets.CIFAR100(root=path+"CIFAR100", train=False, download=True, transform=transforms)

  all_data = torch.utils.data.ConcatDataset([train_data, test_data])

  return all_data


def get_stl10(path):
  transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                               (0.2023, 0.1994, 0.2010))
                                               ])

  data = torchvision.datasets.STL10(root=path+"STL10", split='unlabeled', folds=None, 
                             transform=transforms,
                                    download=True)
  return data

def get_svhn(path):
  transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                               (0.2023, 0.1994, 0.2010))
                                               ])

  data = torchvision.datasets.SVHN(root=path+"SVHN", split='train', transform=transforms,
                                    download=True)
  return data


def get_data(dataset, path):
  return {"cifar10" : get_cifar10, "mnist" : get_mnist, "emnist" : get_emnist,"stl10" : get_stl10, "svhn" : get_svhn, "cifar100" : get_cifar100, "cifar_distill" : get_cifar_distill}[dataset](path)

def get_client_data(train_data, n_clients=10, classes_per_client=0, n_data=None):
  subset_idcs = split_dirichlet(train_data.targets, n_clients, n_data, classes_per_client)
  label_counts = [np.bincount(np.array(train_data.targets)[i], minlength=10) for i in subset_idcs]
  client_data = [IdxSubset(train_data, subset_idcs[i]) for i in range(n_clients)]

  return client_data, label_counts



def split_image_data(labels, n_clients, n_data, classes_per_client):

  np.random.seed(0)

  if isinstance(labels, torch.Tensor):
    labels = labels.numpy()
  if not n_data: 
    n_data = len(labels)
  n_labels = np.max(labels) + 1
  label_idcs = {l : np.argwhere(np.array(labels[:n_data])==l).flatten().tolist() for l in range(n_labels)}

  if classes_per_client == 0:
    idcs = np.random.permutation(n_data//n_clients*n_clients).reshape(n_clients, -1)
  else:
    data_per_client = n_data // n_clients
    data_per_client_per_class = data_per_client // classes_per_client

    idcs = []
    for i in range(n_clients):
      client_idcs = []
      budget = data_per_client
      c = np.random.randint(n_labels)
      while budget > 0:
        take = min(data_per_client_per_class, len(label_idcs[c]), budget)
        
        client_idcs += label_idcs[c][:take]
        label_idcs[c] = label_idcs[c][take:]
        
        budget -= take
        c = (c + 1) % n_labels

      idcs += [client_idcs]


  print_split(idcs, labels)

  return idcs



def split_dirichlet(labels, n_clients, n_data, alpha, double_stochstic=True):
    '''Splits data among the clients according to a dirichlet distribution with parameter alpha'''

    np.random.seed(0)

    if isinstance(labels, torch.Tensor):
      labels = labels.numpy()
    n_classes = np.max(labels)+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    if double_stochstic:
      label_distribution = make_double_stochstic(label_distribution)

    class_idcs = [np.argwhere(np.array(labels)==y).flatten() 
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    print_split(client_idcs, labels)
  
    return client_idcs


def make_double_stochstic(x):
    rsum = None
    csum = None

    n = 0 
    while n < 1000 and (np.any(rsum != 1) or np.any(csum != 1)):
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)
        n += 1

    #x = x / x.sum(axis=0).reshape(1,-1)
    return x



def print_split(idcs, labels):
  n_labels = np.max(labels) + 1 
  print("Data split:")
  splits = []
  for i, idccs in enumerate(idcs):
    split = np.sum(np.array(labels)[idccs].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
    splits += [split]
    if len(idcs) < 30 or i < 10 or i>len(idcs)-10:
      print(" - Client {}: {:55} -> sum={}".format(i,str(split), np.sum(split)), flush=True)
    elif i==len(idcs)-10:
      print(".  "*10+"\n"+".  "*10+"\n"+".  "*10)

  print(" - Total:     {}".format(np.stack(splits, axis=0).sum(axis=0)))
  print()




class AddChannels(object):
  def __init__(self, n_channels=3):
    self.n_channels = n_channels
  def __call__(self, x):
    return torch.cat([x]*self.n_channels, dim=0)



class IdxSubset(torch.utils.data.Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return (*self.dataset[self.indices[idx]], idx)

    def __len__(self):
        return len(self.indices)

class DataMerger(object):
    def __init__(self, datasets, **kwargs):
        assert isinstance(datasets, dict) and (
            "base" in datasets.keys()
        ), 'Please initialize the DataloaderMerger with a dict of names and dataloaders and atleast one loader called "base"'

        self.datasets = datasets
        self.kwargs = kwargs
        self.used_data_sources = list(datasets.keys())

        if callable(kwargs["mixture_coefficients"]):
          self.mixture_coefficients = kwargs["mixture_coefficients"]
        elif isinstance(kwargs["mixture_coefficients"], dict) and sum(kwargs["mixture_coefficients"].values()) == 1:
          self.mixture_coefficients = lambda x: kwargs["mixture_coefficients"]
        elif kwargs["mixture_coefficients"] == 'linear':
          self.mixture_coefficients = lambda x: {'base': 1 - x/(kwargs['communication_rounds']) , 'public': x/(kwargs['communication_rounds'])}
        else:
          raise NotImplementedError('mixture_coefficients must be function or static distribution')

        self.update(c_round=0)

    def update(self, c_round=0):
      self.loaders = {}
      coeffs = self.mixture_coefficients(c_round)
      if len(self.datasets)==1:
        coeffs = {"base" : 1.0}
      for key in self.used_data_sources:
        if coeffs[key]>0:
          self.loaders[key] = DataLoader(self.datasets[key], batch_size=int(coeffs[key]*self.kwargs["batch_size"]), shuffle=True, pin_memory=True)
      

    def __setitem__(self, key, value):
        self.loaders[key] = value

    def __iter__(self):
        def loop(iterable):
            it = iterable.__iter__()

            while True:
                try:
                    yield it.next()
                except StopIteration:
                    it = iterable.__iter__()
                    yield it.next()

        self.iters = [iter(self.loaders["base"])]
        self.iters += [
            loop(self.loaders[loader_name])
            for loader_name in self.loaders.keys()
            if loader_name != "base"
        ]
        return self

    def __len__(self):
      return sum(1 for _ in self)

    def __next__(self):
      try:
        x, y, source, index = None, None, None, None
        if len(self.iters) > 1:
            x, y, index = zip(*(next(iterator) for iterator in self.iters))
            source = torch.cat(
                [i * torch.ones(subbatch.size(0)) for i, subbatch in enumerate(x)]
            )
            x = torch.cat(x)
            y = torch.cat(y)
            index = torch.cat(index)
        else:
            x, y, index = next(self.iters[0])
            source = torch.zeros(x.size(0))
        return x, y, source, index
      except:
        raise StopIteration
