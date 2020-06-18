import torch
import torch.nn as nn
import numpy as np

import click
import pickle

import datetime
import config

from pathlib import Path

from lib.NeuralMatrixFactorization import BiLinearNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
devices = [ 'cuda:{}'.format(i) for i in range(torch.cuda.device_count()) ]

def train(user_ids,item_ids,ratings,num_dimensions,num_epochs,batch_size,verbose,train_split=.75):
  num_users = np.unique(user_ids).shape[0]
  num_items = np.unique(item_ids).shape[0]

  m = BiLinearNet(num_users,num_items,num_dimensions)
  m.to(device)

  optimiser = torch.optim.Adam(
      m.parameters(),
      weight_decay=0,
      lr=1e-2
      )
  loss_function = torch.nn.BCEWithLogitsLoss()
  loss_function.to(device)

  ratings[ratings < 0] = 0

  user_ids_tensor = torch.from_numpy(user_ids).to(device)
  item_ids_tensor = torch.from_numpy(item_ids).to(device)
  ratings_tensor  = torch.from_numpy(ratings).to(device)

  train_test_split = int(user_ids_tensor.size(0) * train_split)

  shuffled_indices = torch.randperm(user_ids_tensor.size(0))
  train_indices = shuffled_indices[:train_test_split]
  test_indices  = shuffled_indices[train_test_split:]

  test_interactions = {
      'user_ids' : user_ids_tensor[test_indices],
      'item_ids' : item_ids_tensor[test_indices],
      'ratings'  : ratings_tensor[test_indices]
      }

  user_ids_tensor = user_ids_tensor[train_indices]
  item_ids_tensor = item_ids_tensor[train_indices]
  ratings_tensor  = ratings_tensor[train_indices]

  m.train()

  for epoch in range(num_epochs):
    epoch_losses = []

    for i in range(0,user_ids_tensor.size(0),batch_size):
      batch_user_ids = user_ids_tensor[i:i+batch_size]
      batch_item_ids = item_ids_tensor[i:i+batch_size]
      batch_ratings  = ratings_tensor[i:i+batch_size]

      optimiser.zero_grad()

      y_pred = m(batch_user_ids,batch_item_ids)

      loss = loss_function(y_pred,batch_ratings)
      epoch_losses.append(loss.item())
      loss.backward()

      optimiser.step()

    if verbose:
      print('{} Epoch {}: loss {}'.format(datetime.datetime.strftime(datetime.datetime.now(),"[%x %X]"),epoch, np.mean(epoch_losses)))


  # Test the embeddings quality on held-out interactions
  m.eval()
  y_test_pred = m(test_interactions['user_ids'],test_interactions['item_ids'])
  
  mse_loss = torch.nn.MSELoss()
  test_mse = mse_loss(torch.sigmoid(y_test_pred),test_interactions['ratings'])
  test_bce = loss_function(y_test_pred,test_interactions['ratings'])

  if verbose:
    print("Test MSE: {} | Test BCE: {}".format(test_mse,test_bce))

  with open(Path('output/{}k.res'.format(num_dimensions)),'a+') as f:
    f.write('{},{}\n'.format(test_mse,test_bce))


  return m.user_embeddings.weight.data.cpu().numpy()


def train_all(num_dimensions,num_epochs,batch_size,verbose):
  """
  Concatenate the interactions for the entire dataset, before
  feeding it to the bilinear network.
  """
  user_ids = []
  item_ids = []
  ratings  = []

  if verbose:
    print('{} Loading interaction data...'.format(datetime.datetime.strftime(datetime.datetime.now(),"[%x %X]")))


  for f in Path( config.DATASET_PATH ).glob('*_user_ids.npy'):
    day = f.name.split('_')[0]
    
    _u = np.load(f)
    _i = np.load( config.DATASET_PATH + day + '_item_ids.npy' )
    _r = np.load( config.DATASET_PATH + day + '_ratings.npy' )

    user_ids.append(_u)
    item_ids.append(_i)
    ratings.append(_r)


  _, cat_item_ids = np.unique(np.concatenate(item_ids),return_inverse=True)
  _, cat_user_ids = np.unique(np.concatenate(user_ids),return_inverse=True)
  ratings = np.concatenate(ratings)

  if verbose:
    print('{} Done...'.format(datetime.datetime.strftime(datetime.datetime.now(),"[%x %X]")))

  user_embeddings = train(cat_user_ids,cat_item_ids,ratings,num_dimensions,num_epochs,batch_size,verbose)

  id2real_user_id = { x:int(_[x]) for x in np.unique(cat_user_ids) }
  with open(Path(config.DATASET_PATH + 'all_id2real_user_id.pkl'), 'wb') as out:
    pickle.dump(id2real_user_id,out)

  np.save(Path(config.DATASET_PATH + 'all_pu_k'+str(num_dimensions)+'.npy'),user_embeddings)


def train_single_day(day,num_dimensions,num_epochs,batch_size,verbose):
  user_ids = np.load(Path(config.DATASET_PATH + day + '_user_ids.npy'))
  item_ids = np.load(Path(config.DATASET_PATH + day + '_item_ids.npy'))
  ratings  = np.load(Path(config.DATASET_PATH + day + '_ratings.npy'))

  _u, cat_user_ids = np.unique(user_ids,return_inverse=True)
  _i, cat_item_ids = np.unique(item_ids,return_inverse=True)

  user_embeddings = train(cat_user_ids,cat_item_ids,ratings,num_dimensions,num_epochs,batch_size,verbose)

  id2real_user_id = { x:int(_u[x]) for x in np.unique(cat_user_ids) }
  with open(Path(config.DATASET_PATH + day + '_id2real_user_id.pkl'), 'wb') as out:
    pickle.dump(id2real_user_id,out)

  np.save(Path(config.DATASET_PATH + day + '_pu_k'+str(num_dimensions)+'.npy'),user_embeddings)


@click.command()
@click.option('--all','-a','tr_all',type=bool,is_flag=True,help='Train user embeddings for all days available in data.')
@click.option('--day','-d','day',type=str,help='Train embeddings for a single day. Provide a date to the yyyymmdd format. Example: 20191119')
@click.option('--num_dim','-k','dim',type=int,help='The number of dimensions of the embeddings.',default=32,show_default=True,required=True)
@click.option('--num_epochs','-e','epochs',type=int,help='The number of training epochs.', default=10, show_default=True, required=True)
@click.option('--batch_size','-b','batch_size',type=int,help='Mini-batch size for each iteration of SGD.', default=256, show_default=True, required=True)
@click.option('--verbose','-v','verbose',default=False,is_flag=True,type=bool,show_default=True)
def parse(tr_all,day,dim,epochs,batch_size,verbose):
  if not day and not tr_all:
    print('Please specify a day for which user embeddings have to be trained, or train all days at once. Type --help for details.')
  elif day:
    train_single_day(day,dim,epochs,batch_size,verbose)
  elif tr_all:
    train_all(dim,epochs,batch_size,verbose)

if __name__ == '__main__':
  parse()
