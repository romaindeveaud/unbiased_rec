import torch
import click
import pickle

import config

import numpy as np

from spotlight.interactions import Interactions
from spotlight.factorization.explicit import ExplicitFactorizationModel

from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(user_ids,item_ids,ratings,num_dimensions,verbose):
  dataset = Interactions(
    np.array(user_ids,dtype=np.int32),
    np.array(item_ids,dtype=np.int32),
    ratings=np.array(ratings,dtype=np.float32)
  )

  is_cuda_available = False if device.type == 'cpu' else True

  m = ExplicitFactorizationModel(loss='logistic',use_cuda=is_cuda_available,embedding_dim=num_dimensions)
  m.fit(dataset,verbose=verbose)

  user_embeddings = m._net.user_embeddings.weight.detach().cpu().numpy()

  return user_embeddings

def train_all(num_dimensions,verbose):
  """
  Concatenate the interactions for the entire dataset, before
  feeding it to the bilinear network.
  """
  user_ids = []
  item_ids = []
  ratings  = []

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

  user_embeddings = train(cat_user_ids,cat_item_ids,ratings,num_dimensions,verbose)

  id2real_user_id = { x:int(_[x]) for x in np.unique(cat_user_ids) }
  with open(Path(config.DATASET_PATH + 'all_id2real_user_id.pkl'), 'wb') as out:
    pickle.dump(id2real_user_id,out)

  np.save(Path(config.DATASET_PATH + 'all_pu_k'+str(num_dimensions)+'.npy'),user_embeddings)


def train_single_day(day,num_dimensions,verbose):
  user_ids = np.load(Path(config.DATASET_PATH + day + '_user_ids.npy'))
  item_ids = np.load(Path(config.DATASET_PATH + day + '_item_ids.npy'))
  ratings  = np.load(Path(config.DATASET_PATH + day + '_ratings.npy'))

  _u, cat_user_ids = np.unique(user_ids,return_inverse=True)
  _i, cat_item_ids = np.unique(item_ids,return_inverse=True)

  user_embeddings = train(cat_user_ids,cat_item_ids,ratings,num_dimensions,verbose)

  id2real_user_id = { x:int(_u[x]) for x in np.unique(cat_user_ids) }
  with open(Path(config.DATASET_PATH + day + '_id2real_user_id.pkl'), 'wb') as out:
    pickle.dump(id2real_user_id,out)

  np.save(Path(config.DATASET_PATH + day + '_pu_k'+num_dimensions+'.npy'),user_embeddings)


@click.command()
@click.option('--all','-a','train_all',type=bool,is_flag=True,help='Train user embeddings for all days available in data.')
@click.option('--day','-d','day',type=str,help='Train embeddings for a single day. Provide a date to the yyyymmdd format. Example: 20191119')
@click.option('--num_dim','-k','dim',type=int,help='The number of dimensions of the embeddings.',default=50,show_default=True,required=True)
@click.option('--verbose','-v','verbose',default=False,is_flag=True,type=bool,show_default=True)
def parse(train_all,day,dim,verbose):
  if not day and not train_all:
    print('Please specify a day for which user embeddings have to be trained, or train all days at once. Type --help for details.')
  elif day:
    train_single_day(day,dim,verbose)
  elif train_all:
    train_all(dim,verbose)
#    for f in Path(config.DATASET_PATH).glob('*_user_ids.npy'):
#      train(f.name.split('_')[0],dim,verbose)

if __name__ == '__main__':
  parse()
