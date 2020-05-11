import torch

import click

from spotlight.interactions import Interactions
from spotlight.factorization.explicit import ExplicitFactorizationModel

from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


DATASET_PATH = './datasets/'


def train(day,verbose):
  user_ids = np.load(Path(DATASET_PATH + day + '_user_ids.npy'))
  item_ids = np.load(Path(DATASET_PATH + day + '_item_ids.npy'))
  ratings  = np.load(Path(DATASET_PATH + day + '_ratings.npy'))

  dataset = Interactions(
    np.array(user_ids,dtype=np.int32),
    np.array(item_ids,dtype=np.int32),
    ratings=np.array(ratings,dtype=np.float32)
  )

  is_cuda_available = False if device == 'cpu' else True

  m = ExplicitFactorizationModel(loss='logistic',use_cuda=is_cuda_available,embedding_dim=50)
  m.fit(dataset,verbose=verbose)

  user_embeddings = model._net.user_embeddings.weight.detach().cpu().numpy()

  np.save(Path(DATASET_PATH + day + '_pu.npy'))


@click.command()
#@click.option('--day','-d',is_flag=True)
@click.option('--day','-d','day',default='20191119',type=str,show_default=True)
@click.option('--verbose','-v','verbose',default=False,type=bool,show_default=True)
#@click.option('--dump-corpus','-c','dump_corpus',is_flag=True)
#@click.option('--train','-t',type=click.Choice(['lda','hdp','author']))
#@click.option('--num-topics','-n','num_topics',default=10,type=int,show_default=True)
def parse(day,verbose):
  train(day,verbose)
#  if build:
#    gensim_parse_corpus(dataset)
#  if dump_corpus:
#    gensim_build_corpus(dataset)
#  if train:
#    gensim_train(dataset,train,num_topics)

if __name__ == '__main__':
  parse()
