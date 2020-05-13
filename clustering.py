import numpy as np

from pathlib          import Path

import config
import click

import importlib
import pickle

def cluster(num_clusters,model,f):
  user_embeddings = np.load(f)

  module = importlib.import_module('sklearn.cluster')
  class_ = getattr(module, model)
  model = class_(n_clusters=num_clusters).fit(user_embeddings)

  return model


@click.command()
@click.option('--num_clusters','-c','num_clusters',type=int,help="The number of user clusters.",default=100,show_default=True,required=True)
@click.option('--model','-m','model',type=click.Choice(['AgglomerativeClustering','MiniBatchKMeans']),help="The clustering model.",default='AgglomerativeClustering',show_default=True,required=True)
@click.option('--embeddings_file','-e','emb_file',type=str,help="Path to the file containing the serialised user embeddings.", default=config.DATASET_PATH + 'all_pu_k50.npy',show_default=True,required=True)
@click.option('--output_query_users','-q','query_users',type=bool,help="Whether the script should save clusters of users to be further used as 'queries'.",default=True,show_default=True)
def parse(num_clusters,model,emb_file,query_users):
  model = cluster(num_clusters,model,Path(emb_file))

  if query_users:
    # After clustering, we serialise the groups of similar users for further use.
    #
    fileprefix = Path(embfile).name.split('_')[0]
    id2real_user_id = pickle.load(open(Path(config.DATASET_PATH + fileprefix + '_id2real_user_id.pkl'), 'rb'))

    cluster_labels = model.labels_
    qu = [ id2real_user_id[np.where(cluster_labels == i)] for i in range(100) ]

    with open(Path(config.DATASET_PATH + fileprefix + '_query_users.pkl'),'wb') as f:
      pickle.dump(qu,f)

if __name__ == '__main__':
  parse()
