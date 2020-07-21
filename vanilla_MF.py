import numpy as np
import pandas as pd

import logging
import click
import config
import pickle

from pathlib import Path

from utils import utils
from utils.session_ranking import RankingDataset


def ndcg_(pred_ranking, true_ranking):
  _pred = np.array(pred_ranking)
  _true = np.array(true_ranking)

  pred_with_labels = _true[_pred.argsort()][::-1]

  idcg = 0.0
  dcg  = 0.0

  for i, _ in enumerate(range(len(_true[_true == 1]))):
    idcg += 1/np.log2(i+2)

  for i, r in enumerate(pred_with_labels):
    dcg += (2**r - 1)/np.log2(i+2)

  return dcg/idcg


def mrr_(pred_ranking, true_ranking):
  _pred = np.array(pred_ranking)
  _true = np.array(true_ranking)

  pred_with_labels = _true[_pred.argsort()][::-1]

  min_rel_rank = pred_with_labels.argmax()+1

  return 1./min_rel_rank


def recall_(pred_ranking, true_ranking, cutoff=5):
  _pred = np.array(pred_ranking)
  _true = np.array(true_ranking)

  pred_with_labels = _true[_pred.argsort()][::-1][:cutoff]

  return len(pred_with_labels[pred_with_labels == 1])/len(_true[_true == 1])


class ExplicitMF:
  def __init__(self, num_dim, num_epochs, num_users, num_items, learning_rate=.005, reg=.02):
    self.num_dim = num_dim
    self.num_epochs = num_epochs
    self.num_users = num_users
    self.num_items = num_items

    self.user_embeddings = np.random.normal(0, 1/self.num_dim, (self.num_users, self.num_dim))
    self.item_embeddings = np.random.normal(0, 1/self.num_dim, (self.num_items, self.num_dim))
    self.user_bias = np.zeros(self.num_users, dtype=np.double)
    self.item_bias = np.zeros(self.num_items, dtype=np.double)

    self.real_item2internal_id = {}
    self.real_user2internal_id = {}

    self.learning_rate  = learning_rate
    self.regularisation = reg

  def train(self, dataset):
    """ Applying Stochastic Gradient Descent to train a Matrix Factorisation model.
    """
    for epoch in range(self.num_epochs):
      logging.info('Processing epoch {}.'.format(epoch + 1))

      for ranking in dataset:
#        ranking.print_ranking()
        for interaction in ranking.interactions:
          if interaction.doc_id not in self.real_item2internal_id:
            self.real_item2internal_id[interaction.doc_id] = len(self.real_item2internal_id)
          if ranking.user_id not in self.real_user2internal_id:
            self.real_user2internal_id[ranking.user_id] = len(self.real_user2internal_id)

          u = self.real_user2internal_id[ranking.user_id]
          i = self.real_item2internal_id[interaction.doc_id]
          click = interaction.click
          rank  = interaction.rank

          ips_click = click / config.observation_propensities[rank-1]

          dot_ = np.dot(self.user_embeddings[u], self.item_embeddings[i])
          err  = ips_click - (dot_ + self.user_bias[u] + self.item_bias[i])

          # Update step
          self.user_bias[u] += self.learning_rate * (err - self.regularisation * self.user_bias[u])
          self.item_bias[i] += self.learning_rate * (err - self.regularisation * self.item_bias[i])

          pu = self.user_embeddings[u]
          qi = self.item_embeddings[i]
          self.user_embeddings[u] += self.learning_rate * (err * qi - self.regularisation * pu)
          self.item_embeddings[i] += self.learning_rate * (err * pu - self.regularisation * qi)

  def predict(self, user_id, item_id):
    u = self.real_user2internal_id[user_id]
    i = self.real_item2internal_id[item_id]

    return np.dot(self.user_embeddings[u], self.item_embeddings[i]) + self.user_bias[u] + self.item_bias[i]

  def test(self, test_dataset):
    ndcgs   = []
    rrs     = []
    recalls = []

    for test_ranking in test_dataset:
      pred_ranking = []
      true_ranking = []
      for test_int in test_ranking.interactions:
        if test_int.doc_id not in self.real_item2internal_id:
          self.real_item2internal_id[test_int.doc_id] = len(self.real_item2internal_id)
        if test_ranking.user_id not in self.real_user2internal_id:
          self.real_user2internal_id[test_ranking.user_id] = len(self.real_user2internal_id)

        y_hat = self.predict(test_ranking.user_id, test_int.doc_id)
        pred_ranking.append(y_hat)
        true_ranking.append(test_int.click)

      pred_ranking = np.array(pred_ranking)
      true_ranking = np.array(true_ranking)

#      ndcg = ndcg_(pred_with_labels)
      ndcg   = ndcg_(pred_ranking, true_ranking)
      rr     = mrr_(pred_ranking, true_ranking)
      recall = recall_(pred_ranking, true_ranking, cutoff=5)
      ndcgs.append(ndcg)
      rrs.append(rr)
      recalls.append(recall)

    logging.info('\nResults for K={}'.format(self.num_dim))
    logging.info('\tMRR: {} | Recall@5: {} | nDCG: {}'.format(np.mean(rrs), np.mean(recalls), np.mean(ndcgs)))


def _split_rankings_train_test(session_rankings, train_test_split, is_random=True):
  split_index = int(len(session_rankings) * train_test_split)

  if is_random:
    np.random.shuffle(session_rankings)

  return session_rankings[:split_index], session_rankings[split_index:]


def train_mf(file, train_test_split, num_dimensions, num_epochs):
  logging.info('Training basic MF with K={}. Input file: {}'.format(num_dimensions, file))

  outfile = Path(config.DATASET_OUTPUT_FOLDER + Path(file).stem + '_sessions_nosampling.pkl')

  if not outfile.is_file():
    logging.info('Loading session2user...')
    session2user = utils.get_session2user()
    logging.info('Completed.')

    logging.info('Loading clicks file...')
    _df = pd.read_csv(file, sep=' ', names=['session_id', 'item_id', 'rank', 'score', 'click_timestamp'])
    _df['user_id'] = _df.session_id.apply(lambda x: session2user[x])
    logging.info('Completed.')

    logging.info('Gathering clicks up to the last clicked in each session...')
    rankings = utils.get_sequential_topk_rankings(_df)
    pickle.dump(rankings, open(outfile, 'wb'))
    logging.info('Completed.')
  else:
    rankings = pickle.load(open(outfile, 'rb'))

  _ = RankingDataset(rankings)
  num_users, num_items = _.num_users, _.num_items
  del _

  train_, test_ = _split_rankings_train_test(rankings, train_test_split, is_random=False)

  logging.info('Number of users: {}; number of items: {}.'.format(num_users, num_items))
  logging.info('Training/test sets are composed of {}/{} sessions.'.format(len(train_), len(test_)))

  model = ExplicitMF(num_dimensions, num_epochs, num_users, num_items)
  model.train(train_)
  model.test(test_)


@click.command()
@click.option('--file', '-f', 'file', type=str, default=Path(config.DATA_FOLDER + '20191001_v2_clicks.dat'))
@click.option('--train_test_split', '-s', 'train_test_split', type=float, default=.8)
@click.option('--num_dimensions', '-k', 'num_dimensions', type=int, default=12)
@click.option('--num_epochs', '-e', 'num_epochs', type=int, default=2)
def parse(file, train_test_split, num_dimensions, num_epochs):
  train_mf(file, train_test_split, num_dimensions, num_epochs)


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  parse()
