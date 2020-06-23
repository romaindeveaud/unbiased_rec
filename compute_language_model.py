import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter

import config
import click
import pickle

from utils import utils


def _build_query_lm(docs_props, doc_word_counts):
  result = {}
  total = 0.0

  for tup in docs_props:
    doc_id = tup[0]
    propensity = tup[1]

    for word, count in doc_word_counts[doc_id]:
      if word not in result:
        result[word] = 0.0

      total += count
      result += (count / propensity)

  return result, total


def _get_docs_word_counts():
  _df = pd.read_csv(Path(config.DATA_FOLDER + 'corpus_okt.dat'), sep=' ', header=None,
                    names=['doc_id', 'word_id', 'section'])

  doc_words_counts = {}

  for doc_id, g in _df.groupby('doc_id'):
    if doc_id not in doc_words_counts:
      doc_words_counts[doc_id] = Counter({})

    _w, _c = np.unique(g.word_id.values, return_counts=True)

    for word, count in zip(_w, _c):
      if word not in doc_words_counts[doc_id]:
        doc_words_counts[doc_id][word] = 0.0

      doc_words_counts[doc_id][word] += count

  return doc_words_counts


def train_collection():
  collection_lm = sum([v for d, v in _get_docs_word_counts().items()], Counter())
  collection_len = sum(collection_lm.values())

  for word, count in collection_lm.items():
    collection_lm[word] /= collection_len

  with open(config.DATASET_OUTPUT_FOLDER + 'collection_lm.pkl', 'wb') as f:
    pickle.dump(collection_lm, f)


def train_all_queries(num_workers):
  user2query = {
    x: i for i, a in enumerate(np.load(Path(config.DATASET_PATH + 'all_query_users.pkl'), allow_pickle=True))
    for x in a
  }

  num_queries = len(np.unique(list(user2query.values())))

  session2user = utils.get_session2user()

  # Two arrays of dimension `num_queries`, each containing tuples of the form `(doc_id, observation_propensity)`.
  queries_positive_docs = [[]] * num_queries
  queries_negative_docs = [[]] * num_queries

  for click_file in Path(config.DATA_FOLDER).glob('*_v2_clicks.dat'):
    _df = pd.read_csv(click_file, sep=' ', names=['session_id', 'item_id', 'rank', 'score', 'click_timestamp'])

    for i in range(int(len(_df) / 49)):
      last_click_ = _df[i * 49:i * 49 + 49]['click_timestamp'].idxmax()

      _temp = _df[i * 49:i * 49 + 49].drop(list(range(last_click_ + 1, i * 49 + 49)))
      positive_clicks = _temp.loc[_temp.click_timestamp > 0]  # , 'click_timestamp']
      negative_clicks = _temp.loc[_temp.click_timestamp == -1]  # , 'click_timestamp']

      user = session2user[_temp.iloc[0].session_id]

      if user in user2query:
        _query = user2query[user]

        positive_docs = list(zip(positive_clicks.item_id,
                                 config.observation_propensities[positive_clicks['rank'].values - 1]))
        negative_docs = list(zip(negative_clicks.item_id,
                                 config.observation_propensities[negative_clicks['rank'].values - 1]))

        if positive_docs:
          queries_positive_docs[_query] += positive_docs
        if negative_docs:
          queries_negative_docs[_query] += negative_docs

  doc_word_counts = _get_docs_word_counts()
  collection_lm   = pickle.load(open(Path(config.DATASET_OUTPUT_FOLDER + 'collection_lm.pkl'), 'rb'))

  for i in range(num_queries):
    pos_query_lm, nb_pos_words = _build_query_lm(queries_positive_docs[i], doc_word_counts)
    neg_query_lm, nb_neg_words = _build_query_lm(queries_negative_docs[i], doc_word_counts)

    for word, ips_count in pos_query_lm.items():
      # compute the lm probability with Dirichlet smoothing
      pos_query_lm[word] = (pos_query_lm[word] + config.DIRICHLET_MU*collection_lm[word]) / \
                           (nb_pos_words + config.DIRICHLET_MU)
      neg_query_lm[word] = (neg_query_lm[word] + config.DIRICHLET_MU*collection_lm[word]) / \
                           (nb_neg_words + config.DIRICHLET_MU)

    with open(config.DATASET_OUTPUT_FOLDER + 'query_lms/query' + str(i) + '_pos_lm.pkl', 'wb') as f:
      pickle.dump(pos_query_lm, f)
    with open(config.DATASET_OUTPUT_FOLDER + 'query_lms/query' + str(i) + '_neg_lm.pkl', 'wb') as f:
      pickle.dump(neg_query_lm, f)


@click.command()
@click.option('--collection', '-c', 'collection', type=bool, is_flag=True, default=False)
@click.option('--query', '-q', 'query', type=str)
@click.option('--num_workers', '-n', 'num_workers', type=int, default=1)
def parse(collection, query, num_workers):
  if collection:
    train_collection()
  elif query:
    if query == 'all':
      train_all_queries(num_workers)


if __name__ == '__main__':
  parse()
