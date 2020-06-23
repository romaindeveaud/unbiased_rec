import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from pathlib import Path

import click
import config
import pickle

from utils import utils
from utils.session_ranking import  SessionRanking


def process_click_file_session_grouped(click_file, output_directory, session2user, sample_users):
  day = click_file.name.split('_')[0]

  if Path(config.DATASET_OUTPUT_FOLDER + output_directory + day + '.pkl').is_file():
    return

  _df = pd.read_csv(click_file, sep=' ', names=['session_id', 'item_id', 'rank', 'score', 'click_timestamp'])

  if sample_users and sample_users > 0:
    _df['user_id'] = _df.session_id.apply(lambda x: session2user[x])
    sampled_users = _df.groupby('user_id')['session_id'].size().sort_values(ascending=False)[0:sample_users].index.values

    _df = _df[_df.user_id.isin(sampled_users)]

  rankings = []

  for i in range(int(len(_df)/49)):
    # Accessing the dataframe by index is faster than grouping by session_id.

    # Identifying the index of the last click for the i-th session.
    last_click_ = _df[i*49:i*49+49]['click_timestamp'].idxmax()

    # Building a temporary dataframe containing the ranking up to the last
    # clicked document.
    _temp = _df[i*49:i*49+49].drop(list(range(last_click_+1, i*49+49)))
    # Clicks are 1s, non-clicks are 0s.
    _temp.loc[_temp.click_timestamp > 0, 'click_timestamp'] = 1
    _temp.loc[_temp.click_timestamp <= 0, 'click_timestamp'] = 0

    session_id = _temp.iloc[0].session_id
    user_id    = session2user[session_id]

    # Storing the relevant information to a session in a single object
    ranking = SessionRanking(session_id, user_id)

    for x in _temp.iterrows():
      ranking.add_interaction(x.item_id, x.rank, x.score, x.click_timestamp)

    rankings.append(ranking)

  with open(Path(config.DATASET_OUTPUT_FOLDER + output_directory + day + '.pkl'), 'wb') as f:
    pickle.dump(rankings, f)


def process_click_file(click_file, model, session2user, sample_users=3000):
  """
  Processing a click file, containing all the interactions for a single day.
  
  This function the implicit feedback (observed clicks over the first 49
  ranked documents) into explicit feedback.
  For each session, the top K documents are considered as observed by the
  user, with the document at rank K being the last clicked.
  From these top K documents, clicks are considered positive feedback and
  non-clicks are considered negative.

  These interactions are stored into 3 numpy arrays that are serialised.
  """
  day = click_file.name.split('_')[0]

  if Path('datasets/'+model+'/'+day+'_user_ids.npy').is_file():
    return

  _df = pd.read_csv(click_file,sep=' ',names=['session_id','item_id','rank','score','click_timestamp'])

  if sample_users and sample_users > 0:
    _df['user_id'] = _df.session_id.apply(lambda x: session2user[x])
    sampled_users = _df.groupby('user_id')['session_id'].size().sort_values(ascending=False)[0:sample_users].index.values

    _df = _df[_df.user_id.isin(sampled_users)]
  
  print("{}, {} sessions.".format(click_file,int(len(_df)/49)))

  user_ids = np.array([])
  item_ids = np.array([])
  ratings  = np.array([])

  for i in range(int(len(_df)/49)):
    # Accessing the dataframe by index is faster than grouping by session_id.

    # Identifying the index of the last click for the i-th session.
    last_click_ = _df[i*49:i*49+49]['click_timestamp'].idxmax()

    # Building a temporary dataframe containing the ranking up to the last
    # clicked document.
    _temp = _df[i*49:i*49+49].drop(list(range(last_click_+1,i*49+49)))
    # Clicks are 1s, non-clicks are -1s.
    _temp.loc[_temp.click_timestamp > 0,'click_timestamp'] = 1

    user_ids = np.concatenate((user_ids,[ session2user[x] for x in _temp.session_id.values ]))
    item_ids = np.concatenate((item_ids, _temp.item_id.values))
    ratings  = np.concatenate((ratings, _temp.click_timestamp.values))

  np.save(Path(config.DATASET_OUTPUT_FOLDER + '/' + model + '/' + day + '_user_ids.npy'),user_ids)
  np.save(Path(config.DATASET_OUTPUT_FOLDER + '/' + model + '/' + day + '_item_ids.npy'),item_ids)
  np.save(Path(config.DATASET_OUTPUT_FOLDER + '/' + model + '/' + day + '_ratings.npy'),ratings)


@click.command()
@click.option('--session_grouped', '-s', 'session', type=bool, default=False)
@click.option('--num_sampled_users', '-n', 'num_users', type=int, default=3000)
def parse(session, num_users):
  session2user = utils.get_session2user()
  if session:
    output_dir = 'sequential_exposure_explicit_sample_top{}/session_grouped/'.format(num_users)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    Parallel(n_jobs=-1)(delayed(process_click_file_session_grouped)(click_file,output_dir,session2user,num_users) for click_file in Path(config.DATA_FOLDER).glob('*_clicks.dat'))
  else:
    model = 'sequential_exposure_explicit_sample_top{}'.format(num_users)
    Parallel(n_jobs=7)(delayed(process_click_file)(click_file,model,session2user,num_users) for click_file in Path(config.DATA_FOLDER).glob('*_clicks.dat'))
