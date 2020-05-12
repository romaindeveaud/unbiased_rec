import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from collections import ChainMap
from pathlib     import Path


DATA_FOLDER = 'data'
OUTPUT_FOLDER = 'datasets'

def merge_dicts(*dict_args):
  """
  Given any number of dicts, shallow copy and merge into a new dict,
  precedence goes to key value pairs in latter dicts.
  """
  result = {}
  for dictionary in dict_args:
      result.update(dictionary)
  return result

def _get_user_from_session(session_file):
  """
  Parse a *_session.dat file and retrieve the user id related to
  each session id.
  """
  session2user = {}

  with(open(session_file,'r')) as f:
    for line in f:
      atts = line.rstrip('\n').split(' ')

      session2user[int(atts[0])] = int(atts[1])

  return session2user



def process_click_file(click_file,model,session2user,sample_users=3000):
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
    last_click_ = _df[i*49:i*49+49]['click_timestamp'].argmax()

    # Building a temporary dataframe containing the ranking up to the last
    # clicked document.
    _temp = _df[i*49:i*49+49].drop(list(range(i*49+(last_click_+1),i*49+49)))
    # Clicks are 1s, non-clicks are -1s.
    _temp.loc[_temp.click_timestamp > 0,'click_timestamp'] = 1

    user_ids = np.concatenate((user_ids,[ session2user[x] for x in _temp.session_id.values ]))
    item_ids = np.concatenate((item_ids, _temp.item_id.values))
    ratings  = np.concatenate((ratings, _temp.click_timestamp.values))

  np.save(Path(OUTPUT_FOLDER + '/' + model + '/' + day + '_user_ids.npy'),user_ids)
  np.save(Path(OUTPUT_FOLDER + '/' + model + '/' + day + '_item_ids.npy'),item_ids)
  np.save(Path(OUTPUT_FOLDER + '/' + model + '/' + day + '_ratings.npy'),ratings)


s2us = [ _get_user_from_session(session_file) for session_file in Path(DATA_FOLDER).glob('*_sessions.dat') ]
session2user = dict(ChainMap(*s2us))

#model = 'sequential_exposure_explicit'
model = 'sequential_exposure_explicit_sample_top3k'

Parallel(n_jobs=7)(delayed(process_click_file)(click_file,model,session2user,3000) for click_file in Path(DATA_FOLDER).glob('*_clicks.dat'))
