import numpy as np
import pandas as pd

from joblib import Parallel, delayed

import glob
import os

from collections import ChainMap

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



def process_click_file(click_file,model,session2user):
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
  day = os.path.basename(click_file).split('_')[0]

  if os.path.isfile('datasets/'+model+'/'+day+'_user_ids.npy'):
    return

  _df = pd.read_csv(click_file,sep=' ',names=['session_id','item_id','rank','score','click_timestamp'])
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
    # Clicks are 1s, non-clicks are 0s.
    _temp.loc[_temp.click_timestamp > 0,'click_timestamp'] = 1

    user_ids = np.concatenate((user_ids,[ session2user[x] for x in _temp.session_id.values ]))
    item_ids = np.concatenate((item_ids, _temp.item_id.values))
    ratings  = np.concatenate((ratings, _temp.click_timestamp.values))

  np.save('datasets/'+model+'/'+day+'_user_ids.npy',user_ids)
  np.save('datasets/'+model+'/'+day+'_item_ids.npy',item_ids)
  np.save('datasets/'+model+'/'+day+'_ratings.npy',ratings)


s2us = [ _get_user_from_session(session_file) for session_file in glob.glob('data/*_sessions.dat') ]
session2user = dict(ChainMap(*s2us))

model = 'sequential_exposure_explicit'

Parallel(n_jobs=7)(delayed(process_click_file)(click_file,model,session2user) for click_file in glob.glob('data/*_clicks.dat'))
