"""
This file is heavily inspired from https://github.com/markovi/PyClick/blob/master/pyclick/utils/YandexRelPredChallengeParser.py
"""

import errno
import os
import pickle

import numpy  as np
import pandas as pd

from pathlib import Path

from vendor.PyClick.pyclick.click_models.task_centric.TaskCentricSearchSession import TaskCentricSearchSession
from vendor.PyClick.pyclick.search_session.SearchResult import SearchResult

import config
from utils import utils

class AIRSClickModelParser:

  @staticmethod
  def parse(click_file_path,query_users_path):
    if not Path(query_users_path).is_file():
      raise FileNotFoundError(
          errno.ENOENT, os.strerror(errno.ENOENT), query_users_path
          )

    # Reversing the query-user file into a dictionary that contains the query id associated to each user.
    user2query = { x:i for i,a in enumerate(np.load(Path(query_users_path),allow_pickle=True)) for x in a }

    session_file = Path( config.DATA_FOLDER + Path(click_file_path).name.split('_')[0] + '_v2_sessions.dat' )
    #session2user = utils.get_session2user()
    session2user = utils._get_user_from_session(session_file)
    
    sessions = []

    _df = pd.read_csv(click_file_path,sep=' ',names=['session_id','item_id','rank','score','click_timestamp'])

    for session_id,g in _df.groupby('session_id'):
      user_id = session2user[session_id]

      if user_id not in user2query:
        # this user is not in our sample users
        continue

      #query   = user2query[session2user[session_id]]
      query   = user2query[user_id]
      session = TaskCentricSearchSession(session_id, query)

      clicks = g.click_timestamp.values

      for i,result in enumerate(g.item_id.values):
        result = SearchResult(result, 0)
        
        if clicks[i] > 0:
          result.click = 1

        session.web_results.append(result)

      sessions.append(session)

    return sessions
