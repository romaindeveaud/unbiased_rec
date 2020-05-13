"""
This file is heavily inspired from https://github.com/markovi/PyClick/blob/master/pyclick/utils/YandexRelPredChallengeParser.py
"""

import pickle

import numpy  as np
import pandas as pd

from pathlib import Path

from vendor.PyClick.pyclick.click_models.task_centric.TaskCentricSearchSession import TaskCentricSearchSession
from vendor.PyClick.pyclick.search_session.SearchResult import SearchResult

from utils import utils

class AIRSClickModelParser:

  @staticmethod
  def parse(click_file_path,query_users_path):
    if not Path(query_users_path).is_file():
      raise FileNotFoundError(
          errno.ENOENT, os.strerror(errno.ENOENT), query_users_path
          )

    # Reversing the query-user file into a dictionary that contains the query id associated to each user.
    user2query = { x:i for i,a in enumerate(pickle.load(open(Path(query_users_path),'rb'))) for x in a }
    session2user = utils.get_session2user()
    
    sessions = []

    _df = pd.read_csv(click_file,sep=' ',names=['session_id','item_id','rank','score','click_timestamp'])

    for session_id,g in _df.groupby('session_id'):
      query   = user2query[session2user[session_id]]
      session = TaskCentricSearchSession(session_id, query)

      clicks = g.click_timestamp.values

      for i,result in enumerate(g.item_id.values):
        result = SearchResult(result, 0)
        
        if clicks[i] > 0:
          result.click = 1

        session.web_results.append(result)

      sessions.append(session)

    return sessions
