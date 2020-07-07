import config

from pathlib     import Path
from collections import ChainMap

from utils.session_ranking import SessionRanking


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


def get_session2user():
  s2us = [ _get_user_from_session(session_file) for session_file in Path(config.DATA_FOLDER).glob('*_sessions.dat') ]
  session2user = dict(ChainMap(*s2us))

  return session2user


def _get_sequential_topk_rankings(df):
  rankings = []

  for i in range(int(len(df) / 49)):
    # Accessing the dataframe by index is faster than grouping by session_id.

    # Identifying the index of the last click for the i-th session.
    last_click_ = df[i * 49:i * 49 + 49]['click_timestamp'].idxmax()

    # Building a temporary dataframe containing the ranking up to the last
    # clicked document.
    _temp = df[i * 49:i * 49 + 49].drop(list(range(last_click_ + 1, i * 49 + 49)))
    # Clicks are 1s, non-clicks are 0s.
    _temp.loc[_temp.click_timestamp > 0, 'click_timestamp'] = 1
    _temp.loc[_temp.click_timestamp <= 0, 'click_timestamp'] = 0

    session_id = _temp.iloc[0].session_id
    user_id = _temp.iloc[0].user_id

    # Storing the relevant information to a session in a single object
    ranking = SessionRanking(session_id, user_id)

    for _, x in _temp.iterrows():
      ranking.add_interaction(x['item_id'], x['rank'], x['score'], x['click_timestamp'])

    rankings.append(ranking)

  return rankings
