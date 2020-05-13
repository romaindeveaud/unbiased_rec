import config

from pathlib     import Path
from collections import ChainMap

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
