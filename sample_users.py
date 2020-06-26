import numpy as np
import pandas as pd

from pathlib import Path

import config
import pickle
import operator
import click

from utils import utils


def _get_user_activities():
  outfile = config.DATASET_OUTPUT_FOLDER + 'user_activities.pkl'

  if Path(outfile).is_file():
    return pickle.load(open(outfile, 'rb'))
  else:
    user_activities = {}
    session2user = utils.get_session2user()

    for i, click_file in enumerate(Path(config.DATA_FOLDER).glob('*_clicks.dat')):
      _df = pd.read_csv(click_file, sep=' ', names=['session_id', 'item_id', 'rank', 'score', 'click_timestamp'])

      _df['user_id'] = _df.session_id.apply(lambda x: session2user[x])

      for user, num_sessions in _df.groupby('user_id')['session_id'].nunique().iteritems():
        if user not in user_activities:
          user_activities[user] = 0

        user_activities[user] += num_sessions

    with open(outfile, 'wb') as f:
      pickle.dump(user_activities, f)

    return user_activities


def sample_from_activity(num_users, fraction_top_users):
  user_activities = dict(sorted(_get_user_activities().items(), key=operator.itemgetter(1), reverse=True))
  sorted_users = np.array(list(user_activities.keys()))

  return_array_len = num_users if num_users < len(sorted_users) else len(sorted_users)

  top_users = sorted_users[:int(return_array_len*fraction_top_users)]
  remaining_users = np.random.choice(sorted_users[int(return_array_len*fraction_top_users):], return_array_len - len(top_users), replace=False)

  return np.random.permutation(np.concatenate((top_users, remaining_users)))


@click.command()
@click.option('--num_users', '-n', 'num_users', type=int, required=True)
@click.option('--fraction_top_users', '-f', 'fraction_top_users', type=float, required=True)
def parse(num_users, fraction_top_users):
  if fraction_top_users > 1 or fraction_top_users < 0:
    raise ValueError('fraction_top_users must be within the [0,1] interval. You provided: {}'.format(fraction_top_users))

  users = sample_from_activity(num_users=num_users, fraction_top_users=fraction_top_users)

  np.save(config.DATASET_OUTPUT_FOLDER + 'sampled_users_{}_{}.npy'.format(num_users, fraction_top_users), users)


if __name__ == '__main__':
  parse()
