import pandas as pd

from pathlib import Path

from utils import utils

def _clicks_to_trec_format(input_file):
  session2user = utils.get_session2user()

  df_ = pd.read_csv(input_file, sep=' ', names=['session_id', 'item_id', 'rank', 'score', 'click_timestamp'])
  df_['user_id'] = df_.session_id.apply(lambda x: session2user[x])

  rankings = utils._get_sequential_topk_rankings(df_)

  filename = Path(input_file).stem
  fout = open(Path(filename + '.trec'), 'w')
  fqrels = open(Path(filename + '.qrels'), 'w')

  for ranking in rankings:
    for interaction in ranking.interactions:
      fout.write("{}\tQ0\t{}\t{}\t{}\t{}\n".format(ranking.session_id,
                                                interaction.doc_id,
                                                interaction.rank,
                                                interaction.policy_score,
                                                filename))

      fqrels.write("{} 0 {} {}\n".format(ranking.session_id,
                                       interaction.doc_id,
                                       interaction.click))

