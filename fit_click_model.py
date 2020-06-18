import sys
sys.path.insert(0, "vendor/PyClick")

import importlib
import click
import config
import pickle

from pathlib import Path

from utils.AIRS_click_model_parser import AIRSClickModelParser

from vendor.PyClick.pyclick.utils.Utils import Utils as pc_utils

modules = {
    'UBM' : 'pyclick.click_models.UBM',
    'DBN' : 'pyclick.click_models.DBN',
    'SDBN' : 'pyclick.click_models.SDBN',
    'DCM' : 'pyclick.click_models.DCM',
    'CCM' : 'pyclick.click_models.CCM',
    'DCTR' : 'pyclick.click_models.CTR',
    'RCTR' : 'pyclick.click_models.CTR',
    'GCTR' : 'pyclick.click_models.CTR',
    'CM' : 'pyclick.click_models.CM',
    'PBM' : 'pyclick.click_models.PBM'
    }


def fit_click_model(search_sessions,cm):
  train_test_split = int(len(search_sessions) * 0.75)
  train_sessions = search_sessions[:train_test_split]
  train_queries = pc_utils.get_unique_queries(train_sessions)

  test_sessions = pc_utils.filter_sessions(search_sessions[train_test_split:], train_queries)
  test_queries = pc_utils.get_unique_queries(test_sessions)

  module = importlib.import_module(modules[cm])
  click_model = getattr(module, cm)()
  
  click_model.train(search_sessions=train_sessions)

  print(click_model.params)

  with open(Path( config.DATASET_OUTPUT_FOLDER + 'click_models/' + cm + '.pkl' ),'wb') as out:
    pickle.dump(click_model,out)


@click.command()
@click.option('--click_file','-f','click_file',type=str,help="The path to an AIRS click file, whose name should be of the following form: yyyymmdd_v2_clicks.dat")
@click.option('--query_users_file','-q','query_users',type=str,help="The path to a query_users serialised array.",default=config.DATASET_PATH + 'all_query_users.pkl',show_default=True,required=True)
@click.option('--all_files','-a','all_f',type=bool,is_flag=True,help="Whether to use all click files contained in the "+config.DATA_FOLDER+" data folder (the value of DATA_FOLDER can be changed in the config.py file).",default=False,show_default=True)
@click.option('--click_model','-c','click_model',type=click.Choice(['UBM','DBN','SDBN','DCM','CCM','DCTR, RCTR, GCTR','CM','PBM']),help="The click model used to fit the data",default='PBM',show_default=True,required=True)
def parse(click_file,query_users,all_f,click_model):
  recommendation_sessions = []

  if all_f:
    for c in Path(config.DATA_FOLDER).glob('*_clicks.dat'): 
      recommendation_sessions += AIRSClickModelParser.parse(c,query_users) 
  else:
    recommendation_sessions = AIRSClickModelParser.parse(click_file,query_users)

  fit_click_model(recommendation_sessions,click_model)

if __name__ == '__main__':
  parse()
