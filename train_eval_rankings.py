import torch
import torch.nn as nn
import numpy    as np

from pathlib import Path

import pickle
import config
import click

from torch.utils.tensorboard import SummaryWriter

from utils.session_ranking import RankingDataset
from lib.NeuralMatrixFactorization import BiLinearNet

devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]


def _get_session_rankings(num_users, fraction_top_users):
  return np.concatenate([pickle.load(open(fpath, 'rb')) for fpath in
                         sorted(Path(config.DATASET_OUTPUT_FOLDER +
                                     'sequential_exposure_explicit_sample_{}/session_grouped_{}'.format(num_users,
                                                                                                        fraction_top_users)).glob('*.pkl'))])
  #return { fpath.name.split('.')[0]:pickle.load(open(fpath, 'rb')) for fpath in Path(config.DATASET_PATH + 'session_grouped').glob('*.pkl') }


def _get_dataset_stats(session_rankings):
  user_ids = []
  item_ids = []

  num_users = 0
  num_items = 0

  for ranking in session_rankings:
    if ranking.user_id not in user_ids:
      user_ids.append(ranking.user_id)
      num_users += 1

    for interaction in ranking.interactions:
      if interaction.doc_id not in item_ids:
        item_ids.append(interaction.doc_id)
        num_items += 1

  return num_users, num_items


def evaluate(model, test, loss_function, batch_size, device, writer, step):
  model.eval()

#  sampler = torch.utils.data.sampler.RandomSampler(test)#test.class_weights, test.len)
  testloader = torch.utils.data.DataLoader(test, batch_size=batch_size)#, sampler=sampler)

  losses = []

  for i, batch in enumerate(testloader, 1):
    test_batch_users = batch['user'].to(device, non_blocking=True)
    test_batch_items = batch['item'].to(device, non_blocking=True)
    test_batch_labels = batch['label'].to(device, non_blocking=True)

    y_hat = model(test_batch_users, test_batch_items)

    loss = loss_function(y_hat, test_batch_labels)
    losses.append(loss.item())

    writer.add_scalar('Loss/test', loss.item(), step)
    step += 1

  model.train()

  return losses


def train_svd(num_dimensions, num_epochs, batch_size, gpu_index, test, train_test_split=.75, num_users_sample=10000,
              fraction_top_users=0.66):
  device = torch.device(devices[gpu_index] if torch.cuda.is_available() else 'cpu')

  session_rankings = _get_session_rankings(num_users_sample, fraction_top_users)

  num_users, num_items = _get_dataset_stats(session_rankings)

  print('num_users {}, num_items {}'.format(num_users, num_items))
  print('session_rankings len {}'.format(len(session_rankings)))

  # Shuffle all the interactions.
  # It might be relevant to simply take them all in order to respect the temporal aspect of news recommendation.
  # shuffled_index = np.random.choice(len(session_rankings), len(session_rankings), replace=False)

  # print('shuffled_index: ', shuffled_index)

  # train_index = shuffled_index[:int(len(shuffled_index) * train_test_split)]
  # test_index  = shuffled_index[int(len(shuffled_index) * train_test_split):]

  # print('train_index len: ',len(train_index))
  # print('test_index len: ',len(test_index))

  # train_rankings = RankingDataset(session_rankings[train_index])
  # test_rankings  = RankingDataset(session_rankings[test_index])

  split_index = int(len(session_rankings) * train_test_split)

  train_rankings = RankingDataset(session_rankings[:split_index])
  test_rankings  = RankingDataset(session_rankings[split_index:])

  print('train_rankings len: ', train_rankings.len)
  print('test_rankings len: ', test_rankings.len)

  m = BiLinearNet(num_users, num_items, num_dimensions)
  m.to(device)

  optimiser = torch.optim.Adam(
    m.parameters(),
    weight_decay=0,
    lr=3e-4
  )
  #loss_function = torch.nn.MSELoss()
  loss_function = torch.nn.BCEWithLogitsLoss()#pos_weight=train_rankings.pos_weight)
  loss_function.to(device)

  writer = SummaryWriter('output/runs/BCE-loss-no-weighting')

  step = 0
  test_step = 0
  for epoch in range(num_epochs):
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.empty(train_rankings.len).random_(10), batch_size)
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(train_rankings.sampler_weights, train_rankings.len)
    trainloader = torch.utils.data.DataLoader(train_rankings, batch_size=batch_size)#, sampler=sampler)#, shuffle=True)

    running_loss = 0.0
    for i, batch in enumerate(trainloader, 1):
      batch_users  = batch['user'].to(device, non_blocking=True)
      batch_items  = batch['item'].to(device, non_blocking=True)
      batch_labels = batch['label'].to(device, non_blocking=True)

      optimiser.zero_grad()

      y_hat = m(batch_users, batch_items)
      loss = loss_function(y_hat, batch_labels)

      loss.backward()
      optimiser.step()

      running_loss += loss.item()

      writer.add_scalar('Loss/train', loss.item(), step)
      step += 1
      if i % 200 == 199:  # print every 200 batches
        print('[%d, %5d] training loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 200))
        running_loss = 0.0

    if test:
      # Run an evaluation cycle at the end of each epoch.
      test_losses = evaluate(m, test_rankings, loss_function, batch_size, device, writer, test_step)
      test_step += len(test_losses)
      print("\t Test loss at epoch {}: {} (len {})".format(epoch+1, np.mean(test_losses), len(test_losses)))



@click.command()
@click.option('--model', '-m', 'model', type=click.Choice(['SVD']), default='SVD', show_default=True, required=True, help='A ranking model.')
@click.option('--num_epochs', '-e', 'num_epochs', type=int, default=10, show_default=True, help='Number of training epochs.')
@click.option('--num_dim', '-k', 'dim', type=int, help='The number of dimensions of the embeddings.', default=32,
              show_default=True, required=True)
@click.option('--batch_size', '-b', 'batch_size', type=int, default=256, show_default=True, help='Batch size.')
@click.option('--gpu', '-g', 'gpu_index', type=int, help='The index of the desired GPU.', default=0)
@click.option('--test', '-t', 'test', type=bool, is_flag=True, default=False,
                                      help='Set to True to perform an evaluation'
                                           'right after training.', show_default=True)
def parse(model, num_epochs, dim, batch_size, gpu_index, test):
  if model == 'SVD':
    train_svd(dim, num_epochs, batch_size, gpu_index, test)

if __name__ == '__main__':
  parse()
