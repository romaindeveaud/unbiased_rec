import torch
import numpy as np

import config

from torch.utils.data import Dataset


class Interaction:
  def __init__(self, doc_id, rank, policy_score, click):
    self.doc_id = int(doc_id)
    self.rank = int(rank)
#    self.relevance = click/config.observation_propensities[self.rank-1]
    self.click = int(click)
    self.policy_score = policy_score

    self.internal_doc_id = None
    self.internal_user_id = None

    self.session_id = None


class SessionRanking:
  def __init__(self, session_id, user_id):
    self.session_id = int(session_id)
    self.user_id = int(user_id)
    self.interactions = []

  def add_interaction(self, doc_id, rank, policy_score, click):
    self.interactions.append(Interaction(doc_id, rank, policy_score, click))

  def print_ranking(self):
    print('Session ID: {}, user ID: {}'.format(self.session_id, self.user_id))
    for interaction in self.interactions:
      print('{} Document: {} - Score: {}. Click: {}'.format(interaction.rank, interaction.doc_id,
                                                            interaction.policy_score, interaction.click))


class AbstractRankingDataset(Dataset):
  def __init__(self, session_rankings):
    self.interactions = []
    self.session_offsets = np.empty((len(session_rankings)*2,), dtype=np.int)

    user_ids = {}
    item_ids = {}

    class_counts = [0, 0]
    mean = 0.0

    for i, ranking in enumerate(session_rankings):
      self.session_offsets[2*i] = len(self.interactions)
      for interaction in ranking.interactions:
        if interaction.doc_id not in item_ids:
          item_ids[interaction.doc_id] = len(item_ids) + 1  # start IDs at 1 to avoid collisions with padding
        if ranking.user_id not in user_ids:
          user_ids[ranking.user_id] = len(user_ids) + 1

        interaction.internal_user_id = user_ids[ranking.user_id]
        interaction.internal_doc_id  = item_ids[interaction.doc_id]

        class_counts[interaction.click] += 1
        mean += interaction.click

        interaction.session_id = ranking.session_id
        self.interactions.append(interaction)

      self.session_offsets[(2*i)+1] = len(self.interactions)

    self.class_weights = 1 / torch.Tensor(class_counts)
    #self.mean = [ l*c for l, c in enumerate(class_counts) ]/sum(class_counts)
    self.mean = mean/sum(class_counts)
    self.interactions = np.array(self.interactions)

  def __len__(self):
    return self.len


class InteractionDataset(AbstractRankingDataset):
  def __init__(self, session_rankings):
    super().__init__(session_rankings)

    self.sampler_weights = [ self.class_weights[int(x.click)] for x in self.interactions ]
    self.len = len(self.interactions)

  def __getitem__(self, idx):
    interaction = self.interactions[idx]
    return {
      'item': interaction.internal_doc_id,
      'user': interaction.internal_user_id,
      'label': interaction.click,
      'relevance': float(interaction.click)/config.observation_propensities[interaction.rank-1]
    }


class RankingDataset(AbstractRankingDataset):
  def __init__(self, session_rankings):
    super().__init__(session_rankings)

    self.len = len(session_rankings)

  @staticmethod
  def collate_fn(batch):
    max_size = max(b['size'] for b in batch)

    padded_items  = torch.zeros(len(batch), max_size, dtype=torch.int)
    padded_users  = torch.zeros(len(batch), max_size, dtype=torch.int)
    padded_labels = torch.zeros(len(batch), max_size)
    padded_rels   = torch.zeros(len(batch), max_size, dtype=torch.float)

    _sizes = torch.zeros(len(batch), dtype=torch.int)

    for i, sample in enumerate(batch):
      max_offset = sample['item'].size(0)

      padded_items[i, 0:max_offset]  = sample['item']
      padded_users[i, 0:max_offset]  = sample['user']
      padded_labels[i, 0:max_offset] = sample['label']
      padded_rels[i, 0:max_offset]   = sample['relevance']

      _sizes[i] = int(max_size)

    return {
      'item': padded_items,
      'user': padded_users,
      'label': padded_labels,
      'relevance': padded_rels,
      'size': _sizes
    }

  def __getitem__(self, idx):
    _s = self.session_offsets[idx*2]
    _e = self.session_offsets[(idx*2)+1]

    ranking = self.interactions[_s:_e]

    _items  = []
    _users  = []
    _labels = []
    _rels   = []

    for x in ranking:
      _items.append(x.internal_doc_id)
      _users.append(x.internal_user_id)
      _labels.append(x.click)
      _rels.append(float(x.click)/config.observation_propensities[x.rank-1])

    return {
      'item': torch.IntTensor(_items),
      'user': torch.IntTensor(_users),
      'label': torch.IntTensor(_labels),
      'relevance': torch.FloatTensor(_rels),
      'size': len(ranking)
    }

