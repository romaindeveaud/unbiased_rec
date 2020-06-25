import torch

from torch.utils.data import Dataset


class Interaction:
  def __init__(self, doc_id, rank, policy_score, click):
    self.doc_id = int(doc_id)
    self.rank = int(rank)
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

  # TODO: define a function that produces pytorch-ready tensors for training.


class RankingDataset(Dataset):
  def __init__(self, session_rankings):
    self.interactions = []

    user_ids = {}
    item_ids = {}

    class_counts = [0, 0]

    for ranking in session_rankings:
      for interaction in ranking.interactions:
        if interaction.doc_id not in item_ids:
          item_ids[interaction.doc_id] = len(item_ids)
        if ranking.user_id not in user_ids:
          user_ids[ranking.user_id] = len(user_ids)

        interaction.internal_user_id = user_ids[ranking.user_id]
        interaction.internal_doc_id  = item_ids[interaction.doc_id]

        class_counts[interaction.click] += 1

        interaction.session_id = ranking.session_id
        self.interactions.append(interaction)

    self.class_weights = 1 / torch.Tensor(class_counts)
    self.sampler_weights = [ self.class_weights[int(x.click)] for x in self.interactions ]

    self.len = len(self.interactions)

  def __getitem__(self, idx):
    return {
      'item': self.interactions[idx].internal_doc_id,
      'user': self.interactions[idx].internal_user_id,
      'label': self.interactions[idx].click
    }

  def __len__(self):
    return self.len
