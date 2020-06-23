class Interaction:
  def __init__(self, doc_id, rank, policy_score, click):
    self.doc_id = doc_id
    self.rank = rank
    self.click = click
    self.policy_score = policy_score


class SessionRanking:
  def __init__(self, session_id, user_id):
    self.session_id = session_id
    self.user_id = user_id
    self.interactions = []

  def add_interaction(self, doc_id, rank, policy_score, click):
    self.interactions.append(Interaction(doc_id, rank, policy_score, click))

  def print_ranking(self):
    print('Session ID: {}, user ID: {}'.format(self.session_id, self.user_id))
    for interaction in self.interactions:
      print('{} Document: {}. Click: {}'.format(interaction.rank, interaction.doc_id, interaction.click))

  # TODO: define a function that produces pytorch-ready tensors for training.
