import torch


def dcg_(rel_pred, rel_true, cutoff=None):
  """ Computing the Discounted Cumulative Gain (DCG).
      https://en.wikipedia.org/wiki/Discounted_cumulative_gain
  Arguments:
    rel_pred:  tensor of size (batch_size, list_size, 1), containing
               the out scores of the neural network.
    rel_true:  tensor of size (batch_size, list_size), containing the
               relevance judgements of documents per query.
    cutoff:    cutoff value, dicarding documents ranked below this value
               when computing the dcg.
  Returns:
    Tensor of size (batch_size) containing the DCG@cutoff of each
    query.
  """
  dcg = 0

  # order the list before computing dcg
  sorted_pred_idx = torch.argsort(rel_pred.squeeze(-1), dim=1, descending=True)
  ranked_rel_true = torch.gather(rel_true, 1, sorted_pred_idx)

  _dcg = (2 ** ranked_rel_true[:, ] - 1) / torch.log2(torch.arange(ranked_rel_true.size(1), dtype=torch.float) + 2)

  if cutoff is not None:
    _dcg = _dcg[:, 0:cutoff]

  dcg = torch.sum(_dcg, dim=1)

  return dcg


def ndcg_(rel_pred, rel_true, cutoff):
  """ Computing the normalized Discounted Cumulative Gain (nDCG).
  Arguments:
    rel_pred:  tensor of size (batch_size, list_size, 1), containing
               the out scores of the neural network.
    rel_true:  tensor of size (batch_size, list_size), containing the
               relevance judgements of documents per query.
    cutoff:    cutoff value, dicarding documents ranked below this value
               when computing the ndcg.
  Returns:
    Tensor of size (batch_size) containing the nDCG@cutoff of each query.
  """
  idcg = dcg_(rel_true.unsqueeze(-1), rel_true, cutoff)
  idcg[idcg == 0.0] = 1

  return dcg_(rel_pred, rel_true, cutoff) / idcg