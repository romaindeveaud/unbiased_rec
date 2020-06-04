import torch
import torch.nn as nn

class BiLinearNet(nn.Module):

  def __init__(self,num_users,num_items,num_dimensions):
    super().__init__()

    self.user_embeddings = nn.Embedding(num_users,num_dimensions)
    self.item_embeddings = nn.Embedding(num_items,num_dimensions)

    self.user_bias = nn.Embedding(num_users, 1)
    self.item_bias = nn.Embedding(num_items, 1)

    # Gaussian initialisation of the latent factors.
    nn.init.normal_( self.user_embeddings.weight, 0, 1.0/num_dimensions )
    nn.init.normal_( self.item_embeddings.weight, 0, 1.0/num_dimensions )
    nn.init.zeros_( self.user_bias.weight ) 
    nn.init.zeros_( self.item_bias.weight ) 


  def forward(self,user_id,item_id):
    u_emb = self.user_embeddings(user_id).squeeze()
    i_emb = self.item_embeddings(item_id).squeeze()

    u_bias = self.user_bias(user_id).squeeze()
    i_bias = self.item_bias(item_id).squeeze()

    dot_ = (u_emb * i_emb).sum(1)

    return dot_ + u_bias + i_bias
