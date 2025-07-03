import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. 
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 
        pe = torch.zeros(1, max_len, embed_dim)

        # Alternating sin and cos
        for i in range(max_len):
          for j in range(embed_dim):
            if j % 2 != 0:
              pe[0, i, j] = math.cos(i * 10000 ** (-(j-1)/embed_dim))
            else:
              pe[0, i, j] = math.sin(i * 10000 ** (-j/embed_dim))

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape

        output = x + self.pe[0, :S, :]
        output = self.dropout(output)

        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, E = query.shape
        N, T, E = value.shape
            
        H = self.n_head

        keys = self.key(key)
        keys = keys.view((N, T, H, E//H)).transpose(1, 2)
        
        values = self.value(value)
        values = values.view((N, T, H, E//H)).transpose(1, 2)
        
        querys = self.query(query)
        querys = querys.view((N, S, H, E//H)).transpose(1, 2)
        
        produ = querys @ keys.transpose(-2, -1)
        produ = produ / torch.sqrt(torch.cuda.FloatTensor([E//H]))
        if attn_mask is not None:
          produ = produ.masked_fill(attn_mask == 0, -1e10)
        
        sm = torch.softmax(produ, dim=-1)

        sm = self.attn_drop(sm)
        output = sm @ values
        
        output = output.transpose(1, 2).contiguous().view((N, S, E))
        output = self.proj(output)

        return output


