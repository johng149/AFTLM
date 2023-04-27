import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gelu
from torch.nn import LayerNorm
from torch.nn.init import xavier_uniform_


class StableEmbedding(nn.Module):
    """
    Like nn.Embedding, but it has:
    1. Uniform Xavier initialization
    2. LayerNorm
    """

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.embedding.weight)

    def forward(self, idx):
        return self.layer_norm(self.embedding(idx))
    
class AFTAttention(nn.Module):

    def __init__(self, embed_dimension: int, max_len: int, bias: bool=False):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        self.embed_dimension = embed_dimension
        # Perform causal masking
        self.pos_bias = nn.Parameter(torch.zeros(max_len, max_len), requires_grad=True)
        self.register_buffer("causal_mask", torch.tril(torch.ones(max_len, max_len)).bool())

    def forward(self, x):
        proj = self.c_attn(x)
        query, key, value = proj.chunk(3, -1)
        _, seq_len, _ = query.shape
        selected_mask = self.causal_mask[:seq_len, :seq_len]
        selected_bias = self.pos_bias[:seq_len, :seq_len]
        selected_bias = selected_bias.unsqueeze(0)
        masked_bias = selected_bias.masked_fill(~selected_mask, float('-inf'))

        # normalize k and bias to prevent numerical instability when taking exp
        maxk = key.max(dim=-1, keepdim=True)[0]
        key = key - maxk
        maxpb = masked_bias.max(dim=-1, keepdim=True)[0]
        masked_bias = masked_bias - maxpb

        key = torch.exp(key)
        expbias = torch.exp(masked_bias)
        num = torch.einsum('bij, bjd -> bid', expbias, key * value)
        denom = torch.einsum('bij, bjd -> bid', expbias, key)

        y = torch.sigmoid(query) * (num / denom)
        return self.c_proj(y)
    
class AFTMultiAttention(nn.Module):
    def __init__(self, embedding_dim, block_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # unlike MultiAttention which uses Attention, we use AFTAttention
        # which already does the concatenation of the heads and FFN
        self.attention = AFTAttention(embed_dimension=embedding_dim, max_len=block_size)
        self.layernorm = LayerNorm(embedding_dim)

    def forward(self, x):
        return self.layernorm(x + self.attention(x))
    
class LMAFT(nn.Module):

    def __init__(self, vocab_size, block_size, embedding_dim, layers):
        super().__init__()
        self.kwargs = {
            "vocab_size": vocab_size,
            "block_size": block_size,
            "embedding_dim": embedding_dim,
            "layers": layers
        }
        self.embeddings = StableEmbedding(vocab_size, embedding_dim)
        self.attention = nn.ModuleList([
            AFTMultiAttention(embedding_dim, block_size) for _ in range(layers)])
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx):
        x = self.embeddings(idx)
        for layer in self.attention:
            x = layer(x)
        x = self.lm_head(x)
        return x