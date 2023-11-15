import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) 
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class TimeSeriesAttentionModel(nn.Module):
    def __init__(self, input_dim, embed_size, heads, num_layers, output_dim, dropout):
        super(TimeSeriesAttentionModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, embed_size)
        self.layers = nn.ModuleList()
        self.heads = heads
        self.head_dim = embed_size // heads
        for _ in range(num_layers):
            self.layers.append(SelfAttention(embed_size, heads))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_size, output_dim)

    def forward(self, x, mask=None):
        query = self.input_linear(x)
        values = self.input_linear(x)        
        keys = self.input_linear(x)
        x = query
        for attention in self.layers:
            x = attention(values, keys, x, mask)
        x = self.drop(x[:, -1, :])
        x = self.fc(x)
        return x
