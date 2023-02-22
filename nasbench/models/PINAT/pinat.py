import dgl
import math
import numpy as np
from scipy import sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data
from models.PINAT.gatset_conv import GATSetConv_v5 as GATConv


def graph_pooling(inputs, num_vertices):
    out = inputs.sum(1)
    num_vertices = num_vertices.unsqueeze(-1).expand_as(out)

    return torch.div(out, num_vertices)


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, sa_heads, d_model, d_k, d_v, pine_hidden=256, dropout=0.1, pine_heads=2, bench='101'):
        super().__init__()

        sa_heads = 2
        self.n_head = sa_heads
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, sa_heads * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, sa_heads * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, sa_heads * d_v, bias=False)
        self.fc = nn.Linear((sa_heads + pine_heads) * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # pine structure
        self.conv1 = GATConv(d_model, pine_hidden, heads=pine_heads)
        self.lin1 = torch.nn.Linear(d_model, pine_heads * pine_hidden)
        self.conv2 = GATConv(pine_heads * pine_hidden, pine_hidden, heads=pine_heads)
        self.lin2 = torch.nn.Linear(pine_heads * pine_hidden, pine_heads * pine_hidden)
        self.conv3 = GATConv(
            pine_heads * pine_hidden, pine_heads * d_k, heads=pine_heads, concat=False)
        self.lin3 = torch.nn.Linear(pine_heads * pine_hidden, pine_heads * d_k)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.bench = bench
        if self.bench == '201':
            self.proj_func = nn.Linear(4, 6)

    def to_pyg_batch(self, xs, edge_index_list, num_nodes):
        assert xs.shape[0] == len(edge_index_list)
        # assert xs.shape[0] == num_nodes.shape[0]
        assert xs.shape[0] == len(num_nodes)
        data_list = []
        for x, e, n in zip(xs, edge_index_list, num_nodes):
            data_list.append(torch_geometric.data.Data(x=x[:n], edge_index=e))
        batch = torch_geometric.data.Batch.from_data_list(data_list)
        return batch

    def forward(self, q, k, v, edge_index_list, num_nodes, mask=None):
        # PISA
        x = q
        bs = x.shape[0]
        pyg_batch = self.to_pyg_batch(x, edge_index_list, num_nodes)
        x = F.elu(self.conv1(pyg_batch.x, pyg_batch.edge_index) + self.lin1(pyg_batch.x))
        x = F.elu(self.conv2(x, pyg_batch.edge_index) + self.lin2(x))
        x = self.conv3(x, pyg_batch.edge_index) + self.lin3(x)
        x = x.view(bs, -1, x.shape[-1])
        if self.bench == '201':
            x = x.transpose(1, 2)
            x = self.proj_func(x)
            x = x.transpose(1, 2)

        # SA
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        # other layers
        q = torch.cat((x, q), dim=-1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, pine_hidden, dropout=0.1, bench='101'):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, pine_hidden, dropout=dropout, bench=bench)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, edge_index_list, num_nodes, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, edge_index_list, num_nodes, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    """ An encoder model with self attention mechanism. """

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, pos_enc_dim=7, dropout=0.1, n_position=200, bench='101',
            in_features=5, pine_hidden=256, heads=6, linear_input=80):
        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.bench = bench
        if self.bench == '101':
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, d_word_vec)
        elif self.bench == '201':
            self.pos_map = nn.Linear(pos_enc_dim, n_src_vocab + 1)
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, d_word_vec)
            self.proj_func = nn.Linear(4, 6)
        else:
            raise ValueError('No defined NAS bench.')

        # pine structure
        self.conv1 = GATConv(in_features, pine_hidden, heads=heads)
        self.lin1 = torch.nn.Linear(in_features, heads * pine_hidden)
        self.conv2 = GATConv(heads * pine_hidden, pine_hidden, heads=heads)
        self.lin2 = torch.nn.Linear(heads * pine_hidden, heads * pine_hidden)
        self.conv3 = GATConv(
            heads * pine_hidden, linear_input, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(heads * pine_hidden, linear_input)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v,
                         dropout=dropout, pine_hidden=pine_hidden, bench=bench)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def to_pyg_batch(self, xs, edge_index_list, num_nodes):
        assert xs.shape[0] == len(edge_index_list)
        assert xs.shape[0] == len(num_nodes)
        data_list = []
        for x, e, n in zip(xs, edge_index_list, num_nodes):
            data_list.append(torch_geometric.data.Data(x=x[:n], edge_index=e))
        batch = torch_geometric.data.Batch.from_data_list(data_list)
        return batch

    def forward(self, src_seq, pos_seq, operations, edge_index_list, num_nodes, src_mask=None):
        # op emb and pos emb
        enc_output = self.src_word_emb(src_seq)
        if self.bench == '101':
            pos_output = self.embedding_lap_pos_enc(pos_seq)
            enc_output += pos_output
            # enc_output = pos_output
            enc_output = self.dropout(enc_output)
        elif self.bench == '201':
            pos_output = self.pos_map(pos_seq).transpose(1, 2)
            pos_output = self.embedding_lap_pos_enc(pos_output)
            enc_output += pos_output
            enc_output = self.dropout(enc_output)
        else:
            raise ValueError('No defined NAS bench.')

        # PITE
        x = operations
        bs = operations.shape[0]
        pyg_batch = self.to_pyg_batch(x, edge_index_list, num_nodes)
        x = F.elu(self.conv1(pyg_batch.x, pyg_batch.edge_index) + self.lin1(pyg_batch.x))
        x = F.elu(self.conv2(x, pyg_batch.edge_index) + self.lin2(x))
        x = self.conv3(x, pyg_batch.edge_index) + self.lin3(x)
        x = x.view(bs, -1, x.shape[-1])
        if self.bench == '201':
            x = x.transpose(1, 2)
            x = self.proj_func(x)
            x = x.transpose(1, 2)
        enc_output += self.dropout(x)
        enc_output = self.layer_norm(enc_output)

        # backone forward
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, edge_index_list, num_nodes, slf_attn_mask=src_mask)


        return enc_output


class PINATModel(nn.Module):
    """
        PINATModel: Concat(2 self-attention heads, 2 pine heads)
    """
    def __init__(self, adj_type, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner,
                 pad_idx=None, pos_enc_dim=7, linear_hidden=80, pine_hidden=256, bench='101'):
        super(PINATModel, self).__init__()

        # backone
        self.bench = bench
        self.adj_type = adj_type
        self.encoder = Encoder(n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, pad_idx,
                               pos_enc_dim=pos_enc_dim, dropout=0.1, pine_hidden=pine_hidden, bench=bench)

        # regressor
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(d_word_vec, linear_hidden, bias=False)
        self.fc2 = nn.Linear(linear_hidden, 1, bias=False)

    def forward(self, inputs):
        # get arch topology
        numv = inputs["num_vertices"]
        assert self.adj_type == 'adj_lapla'
        adj_matrix = inputs["lapla"].float()
        edge_index_list = []
        for edge_num, edge_index in zip(inputs['edge_num'], inputs['edge_index_list']):
            edge_index_list.append(edge_index[:, :edge_num])

        # backone feature
        out = self.encoder(src_seq=inputs["features"], pos_seq=adj_matrix.float(),
                           operations=inputs['operations'].squeeze(0),
                           num_nodes=numv, edge_index_list=edge_index_list)

        # regressor forward
        out = graph_pooling(out, numv)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out).view(-1)
        return out
