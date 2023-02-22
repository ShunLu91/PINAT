import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros


class GATSetConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated. (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(GATSetConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.heads_2 = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # self.weight = Parameter(
        #     torch.Tensor(in_channels, heads * out_channels))
        # self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.a = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.b = Parameter(
            torch.Tensor(1, out_channels, self.heads_2))
        self.e = Parameter(
            torch.Tensor(1, out_channels, self.heads_2))
        self.w = Parameter(
            torch.Tensor(self.heads, out_channels, self.heads_2))
        self.h = Parameter(
            torch.Tensor(self.heads, out_channels))
        self.c = Parameter(
            torch.Tensor(self.heads, out_channels))

        # if bias and concat:
        #     self.bias = Parameter(torch.Tensor(heads * out_channels))
        # elif bias and not concat:
        #     self.bias = Parameter(torch.Tensor(out_channels))
        # else:
        #     self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # glorot(self.weight)
        # glorot(self.att)
        # zeros(self.bias)
        glorot(self.a)
        glorot(self.b)
        glorot(self.w)
        glorot(self.c)
        zeros(self.e)
        zeros(self.h)
        pass

    def forward(self, x, edge_index):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        # x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        x = torch.mm(x, self.a).view(-1, self.heads, self.out_channels)
        x_repeat = x.unsqueeze(-1).repeat(1, 1, 1, self.heads_2)
        x = x_repeat * self.b + self.e
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        return self.propagate(edge_index, x=x, num_nodes=x.size(0))

    # def message(self, x_i, x_j, edge_index, num_nodes):
    def message(self, x_j):
        # # Compute attention coefficients.
        # alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        # alpha = F.leaky_relu(alpha, self.negative_slope)
        # alpha = softmax(alpha, edge_index[0], num_nodes)

        # # Sample attention coefficients stochastically.
        # if self.training and self.dropout > 0:
        #     alpha = F.dropout(alpha, p=self.dropout, training=True)

        # return x_j * alpha.view(-1, self.heads, 1)
        # print(x_j.shape)
        """ x_j_repeat = x_j.unsqueeze(-1).repeat(1, 1, 1, self.heads_2)
        # print(x_j_repeat[0,0,0,:])
        # print(x_j_repeat.shape, self.b.shape, self.e.shape)
        out = x_j_repeat * self.b + self.e
        # print(out.shape)
        out = F.leaky_relu(out, negative_slope=self.negative_slope) """
        return x_j

    def update(self, aggr_out):

        # print(aggr_out.shape)
        out = self.w * aggr_out
        # print(out.shape)
        out = out.sum(-1) + self.h
        out = F.leaky_relu(out, self.negative_slope)
        out = self.c * out
        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.sum(dim=1)
            # out = out.sum(1)
        # print(out.shape)
        return out
        # if self.concat is True:
        #     aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        # else:
        #     aggr_out = aggr_out.mean(dim=1)

        # if self.bias is not None:
        #     aggr_out = aggr_out + self.bias
        # return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GATSetConv_v4(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated. (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(GATSetConv_v4, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t = 8
        self.t2 = 8
        self.t3 = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # self.weight = Parameter(
        #     torch.Tensor(in_channels, heads * out_channels))
        # self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.a = Parameter(
            torch.Tensor(in_channels, self.t))
        self.u = Parameter(
            torch.Tensor(1, self.t2))
        self.v = Parameter(
            torch.Tensor(1, self.t2))

        self.wb = nn.Linear(self.t * self.t2, out_channels * self.t3)
        # self.c = nn.Linear(self.t3, out_channels, bias=False)
        self.c = Parameter(
            torch.Tensor(1, self.out_channels, self.t3))

        self.reset_parameters()

    def reset_parameters(self):
        # glorot(self.weight)
        # glorot(self.att)
        # zeros(self.bias)
        glorot(self.a)
        glorot(self.u)
        glorot(self.wb.weight)
        glorot(self.c)
        zeros(self.v)
        zeros(self.wb.bias)
        pass

    def forward(self, x, edge_index):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        # print(x.shape, edge_index.size(1))
        # x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        # x = torch.mm(x, self.a).view(-1, self.heads, self.out_channels)
        # x_repeat = x.unsqueeze(-1).repeat(1, 1, 1, self.heads_2)
        # x = x_repeat * self.b + self.e
        # x = F.leaky_relu(x, negative_slope=self.negative_slope)
        out = self.propagate(edge_index, x=x, num_nodes=x.size(0))
        # print(out.shape)
        return out

    # def message(self, x_i, x_j, edge_index, num_nodes):
    def message(self, x_j):
        # # Compute attention coefficients.
        x_j = torch.mm(x_j, self.a)
        # print(x_j.shape)
        x_j = x_j.unsqueeze(-1).repeat(1, 1, self.t2)
        # print(x_j.shape)
        x_j = x_j * self.u + self.v
        x_j = F.leaky_relu(x_j, negative_slope=self.negative_slope).view(-1, self.t * self.t2)
        # print(x_j.shape)
        return x_j

    def update(self, aggr_out):
        # print(aggr_out.shape)
        out = self.wb(aggr_out)
        # print(out.shape)
        out = F.leaky_relu(out, negative_slope=self.negative_slope).view(-1, self.out_channels, self.t3)
        # print(out.shape)
        # out = out.permute(0, 2, 1)
        # print(out.shape)
        out = out * self.c
        # print(out.shape)
        # out = self.c(out).view(-1, self.out_channels)
        if self.concat is True:
            out = out.view(-1, self.t3 * self.out_channels)
        else:
            out = out.sum(dim=-1)

        # out = out.sum(-1)
        # print(out.shape)

        return out

    def __repr__(self):
        return '{}({}, {}, t={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.t)


class GATSetConv_v5(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated. (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(GATSetConv_v5, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t = heads
        self.t2 = 1
        self.t3 = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # self.weight = Parameter(
        #     torch.Tensor(in_channels, heads * out_channels))
        # self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.a = Parameter(
            torch.Tensor(in_channels, out_channels * self.t))
        self.u = Parameter(
            torch.Tensor(out_channels, 1, self.t2))
        self.v = Parameter(
            torch.Tensor(out_channels, 1, self.t2))

        # self.wb = nn.Linear(self.t * self.t2, out_channels * self.t3)
        self.w = Parameter(torch.Tensor(out_channels, self.t, self.t2, self.t3))
        self.b = Parameter(torch.Tensor(out_channels, self.t3))
        # self.c = nn.Linear(self.t3, out_channels, bias=False)
        self.c = Parameter(
            torch.Tensor(1, self.out_channels, self.t3))

        self.reset_parameters()

    def reset_parameters(self):
        # glorot(self.weight)
        # glorot(self.att)
        # zeros(self.bias)
        glorot(self.a)
        glorot(self.u)
        # glorot(self.wb.weight)
        glorot(self.w)
        glorot(self.c)
        zeros(self.v)
        # zeros(self.wb.bias)
        zeros(self.b)
        pass

    def forward(self, x, edge_index):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # print(x.shape, edge_index.size(1))
        # x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        # x = torch.mm(x, self.a).view(-1, self.heads, self.out_channels)
        # x_repeat = x.unsqueeze(-1).repeat(1, 1, 1, self.heads_2)
        # x = x_repeat * self.b + self.e
        # x = F.leaky_relu(x, negative_slope=self.negative_slope)
        x = torch.mm(x, self.a).view(-1, self.out_channels, self.t) # n x in_c => n x out_c x t1
        # print(x.shape)
        x = x.unsqueeze(-1)# n x out_c x t1 x 1
        # print(x.shape)
        x = x * self.u + self.v # out_c x 1 x t2
        x = F.leaky_relu(x, negative_slope=self.negative_slope).view(-1, self.out_channels * self.t * self.t2) # n x out_c x t1 x t2
        out = self.propagate(edge_index, x=x, num_nodes=x.size(0))
        # print(out.shape)
        return out

    # def message(self, x_i, x_j, edge_index, num_nodes):
    def message(self, x_j):
        # # # Compute attention coefficients.
        # x_j = torch.mm(x_j, self.a).view(-1, self.out_channels, self.t) # n x in_c => n x out_c x t1
        # # print(x_j.shape)
        # x_j = x_j.unsqueeze(-1)# n x out_c x t1 x 1
        # # print(x_j.shape)
        # x_j = x_j * self.u + self.v # out_c x 1 x t2
        # x_j = F.leaky_relu(x_j, negative_slope=self.negative_slope).view(-1, self.out_channels * self.t * self.t2) # n x out_c x t1 x t2
        # # print(x_j.shape)
        return x_j

    def update(self, aggr_out):
        # print(aggr_out.shape)
        aggr_out = aggr_out.view(-1 , self.out_channels, self.t, self.t2).unsqueeze(-1)
        # print(aggr_out.shape, self.w.shape, self.b.shape)
        out = aggr_out * self.w
        out = out.sum(2).sum(2)
        out = out + self.b
        # out = self.wb(aggr_out).view(-1, self.out_channels)
        # print(out.shape)
        out = F.leaky_relu(out, negative_slope=self.negative_slope)
        # print(out.shape)
        # out = out.permute(0, 2, 1)
        # print(out.shape)
        out = out * self.c
        # print(out.shape)
        # out = self.c(out).view(-1, self.out_channels)
        if self.concat is True:
            out = out.view(-1, self.t3 * self.out_channels)
        else:
            out = out.sum(dim=-1)

        # out = out.sum(-1)
        # print(out.shape)

        return out

    def __repr__(self):
        return '{}({}, {}, t={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.t)

class GATSetConv_v2(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated. (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(GATSetConv_v2, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.heads_2 = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # self.weight = Parameter(
        #     torch.Tensor(in_channels, heads * out_channels))
        # self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.a = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.b = Parameter(
            torch.Tensor(1, out_channels, self.heads_2))
        self.e = Parameter(
            torch.Tensor(1, out_channels, self.heads_2))
        self.w = Parameter(
            torch.Tensor(self.heads, out_channels, self.heads_2))
        self.h = Parameter(
            torch.Tensor(self.heads, out_channels))
        self.c = Parameter(
            torch.Tensor(self.heads, out_channels))
        if concat:
            self.lin = nn.Linear(in_channels, heads * out_channels)
        else:
            self.lin = nn.Linear(in_features=in_channels, out_features=out_channels)

        # if bias and concat:
        #     self.bias = Parameter(torch.Tensor(heads * out_channels))
        # elif bias and not concat:
        #     self.bias = Parameter(torch.Tensor(out_channels))
        # else:
        #     self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # glorot(self.weight)
        # glorot(self.att)
        # zeros(self.bias)
        glorot(self.a)
        glorot(self.b)
        glorot(self.w)
        glorot(self.c)
        zeros(self.e)
        zeros(self.h)
        pass

    def forward(self, x, edge_index):
        """"""
        # edge_index, _ = remove_self_loops(edge_index)
        
        cpx = F.relu(self.lin(x))
        # edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_index, norm = GCNConv.norm(
            edge_index, x.size(0), None, dtype=x.dtype)
        # x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        x = torch.mm(x, self.a).view(-1, self.heads, self.out_channels)
        # print(x.shape)
        x_repeat = x.unsqueeze(-1).repeat(1, 1, 1, self.heads_2)
        x = x_repeat * self.b + self.e
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        # x = F.relu(x)
        x = x * self.w
        x = x.sum(-1) + self.h
        # print(x.shape)
        hidden = x.clone()
        alpha = 0.1
        for _ in range(10):
            # print(x.shape)
            x = self.propagate(edge_index, x=x, norm=norm, num_nodes=x.size(0))
            x = x * (1 - alpha)
            x = x + alpha * hidden
        # print(norm.shape, x.shape)
        # x = norm.view(-1, 1) * x
        out = F.leaky_relu(x, self.negative_slope)
        # out = F.relu(x)
        out = self.c * out
        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.sum(dim=1)
        # out = out + cpx
        # print(norm.shape, out.shape)
        # out = norm.view(-1, 1) * out
        return out

    # def message(self, x_i, x_j, edge_index, num_nodes):
    def message(self, x_j, norm):
        # print(x_j.shape)
        # exit(0)
        # print(x_j.shape)
        x_j = norm.view(-1, 1, 1) * x_j
        
        # print(x_j)
        # # Compute attention coefficients.
        # alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        # alpha = F.leaky_relu(alpha, self.negative_slope)
        # alpha = softmax(alpha, edge_index[0], num_nodes)

        # # Sample attention coefficients stochastically.
        # if self.training and self.dropout > 0:
        #     alpha = F.dropout(alpha, p=self.dropout, training=True)

        # return x_j * alpha.view(-1, self.heads, 1)
        # print(x_j.shape)
        """ x_j_repeat = x_j.unsqueeze(-1).repeat(1, 1, 1, self.heads_2)
        # print(x_j_repeat[0,0,0,:])
        # print(x_j_repeat.shape, self.b.shape, self.e.shape)
        out = x_j_repeat * self.b + self.e
        # print(out.shape)
        out = F.leaky_relu(out, negative_slope=self.negative_slope) """
        return x_j

    def update(self, aggr_out):

        # print(aggr_out.shape)
        # out = self.w * aggr_out
        # # print(out.shape)
        # out = out.sum(-1) + self.h
        # out = F.leaky_relu(aggr_out, self.negative_slope)
        # out = self.c * out
        # if self.concat is True:
        #     out = out.view(-1, self.heads * self.out_channels)
        # else:
        #     out = out.sum(dim=1)
            # out = out.sum(1)
        # print(out.shape)
        # out = F.normalize(out, dim=-1)
        # return out
        # if self.concat is True:
        #     aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        # else:
        #     aggr_out = aggr_out.mean(dim=1)

        # if self.bias is not None:
        #     aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GATSetConv_v3(MessagePassing):
    r"""

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated. (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(GATSetConv_v3, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.heads_2 = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # self.weight = Parameter(
        #     torch.Tensor(in_channels, heads * out_channels))
        # self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.a = Parameter(
            torch.Tensor(out_channels, heads * out_channels))
        self.b = Parameter(
            torch.Tensor(1, out_channels, self.heads_2))
        self.e = Parameter(
            torch.Tensor(1, out_channels, self.heads_2))
        self.w = Parameter(
            torch.Tensor(self.heads, out_channels, self.heads_2))
        self.h = Parameter(
            torch.Tensor(self.heads, out_channels))
        self.c = Parameter(
            torch.Tensor(self.heads, out_channels))
        if concat:
            self.lin = nn.Linear(in_channels, heads * out_channels)
        else:
            self.lin = nn.Linear(in_features=in_channels, out_features=out_channels)

        # if bias and concat:
        #     self.bias = Parameter(torch.Tensor(heads * out_channels))
        # elif bias and not concat:
        #     self.bias = Parameter(torch.Tensor(out_channels))
        # else:
        #     self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # glorot(self.weight)
        # glorot(self.att)
        # zeros(self.bias)
        glorot(self.a)
        glorot(self.b)
        glorot(self.w)
        glorot(self.c)
        zeros(self.e)
        zeros(self.h)
        pass

    def forward(self, x, edge_index):
        """"""
        # edge_index, _ = remove_self_loops(edge_index)
        
        cpx = self.lin(x)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_index, norm = GCNConv.norm(
            edge_index, x.size(0), None, dtype=x.dtype)
        # x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        hidden = cpx.clone()
        alpha = 0.1
        for _ in range(2):
            # print(cpx.shape)
            cpx = self.propagate(edge_index, x=cpx, norm=norm, num_nodes=x.size(0))
            cpx = cpx * (1 - alpha)
            cpx = cpx + alpha * hidden

        # print(norm.shape, x.shape)
        # x = norm.view(-1, 1) * x
        # out = F.leaky_relu(x, self.negative_slope)
        # out = F.relu(x)
        # out = self.c * out
        
        # out = out + cpx
        # print(norm.shape, out.shape)
        # out = norm.view(-1, 1) * out
        return cpx

    # def message(self, x_i, x_j, edge_index, num_nodes):
    def message(self, x_j, norm):
        # print(x_j.shape)
        # exit(0)
        # print(x_j.shape)
        # print(x_j.shape, self.a.shape)
        x_j = torch.mm(x_j, self.a).view(-1, self.heads, self.out_channels)
        x_repeat = x_j.unsqueeze(-1).repeat(1, 1, 1, self.heads_2)
        x_j = x_repeat * self.b + self.e
        x_j = F.leaky_relu(x_j, negative_slope=self.negative_slope)
        x_j = x_j * self.w
        x_j = x_j.sum(-1) + self.h
        x_j = norm.view(-1, 1, 1) * x_j
        return x_j

    def update(self, aggr_out):

        aggr_out = F.leaky_relu(aggr_out, self.negative_slope)
        aggr_out = self.c * aggr_out
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.sum(dim=1)
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 improved=False,
                 cached=False,
                 bias=True):
        super(GCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index = add_self_loops(edge_index, num_nodes)
        loop_weight = torch.full((num_nodes, ),
                                 1 if not improved else 2,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)
        print(x.shape, self.weight.shape)

        if not self.cached or self.cached_result is None:
            edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight,
                                            self.improved, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result
        # print(x.shape, norm.shape)
        # exit(0)
        print(x.shape, edge_index.shape)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # print(norm.shape, x_j.shape)
        # exit(0)
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
