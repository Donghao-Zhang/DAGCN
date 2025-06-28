from torch import nn
import torch
import math
import numpy as np


class ConcatAttention(nn.Module):
    def __init__(self, quary_size, key_size, value_size, dropout=0.0):
        super(ConcatAttention, self).__init__()
        self.att = nn.Sequential(
            nn.Linear(quary_size + key_size, quary_size + key_size, bias=True),
            nn.LeakyReLU(),
            nn.Linear(quary_size + key_size, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask_out=None, dist=None):
        """
        :param Q: [..., seq_len_q, key_size]
        :param K: [..., seq_len_k, key_size]
        :param V: [..., seq_len_k, value_size]
        :param mask_out: [..., 1, seq_len] or [..., seq_len_q, seq_len_k]
        """
        q_input = Q.unsqueeze(2).expand(-1, -1, K.size(1), -1)
        k_input = K.unsqueeze(1).expand(-1, Q.size(1), -1, -1)
        output = self.att(torch.cat([q_input, k_input], dim=3)).squeeze(3)
        mask = None
        if dist is not None:
            if isinstance(dist, tuple):
                Q += dist[0]
                K += dist[1]
            else:
                adj_self_loop = dist + torch.eye(dist.size(1)).unsqueeze(0).expand(dist.size(0), -1, -1).cuda()
                mask = adj_self_loop.eq(0)
        else:
            mask = mask_out
        if mask is not None:
            output.masked_fill_(mask, -1e9)
        output = self.softmax(output)
        output = self.drop(output)
        output = torch.matmul(output, V)
        return output


class GeneralDotAttention(nn.Module):
    def __init__(self, quary_size, key_size, value_size, dropout=0.0):
        super(GeneralDotAttention, self).__init__()
        self.dot_att = DotAttention(key_size, value_size, dropout)
        self.weight = nn.Linear(quary_size, key_size, bias=False)

    def forward(self, Q, K, V, mask_out=None, dist=None):
        """
        :param Q: [..., seq_len_q, key_size]
        :param K: [..., seq_len_k, key_size]
        :param V: [..., seq_len_k, value_size]
        :param mask_out: [..., 1, seq_len] or [..., seq_len_q, seq_len_k]
        """
        Q_weight = self.weight(Q)
        mask = mask_out
        if dist is not None:
            if isinstance(dist, tuple):
                Q += dist[0]
                K += dist[1]
            else:
                adj_self_loop = dist + torch.eye(dist.size(1)).unsqueeze(0).expand(dist.size(0), -1, -1).cuda()
                # adj_self_loop += adj_self_loop.permute(0, 2, 1)
                mask = adj_self_loop.eq(0)
        output = self.dot_att(Q_weight, K, V, mask)
        return output


class DotAttention(nn.Module):
    """
    Transformer当中的DotAttention
    """

    def __init__(self, key_size, value_size, dropout=0.0):
        super(DotAttention, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.scale = math.sqrt(key_size)
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask_out=None):
        """
        :param Q: [..., seq_len_q, key_size]
        :param K: [..., seq_len_k, key_size]
        :param V: [..., seq_len_k, value_size]
        :param mask_out: [..., 1, seq_len] or [..., seq_len_q, seq_len_k]
        """
        output = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        if mask_out is not None:
            output.masked_fill_(mask_out, -1e12)
        output = self.softmax(output)
        output = self.drop(output)
        return torch.matmul(output, V)


class ConstrainDotAttention(nn.Module):
    def __init__(self, key_size, value_size, dropout=0.0):
        super(ConstrainDotAttention, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.scale = math.sqrt(key_size)
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    # 修改
    def forward(self, Q, K, V, mask_out=None):
        """
        :param Q: [..., seq_len_q, key_size]
        :param K: [..., seq_len_k, key_size]
        :param V: [..., seq_len_k, value_size]
        :param mask_out: [..., 1, seq_len] or [..., seq_len_q, seq_len_k]
        """
        output = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        output = self.softmax(output)
        output = mask_out * output
        # 修改，注释+增加
        output = self.drop(output)
        return torch.matmul(output, V)


class MultiLayerConstrainMultiHeadAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_head, num_layer, dropout=0.1):
        super(MultiLayerConstrainMultiHeadAttention, self).__init__()
        self.num_layer = num_layer
        self.aggregator = nn.ModuleList([
            ConstrainMultiHeadAttention(query_size, key_size, value_size, num_head, dropout) for _ in range(num_layer)
        ])

    def forward(self, Q, K, V, atte_mask_out=None, dist=None):
        input_value = V
        for i in range(self.num_layer):
            K = V = self.aggregator[i](Q, K, V, atte_mask_out, dist)
        # return V
        # return self.dropout(V) + input_value
        return V + input_value


class ConstrainMultiHeadAttention(nn.Module):
    """
    Transformer当中的MultiHeadAttention
    """
    def __init__(self, query_size, key_size, value_size, num_head, dropout=0.1):
        """
        :param query_size: int, 输入维度的大小。同时也是输出维度的大小。
        :param key_size: int, 每个head的维度大小。
        :param value_size: int，每个head中value的维度。
        :param num_head: int，head的数量。
        :param dropout: float。
        """
        super(ConstrainMultiHeadAttention, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        assert query_size % num_head == 0, "d_model is %d, n_head is %d" % (query_size, num_head)
        assert value_size % num_head == 0, "value is %d, n_head is %d" % (value_size, num_head)
        self.value_size = value_size
        self.num_head = num_head

        self.head_size_q = int(query_size / num_head)
        self.head_size_k = int(key_size / num_head)
        self.head_size_v = int(value_size / num_head)

        # in_size = head_size_q * num_head
        self.q_in = nn.Linear(query_size, key_size)
        self.k_in = nn.Linear(key_size, key_size)
        self.v_in = nn.Linear(value_size, value_size)
        self.attention = ConstrainDotAttention(key_size=self.head_size_k, value_size=self.head_size_v, dropout=dropout)
        self.out = nn.Linear(value_size, value_size)
        self.reset_parameters()

    def reset_parameters(self):
        sqrt = math.sqrt
        nn.init.normal_(self.q_in.weight, mean=0, std=sqrt(1.0 / self.query_size))
        nn.init.normal_(self.k_in.weight, mean=0, std=sqrt(1.0 / self.key_size))
        nn.init.normal_(self.v_in.weight, mean=0, std=sqrt(1.0 / self.value_size))
        nn.init.normal_(self.out.weight, mean=0, std=sqrt(1.0 / self.value_size))

    def forward(self, Q, K, V, atte_mask_out=None, dist=None):
        """
        :param Q: [batch, seq_len_q, model_size]
        :param K: [batch, seq_len_k, model_size]
        :param V: [batch, seq_len_k, model_size]
        :param seq_mask: [batch, seq_len]
        """
        batch, sq, _ = Q.size()
        mask = None
        if dist is not None:
            if isinstance(dist, tuple):
                Q += dist[0]
                K += dist[1]
            else:
                adj_self_loop = dist + torch.eye(dist.size(1)).unsqueeze(0).expand(dist.size(0), -1, -1).cuda()
                mask = adj_self_loop.unsqueeze(1)

        sk = K.size(1)
        d_k, d_v, n_head = self.head_size_k, self.head_size_v, self.num_head
        # input linear
        q = self.q_in(Q).view(batch, sq, n_head, d_k).transpose(1, 2)
        k = self.k_in(K).view(batch, sk, n_head, d_k).transpose(1, 2)
        v = self.v_in(V).view(batch, sk, n_head, d_v).transpose(1, 2)

        if atte_mask_out is not None and mask is None:
            mask = atte_mask_out[:, None, :, :]  # [bsz,1,1,len]
        # 修改
        atte = self.attention(q, k, v, mask).view(batch, n_head, sq, d_v)
        # atte = self.attention(q, k, v, atte_mask_out[:, None, :, :], mask).view(batch, n_head, sq, d_v)

        # concat all heads, do output linear
        atte = atte.transpose(1, 2).contiguous().view(batch, sq, -1)
        output = self.out(atte)
        return output


class LocalDotAttention(nn.Module):
    """
    Transformer当中的DotAttention
    """

    def __init__(self, key_size, value_size, dropout=0.0):
        super(LocalDotAttention, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.scale = math.sqrt(key_size)
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask_out=None):
        """
        :param Q: [..., seq_len_q, key_size]
        :param K: [..., seq_len_k, key_size]
        :param V: [..., seq_len_k, value_size]
        :param mask_out: [..., 1, seq_len] or [..., seq_len_q, seq_len_k]
        """
        relative_position = torch.from_numpy(
            relative_position_mat_create(Q.size(2))
        ).unsqueeze(0).unsqueeze(1).expand(Q.size(0), Q.size(1), -1, -1).cuda()
        output = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        if mask_out is not None:
            output.masked_fill_(mask_out, -1e9)
        output = torch.mul(self.softmax(output), torch.exp(
            -torch.pow(relative_position, 2) / (output.size(-1)**2/2)
        ))
        output = self.drop(output)
        return torch.matmul(output, V)


def relative_position_mat_create(max_length):
    relative_position_mat = np.zeros((max_length, max_length), dtype=np.float32)
    for i in range(1, max_length):
        index = np.arange(max_length-i)
        relative_position_mat[index, index+i] = i
        relative_position_mat[index+i, index] = i
    return relative_position_mat


class MultiHeadAttention(nn.Module):
    """
    Transformer当中的MultiHeadAttention
    """
    def __init__(self, query_size, key_size, value_size, num_head, dropout=0.1):
        """
        :param query_size: int, 输入维度的大小。同时也是输出维度的大小。
        :param key_size: int, 每个head的维度大小。
        :param value_size: int，每个head中value的维度。
        :param num_head: int，head的数量。
        :param dropout: float。
        """
        super(MultiHeadAttention, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        assert query_size % num_head == 0, "d_model is %d, n_head is %d" % (query_size, num_head)
        assert value_size % num_head == 0, "value is %d, n_head is %d" % (value_size, num_head)
        self.value_size = value_size
        self.num_head = num_head

        self.head_size_q = int(query_size / num_head)
        self.head_size_k = int(key_size / num_head)
        self.head_size_v = int(value_size / num_head)

        # in_size = head_size_q * num_head
        self.q_in = nn.Linear(query_size, key_size)
        self.k_in = nn.Linear(key_size, key_size)
        self.v_in = nn.Linear(value_size, value_size)
        self.attention = DotAttention(key_size=self.head_size_k, value_size=self.head_size_v, dropout=dropout)
        self.out = nn.Linear(value_size, value_size)
        self.reset_parameters()

    def reset_parameters(self):
        sqrt = math.sqrt
        nn.init.normal_(self.q_in.weight, mean=0, std=sqrt(1.0 / self.query_size))
        nn.init.normal_(self.k_in.weight, mean=0, std=sqrt(1.0 / self.key_size))
        nn.init.normal_(self.v_in.weight, mean=0, std=sqrt(1.0 / self.value_size))
        nn.init.normal_(self.out.weight, mean=0, std=sqrt(1.0 / self.value_size))


    def forward(self, Q, K, V, atte_mask_out=None):
        """
        :param Q: [batch, seq_len_q, model_size]
        :param K: [batch, seq_len_k, model_size]
        :param V: [batch, seq_len_k, model_size]
        :param seq_mask: [batch, seq_len]
        """
        batch, sq, _ = Q.size()
        sk = K.size(1)
        d_k, d_v, n_head = self.head_size_k, self.head_size_v, self.num_head
        # input linear
        q = self.q_in(Q).view(batch, sq, n_head, d_k).transpose(1, 2)
        k = self.k_in(K).view(batch, sk, n_head, d_k).transpose(1, 2)
        v = self.v_in(V).view(batch, sk, n_head, d_v).transpose(1, 2)

        if atte_mask_out is not None:
            atte_mask_out = atte_mask_out[:, None, :, :]  # [bsz,1,1,len]
        atte = self.attention(q, k, v, atte_mask_out).view(batch, n_head, sq, d_v)

        # concat all heads, do output linear
        atte = atte.transpose(1, 2).contiguous().view(batch, sq, -1)
        output = self.out(atte)
        return output


class MultiLayerRelativeMultiHead(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_head, num_layer, dropout=0.1):
        super(MultiLayerRelativeMultiHead, self).__init__()
        self.num_layer = num_layer
        self.aggregator = nn.ModuleList([
            RelativeMultiHeadAttention(query_size, key_size, value_size, num_head, dropout) for _ in range(num_layer)
        ])

    def forward(self, Q, K, V, atte_mask_out=None, dist=None):
        input_value = V
        for i in range(self.num_layer):
            K = V = self.aggregator[i](Q, K, V, atte_mask_out, dist)
        return V + input_value


class RelativeMultiHeadAttention(nn.Module):
    """
    Transformer当中的MultiHeadAttention
    """
    def __init__(self, query_size, key_size, value_size, num_head, dropout=0.1):
        """
        :param query_size: int, 输入维度的大小。同时也是输出维度的大小。
        :param key_size: int, 每个head的维度大小。
        :param value_size: int，每个head中value的维度。
        :param num_head: int，head的数量。
        :param dropout: float。
        """
        super(RelativeMultiHeadAttention, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        assert query_size % num_head == 0, "d_model is %d, n_head is %d" % (query_size, num_head)
        assert value_size % num_head == 0, "value is %d, n_head is %d" % (value_size, num_head)
        self.value_size = value_size
        self.num_head = num_head

        self.head_size_q = int(query_size / num_head)
        self.head_size_k = int(key_size / num_head)
        self.head_size_v = int(value_size / num_head)

        # in_size = head_size_q * num_head
        self.q_in = nn.Linear(query_size, key_size)
        self.k_in = nn.Linear(key_size, key_size)
        self.v_in = nn.Linear(value_size, value_size)
        self.attention = DotAttention(key_size=self.head_size_k, value_size=self.head_size_v, dropout=dropout)
        self.out = nn.Linear(value_size, value_size)
        self.reset_parameters()

    def reset_parameters(self):
        sqrt = math.sqrt
        nn.init.normal_(self.q_in.weight, mean=0, std=sqrt(1.0 / self.query_size))
        nn.init.normal_(self.k_in.weight, mean=0, std=sqrt(1.0 / self.key_size))
        nn.init.normal_(self.v_in.weight, mean=0, std=sqrt(1.0 / self.value_size))
        nn.init.normal_(self.out.weight, mean=0, std=sqrt(1.0 / self.value_size))


    def forward(self, Q, K, V, atte_mask_out=None, dist=None):
        """
        :param Q: [batch, seq_len_q, model_size]
        :param K: [batch, seq_len_k, model_size]
        :param V: [batch, seq_len_k, model_size]
        :param seq_mask: [batch, seq_len]
        """
        batch, sq, _ = Q.size()
        mask = None
        if dist is not None:
            if isinstance(dist, tuple):
                Q += dist[0]
                K += dist[1]
            else:
                adj_self_loop = dist + torch.eye(dist.size(1)).unsqueeze(0).expand(dist.size(0), -1, -1).cuda()
                # adj_self_loop += adj_self_loop.permute(0, 2, 1)
                mask = adj_self_loop.unsqueeze(1).eq(0)

        sk = K.size(1)
        d_k, d_v, n_head = self.head_size_k, self.head_size_v, self.num_head
        # input linear
        q = self.q_in(Q).view(batch, sq, n_head, d_k).transpose(1, 2)
        k = self.k_in(K).view(batch, sk, n_head, d_k).transpose(1, 2)
        v = self.v_in(V).view(batch, sk, n_head, d_v).transpose(1, 2)

        if atte_mask_out is not None and mask is None:
            mask = atte_mask_out[:, None, :, :]  # [bsz,1,1,len]
        atte = self.attention(q, k, v, mask).view(batch, n_head, sq, d_v)

        # concat all heads, do output linear
        atte = atte.transpose(1, 2).contiguous().view(batch, sq, -1)
        output = self.out(atte)
        return output


class LocalMultiHeadAttention(nn.Module):
    """
    Transformer当中的MultiHeadAttention
    """
    def __init__(self, query_size, key_size, value_size, num_head, dropout=0.1):
        """
        :param query_size: int, 输入维度的大小。同时也是输出维度的大小。
        :param key_size: int, 每个head的维度大小。
        :param value_size: int，每个head中value的维度。
        :param num_head: int，head的数量。
        :param dropout: float。
        """
        super(LocalMultiHeadAttention, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        assert query_size % num_head == 0, "d_model is %d, n_head is %d" % (query_size, num_head)
        assert value_size % num_head == 0, "value is %d, n_head is %d" % (value_size, num_head)
        self.value_size = value_size
        self.num_head = num_head

        self.head_size_q = int(query_size / num_head)
        self.head_size_k = int(key_size / num_head)
        self.head_size_v = int(value_size / num_head)

        # in_size = head_size_q * num_head
        self.q_in = nn.Linear(query_size, key_size)
        self.k_in = nn.Linear(key_size, key_size)
        self.v_in = nn.Linear(value_size, value_size)
        self.attention = LocalDotAttention(key_size=self.head_size_k, value_size=self.head_size_v, dropout=dropout)
        self.out = nn.Linear(value_size, value_size)
        self.reset_parameters()

    def reset_parameters(self):
        sqrt = math.sqrt
        nn.init.normal_(self.q_in.weight, mean=0, std=sqrt(1.0 / self.query_size))
        nn.init.normal_(self.k_in.weight, mean=0, std=sqrt(1.0 / self.query_size))
        nn.init.normal_(self.v_in.weight, mean=0, std=sqrt(1.0 / self.value_size))
        nn.init.normal_(self.out.weight, mean=0, std=sqrt(1.0 / self.value_size))

    def forward(self, Q, K, V, atte_mask_out=None):
        """
        :param Q: [batch, seq_len_q, model_size]
        :param K: [batch, seq_len_k, model_size]
        :param V: [batch, seq_len_k, model_size]
        :param seq_mask: [batch, seq_len]
        """
        batch, sq, _ = Q.size()
        sk = K.size(1)
        d_k, d_v, n_head = self.head_size_k, self.head_size_v, self.num_head
        # input linear
        q = self.q_in(Q).view(batch, sq, n_head, d_k).transpose(1, 2)
        k = self.k_in(K).view(batch, sk, n_head, d_k).transpose(1, 2)
        v = self.v_in(V).view(batch, sk, n_head, d_v).transpose(1, 2)

        if atte_mask_out is not None:
            atte_mask_out = atte_mask_out[:, None, :, :]  # [bsz,1,1,len]
        atte = self.attention(q, k, v, atte_mask_out).view(batch, n_head, sq, d_v)

        # concat all heads, do output linear
        atte = atte.transpose(1, 2).contiguous().view(batch, sq, -1)
        output = self.out(atte)
        return output