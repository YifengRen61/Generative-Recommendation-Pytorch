from pathlib import Path

import math
import abc
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from typing import Callable
from dataset import save_emb

class RelativeAttentionBiasModule(torch.nn.Module):
    @abc.abstractmethod
    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: [B, N] x int64
        Returns:
            torch.float tensor broadcastable to [B, N, N]
        """
        pass

class RelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """

    def __init__(
        self,
        maxlen: int,
        num_buckets: int,
        bucketization_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        self._maxlen: int = maxlen
        self._ts_w = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * maxlen - 1).normal_(mean=0, std=0.02),
        )
        
        self._num_buckets: int = num_buckets
        self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = (
            bucketization_fn
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = self._maxlen
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1
        )
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
            ),
            min=0,
            max=self._num_buckets,
        ).detach()

        rel_pos_bias = t[:, :, r:-r]
        rel_ts_bias = torch.index_select(
            self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)
        return rel_pos_bias + rel_ts_bias

class TimeEmbEncoder(torch.nn.Module):
    def __init__(self, hidden_units, dev, num_buckets, bucketization_fn, dropout_rate=0.2):
        super().__init__()
        self.dev = dev
        self._bucketization_fn = bucketization_fn
        self._num_buckets = num_buckets
        self.hour_emb = torch.nn.Embedding(24 + 1, 8, padding_idx=0)
        self.weekday_emb = torch.nn.Embedding(7 + 1, 8, padding_idx=0)
        self.isweekend_emb = torch.nn.Embedding(2 + 1, 2, padding_idx=0)
        self.t_from_prev_emb = torch.nn.Embedding(num_buckets + 2, 16, padding_idx=0)
        self.norm = torch.nn.RMSNorm(8 + 8 + 2 + 16)
        self.clock_proj = torch.nn.Linear(8 + 8 + 2 + 16, hidden_units, bias=False)
        self.drop = torch.nn.Dropout(dropout_rate)        
    
    def timestamp_to_weekday_hour(self, timestamps):
        """
        timestamps: np.ndarray[int64], Unix 时间戳 (单位秒), 默认 UTC
        return: (weekday, hour)
            weekday: 1=Monday, ..., 7=Sunday
            hour: 1..24
        """
        mask = (timestamps != 0)
        
        hour = ((timestamps // 3600) % 24) + 1 # 小时 (1-24)
        weekday = (((timestamps // 86400) + 4) % 7) + 1 # 星期 (1-7), 1970-01-01 是周四(+4)
        isweekend = (weekday >= 6).to(torch.int32) + 1 # 周末 0 = padding, 1 = 否， 2 = 是
        hour_masked = hour * mask
        weekday_masked = weekday * mask
        isweekend_masked = isweekend * mask
        return weekday_masked, hour_masked, isweekend_masked

    def forward(self, timestamps, t_from_prev, mask=None):
        weekday, hour, isweekend = self.timestamp_to_weekday_hour(timestamps)
        batch_weekday_emb = self.weekday_emb(weekday)
        batch_hour_emb = self.hour_emb(hour)
        batch_isweekend_emb = self.isweekend_emb(isweekend)

        buckets = (torch.clamp(
            self._bucketization_fn(
                t_from_prev
            ),
            min=0,
            max=self._num_buckets,
        ) + 1).detach()
        buckets = torch.where(mask.bool(), buckets, torch.zeros_like(buckets))
        t_from_prev_bucket_emb = self.t_from_prev_emb(buckets)
        
        time_emb = torch.cat([batch_weekday_emb, batch_hour_emb, batch_isweekend_emb, t_from_prev_bucket_emb], dim=-1)
        time_emb = self.norm(time_emb)
        final_emb = self.drop(self.clock_proj(time_emb))
        return final_emb

class HSTUAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate, maxlen):
        super(HSTUAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate
        self.maxlen = maxlen

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"
        self.qkvu_proj = torch.nn.Parameter(
            torch.empty(
                (
                    hidden_units,
                    4 * hidden_units,
                )
            ).normal_(mean=0, std=0.02),
        )
        self.output = torch.nn.Linear(hidden_units, hidden_units)
        
        relative_attention_bias_module=(
                        RelativeBucketedTimeAndPositionBasedBias(
                            maxlen = maxlen + 1,  # accounts for next item.
                            num_buckets=64,
                            bucketization_fn=lambda x: (
                                torch.log(torch.abs(x).clamp(min=1)) / 0.301 # 相对时间需要更细致的分桶
                            ).long(),
                        )
                    )
        self._rel_attn_bias = relative_attention_bias_module

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self.hidden_units], eps=1e-8)
    
    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, normalized_shape=[self.head_dim * self.num_heads], eps=1e-8
        )

    def forward(self, seqs, mask, action_ts):
        batch_size, seq_len, _ = seqs.size()
        n = mask.size(-1)
        # pre-norm
        x = self._norm_input(seqs)
        # U(X), V (X), Q(X), K(X) = Split(ϕ1(f1(X))) 
        QKVU = torch.matmul(x, self.qkvu_proj)
        QKVU = F.silu(QKVU)
        Q, K, V, U = torch.chunk(QKVU, 4, dim=-1)
        
        # 将QKVU 拆成num_heads 个 （batch_size, seq_len, self.head_dim）
        # transpose QKVU，让最后两个维度为（..., seq_len, self.head_dim）
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        # matmul 得到(..., seq_len, seq_len)
        QK_attn = torch.matmul(Q, K.transpose(-2,-1)) # 互换最后两个维度
        rel_attn_bias = self._rel_attn_bias(action_ts).unsqueeze(1)
        QK_attn = QK_attn + rel_attn_bias
        QK_attn = F.silu(QK_attn) / n

        if mask is not None:
            QK_attn = QK_attn * mask.unsqueeze(1)
        AV = torch.matmul(QK_attn, V)
        AV = AV.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
        UnormAV = U * self._norm_attn_output(AV)
        
        output = F.dropout(
                    UnormAV,
                    p=self.dropout_rate,
                    training=self.training,
                )
        new_output = self.output(output)
        return new_output, None

class SwiGLU(torch.nn.Module):
    def __init__(self, in_dim, out_dim, mult=2, p=0.1):
        super().__init__()
        self.fc1  = torch.nn.Linear(in_dim, mult*in_dim*2)      
        self.fc2 = torch.nn.Linear(mult*in_dim, out_dim)
        self.drop   = torch.nn.Dropout(p)

    # SwiGLU(x)=(W_v​x+b_v​) ⊙ SiLU(W_​gx+b_g​)
    def forward(self, x):
        v, g = self.fc1(x).chunk(2, -1)
        y = F.silu(g) * v
        return self.fc2(self.drop(y))
    
# 简化一层DCN
class TinyCross(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.w = torch.nn.Linear(in_dim, 1, bias=True)

    def forward(self, x0, xi):
        return x0 * self.w(xi) + xi

# 物品塔和序列塔
class ItemDCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_mult=1, dropout=0.2, eps=1e-8):
        super().__init__()
        #self.bn = torch.nn.BatchNorm1d(in_dim, eps=eps)
        self.norm = torch.nn.RMSNorm(in_dim, eps=eps)
        self.ffn = SwiGLU(in_dim, out_dim, hidden_mult, dropout)
        self.res_proj = torch.nn.Linear(in_dim, out_dim, bias=False)
        #self.cross = TinyCross(in_dim) # 1 层 cross，够用了
        
    def forward(self, x):
        x0 = x
        x = self.norm(x)
        #x = self.cross(x0, x)
        y = self.ffn(x) 
        output = self.res_proj(x0) + y
        return output

# 用户塔
class UserDNN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_mult=1, dropout=0.2, eps=1e-8):
        super().__init__()
        self.norm = torch.nn.RMSNorm(in_dim, eps=eps)
        self.ffn = SwiGLU(in_dim, out_dim, hidden_mult, dropout)
        self.res_proj = torch.nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        x0 = x
        x = self.norm(x)
        y = self.ffn(x)
        output = self.res_proj(x0) + y
        return output
    
class HSTUModel(torch.nn.Module):
    """
    Args:
        user_num: 用户数量
        item_num: 物品数量
        feat_statistics: 特征统计信息，key为特征ID，value为特征数量
        feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        args: 全局参数
    Attributes:
        user_num: 用户数量
        item_num: 物品数量
        dev: 设备
        norm_first: 是否先归一化
        maxlen: 序列最大长度
        item_emb: Item Embedding Table
        user_emb: User Embedding Table
        sparse_emb: 稀疏特征Embedding Table
        emb_transform: 多模态特征的线性变换
        userdnn: 用户特征拼接后经过的全连接层
        itemdnn: 物品特征拼接后经过的全连接层
    """
    def __init__(self, user_num, item_num, user_feat_statistics, user_feat_types, item_feat_statistics, item_feat_types, args):
        super(HSTUModel, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.maxlen = args.maxlen
        self.complex_feat_emb = args.complex_feat_emb
        self.hidden_units = args.hidden_units
        # creating embedding template
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate) 
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()
        self.hstu_layers = torch.nn.ModuleList()
        self._init_feat_info(user_feat_statistics, user_feat_types, item_feat_statistics, item_feat_types)

        # 用户信息融合进序列
        self.fuse_pre_norm = torch.nn.LayerNorm(args.hidden_units)
        self.crossAttention = torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, batch_first=True)
        self.dropout = torch.nn.Dropout(args.dropout_rate)
        # 把门初始化成小开度：sigmoid(gate) ≈ init_gate
        gate = 0.05
        self.gate = torch.nn.Parameter(torch.tensor(math.log(gate / (1 - gate))))
        
        # 绝对时间(周期时间emb) 和 相邻时间emb
        absolute_time_emb_module = (
                        TimeEmbEncoder(
                            args.hidden_units, 
                            self.dev,
                            num_buckets=32,
                            bucketization_fn=lambda x: (
                                torch.log(torch.abs(x).clamp(min=1)) / 0.693 # recency 时间分桶不需要如相对时间一样的细致分桶
                            ).long()
                        )
                    )
        self.absolute_time_emb = absolute_time_emb_module

        # hstu层
        for _ in range(args.num_blocks):
            new_hstu_layer = HSTUAttention(args.hidden_units, args.num_heads, args.dropout_rate, args.maxlen)
            self.hstu_layers.append(new_hstu_layer)

        # 用户所有特征拼接后的向量
        userdim = len(self.USER_CONTINUAL_FEAT) + args.hidden_units
        itemdim = len(self.ITEM_CONTINUAL_FEAT) + args.hidden_units

        # 为每一个特征创建一个Embedding空间
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, self.USER_SPARSE_FEAT_DIM[k], padding_idx=0)
            userdim += self.USER_SPARSE_FEAT_DIM[k]
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, self.ITEM_SPARSE_FEAT_DIM[k], padding_idx=0)
            itemdim += self.ITEM_SPARSE_FEAT_DIM[k]
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, self.ITEM_ARRAY_FEAT_DIM[k], padding_idx=0)
            itemdim += self.ITEM_ARRAY_FEAT_DIM[k]
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, self.USER_ARRAY_FEAT_DIM[k], padding_idx=0)
            userdim += self.USER_ARRAY_FEAT_DIM[k]
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)
            itemdim += args.hidden_units
        
        # 使用更复杂的MLP
        if args.complex_feat_emb:
            self.userdnn = UserDNN(userdim, args.hidden_units)
            self.itemdnn = ItemDCN(itemdim, args.hidden_units)
            #self.seqdnn = smallDNN(itemdim, args.hidden_units)
        else:
            self.userdnn = torch.nn.Linear(userdim, args.hidden_units)
            self.itemdnn = torch.nn.Linear(itemdim, args.hidden_units)

        self.register_buffer(
            "_attn_mask",
            torch.tril(
                torch.ones(
                    (
                        self.maxlen + 1,
                        self.maxlen + 1,
                    ),
                    dtype=torch.bool,
                ),
                diagonal=0,
            ),
        )

    def _init_feat_info(self, user_feat_statistics, user_feat_types, item_feat_statistics, item_feat_types):
            """
            将特征统计信息（特征数量）按特征类型分组产生不同的字典，方便声明稀疏特征的Embedding Table
            Args:
                feat_statistics: 特征统计信息，key为特征ID，value为特征数量
                feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
            """
            self.USER_SPARSE_FEAT = {k: user_feat_statistics[k] for k in user_feat_types['user_sparse']}
            self.USER_SPARSE_FEAT_DIM = {k: min(math.ceil(6 * user_feat_statistics[k]**0.25), self.hidden_units) for k in user_feat_types['user_sparse']}
            self.USER_CONTINUAL_FEAT = user_feat_types['user_continual']
            self.USER_ARRAY_FEAT = {k: user_feat_statistics[k] for k in user_feat_types['user_array']}
            self.USER_ARRAY_FEAT_DIM = {k: min(math.ceil(6 * user_feat_statistics[k]**0.25), self.hidden_units) for k in user_feat_types['user_array']}

            self.ITEM_SPARSE_FEAT = {k: item_feat_statistics[k] for k in item_feat_types['item_sparse']}
            self.ITEM_SPARSE_FEAT_DIM = {k: min(math.ceil(6 * item_feat_statistics[k]**0.25), self.hidden_units) for k in item_feat_types['item_sparse']}
            self.ITEM_CONTINUAL_FEAT = item_feat_types['item_continual']
            self.ITEM_ARRAY_FEAT = {k: item_feat_statistics[k] for k in item_feat_types['item_array']}
            self.ITEM_ARRAY_FEAT_DIM = {k: min(math.ceil(6 * item_feat_statistics[k]**0.25), self.hidden_units) for k in item_feat_types['item_array']}
            EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
            self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in item_feat_types['item_emb']}  # 记录的是不同多模态特征的维度

    def itemfeat2emb(self, seq, feat_tensor, mask=None):
        """
        Args:
            seq: 序列ID
            feature_array: 特征list，每个元素为当前时刻的特征字典
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征，在两种情况下不打开：1) 训练时在转换正负样本的特征时（因为正负样本都是item）;2) 生成候选库item embedding时。

        Returns:
            seqs_emb: 序列特征的Embedding
        """
        seq = seq.to(self.dev)
        # pre-compute embedding
        item_embedding = self.item_emb(seq)
        item_feat_list = [item_embedding]

        # batch-process all feature types
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
        ]

        # batch-process each feature type
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                tensor_feature = feat_tensor[k].to(self.dev)

                if feat_type.endswith('sparse'):
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type.endswith('array'):
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type.endswith('continual'):
                    feat_list.append(tensor_feature.unsqueeze(2))

        for k in self.ITEM_EMB_FEAT:
            # collect all data to numpy, then batch-convert
            batch_size = len(feat_tensor['100'])
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feat_tensor['100'][0])

            if k in feat_tensor:
                batch_emb_data = feat_tensor[k]
            else:
                batch_emb_data = torch.zeros((batch_size, seq_len, emb_dim), dtype=torch.float32)
            # batch-convert and transfer to GPU
            item_feat_list.append(self.emb_transform[k](batch_emb_data.to(self.dev)))

        # merge features
        all_item_emb = torch.cat(item_feat_list, dim=2)
        if self.complex_feat_emb:
            all_item_emb = self.itemdnn(all_item_emb)
        else:
            all_item_emb = torch.relu(self.itemdnn(all_item_emb))
            
        seqs_emb = all_item_emb
        return seqs_emb

    def userfeat2emb(self, user_id, feat_tensor, mask=None):
        """
        Args:
            seq: 序列ID
            feature_array: 特征list，每个元素为当前时刻的特征字典
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征，在两种情况下不打开：1) 训练时在转换正负样本的特征时（因为正负样本都是item）;2) 生成候选库item embedding时。

        Returns:
            seqs_emb: 序列特征的Embedding
        """
        user_id = user_id.to(self.dev)
        # pre-compute embedding
        user_embedding = self.user_emb(user_id)
        user_feat_list = [user_embedding]

        # batch-process all feature types
        all_feat_types = [
            (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
            (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
            (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
        ]

        # batch-process each feature type
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                tensor_feature = feat_tensor[k].to(self.dev)

                if feat_type.endswith('sparse'):
                    a = self.sparse_emb[k](tensor_feature)
                    feat_list.append(a)
                elif feat_type.endswith('array'):
                    a = self.sparse_emb[k](tensor_feature).sum(1)
                    feat_list.append(a)
                elif feat_type.endswith('continual'):
                    feat_list.append(tensor_feature.unsqueeze(1))

        # merge features
        all_user_emb = torch.cat(user_feat_list, dim=1)
        if self.complex_feat_emb:
            all_user_emb = self.userdnn(all_user_emb)
        else:
            all_user_emb = torch.relu(self.userdnn(all_user_emb))
        seqs_emb = all_user_emb
        return seqs_emb

    def log2feats(self, log_seqs, user_emb, mask, seq_feature, action_ts, t_from_prev):
        """
        Args:
            log_seqs: 序列ID
            mask: token类型掩码，1表示item token，2表示user token
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典

        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """
        seqs = self.itemfeat2emb(log_seqs, seq_feature, mask=mask)
        # 初始化是xavier，范数已经是1了
        seqs *= self.item_emb.embedding_dim**0.5
        action_ts = action_ts.to(self.dev)
        t_from_prev = t_from_prev.to(self.dev)
        time_emb = self.absolute_time_emb(action_ts, t_from_prev, mask)
        seqs = seqs + time_emb

        seqs = self.emb_dropout(seqs)
        #attention_mask_pad = (mask != 0).to(self.dev)
        
        # 用户feat emb 融合进序列
        user_kv = user_emb.unsqueeze(1)
        seqs_norm = self.fuse_pre_norm(seqs)
        cross, _ = self.crossAttention(
            query=seqs_norm, 
            key=user_kv,
            value=user_kv,
            need_weights=False
        )
        cross = cross * mask.unsqueeze(-1)
        seqs = seqs + torch.sigmoid(self.gate) * self.dropout(cross)

        # 只看过去 & 剔除无效交互
        attention_mask = self._attn_mask.unsqueeze(0) & mask.unsqueeze(1)

        for i in range(len(self.hstu_layers)):
            # 里面用的一定是pre-norm
            mha_outputs, _ = self.hstu_layers[i](seqs, attention_mask, action_ts)
            seqs = seqs + mha_outputs
        return seqs

    def forward(
        self, user_item, pos_seqs, neg_seqs, user_id, mask, next_mask, action_ts, t_from_prev, next_action_type, seq_dict_merged, pos_feat_merged, neg_feat_merged, user_feat_merged
    ):
        """
        训练时调用，计算正负样本的logits

        Args:
            user_item: 用户序列ID
            pos_seqs: 正样本序列ID
            neg_seqs: 负样本序列ID
            mask: token类型掩码，1表示item token，2表示user token
            next_mask: 下一个token类型掩码，1表示item token，2表示user token
            next_action_type: 下一个token动作类型，0表示曝光，1表示点击
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            pos_feature: 正样本特征list，每个元素为当前时刻的特征字典
            neg_feature: 负样本特征list，每个元素为当前时刻的特征字典

        Returns:
            pos_logits: 正样本logits，形状为 [batch_size, maxlen]
            neg_logits: 负样本logits，形状为 [batch_size, maxlen]
        """
        # 用户塔
        user_emb = self.userfeat2emb(user_id, user_feat_merged)

        # 序列塔
        attention_mask_pad = (mask != 0).to(self.dev)
        log_feats = self.log2feats(user_item, user_emb, attention_mask_pad, seq_dict_merged, action_ts, t_from_prev)

        # 物品塔
        pos_embs = self.itemfeat2emb(pos_seqs, pos_feat_merged)
        neg_embs = self.itemfeat2emb(neg_seqs, neg_feat_merged)

        return pos_embs, neg_embs, log_feats
    
    def predict(self, log_seqs, user_id, seq_feature, mask, action_ts, t_from_prev, user_feat_merged):
        """
        计算用户序列的表征
        Args:
            log_seqs: 用户序列ID
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            mask: token类型掩码，1表示item token，2表示user token
        Returns:
            final_feat: 用户序列的表征，形状为 [batch_size, hidden_units]
        """
        user_emb = self.userfeat2emb(user_id, user_feat_merged)
        log_feats = self.log2feats(log_seqs, user_emb, mask, seq_feature, action_ts, t_from_prev)
        final_feat = log_feats[:, -1, :]
        final_feat = F.normalize(final_feat, dim=-1, eps=1e-8)
        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """
        生成候选库item embedding，用于检索
        Args:
            item_ids: 候选item ID（re-id形式）
            retrieval_ids: 候选item ID（检索ID，从0开始编号，检索脚本使用）
            feat_dict: 训练集所有item特征字典，key为特征ID，value为特征值
            save_path: 保存路径
            batch_size: 批次大小
        """
        all_embs = []
        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))
            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])
            batch_feat = np.array(batch_feat, dtype=object)
            emb_feats = [str(s) for s in range(81, 87)]
            def padding(seq_feat):
                batch_size = len(seq_feat)
                max_seq_len = len(seq_feat[0])
                feats = seq_feat[0][0].keys()
                final_dict = dict()
                for k in feats:
                    k_type = seq_feat[0][0][k]
                    if k in emb_feats:
                        emb_size = len(seq_feat[0][0][k])
                        batch_data = np.zeros((batch_size, max_seq_len, emb_size), dtype=np.float32)
                        for i in range(batch_size):
                            seq_data = [item[k] for item in seq_feat[i]]
                            batch_data[i] = seq_data
                        final_dict[k] = torch.from_numpy(batch_data)
                    elif not np.isscalar(k_type):
                        # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
                        max_array_len = 0
                        # batch 里的每个用户
                        for i in range(batch_size):
                            # 每个用户每有一个item，就把他放进seq_data
                            seq_data = [item[k] for item in seq_feat[i]]
                            # 最长的item
                            max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))
                        # 初始化位置
                        batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
      
                        for i in range(batch_size):
                            seq_data = [item[k] for item in seq_feat[i]]
                            for j, item_data in enumerate(seq_data):
                                actual_len = min(len(item_data), max_array_len)
                                batch_data[i, j, :actual_len] = item_data[:actual_len]
                        final_dict[k] = torch.from_numpy(batch_data)
                    else:
                        # 如果特征是Sparse类型，直接转换为tensor
                        batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)
                        for i in range(batch_size):
                            seq_data = [item[k] for item in seq_feat[i]]
                            batch_data[i] = seq_data
                        final_dict[k] = torch.from_numpy(batch_data)
                return final_dict
        
            batch_feat = padding([batch_feat])
            batch_emb = self.itemfeat2emb(item_seq, batch_feat).squeeze(0)
            batch_emb = F.normalize(batch_emb, dim=-1, eps=1e-8)
            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))
        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))