import json
import pickle
import struct
import os

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import mmap
import pickle

# 可选：优先使用 orjson，更快；没有则回退到内置 json
try:
    import orjson as _fastjson
    _FASTJSON = True
except Exception:
    import json as _fastjson
    _FASTJSON = False

class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)

        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        self.item_feature_default_value, self.item_feature_types, self.item_feat_statistics = self._init_item_feat_info()
        self.user_feature_default_value, self.user_feature_types, self.user_feat_statistics = self._init_user_feat_info()

    def _load_data_and_offsets(self):
        """
        只加载路径与 offsets，不在这里打开文件句柄（避免跨进程继承）。
        这里的 seq_offsets.pkl 是一个 list[int]，每个元素是对应行的起始偏移。
        """
        self.data_file_path = self.data_dir / "seq.jsonl"

        with open(self.data_dir / "seq_offsets.pkl", "rb") as f:
            seq_offsets = pickle.load(f)

        # 校验与规范化：要求是 list[int]
        if not isinstance(seq_offsets, list):
            raise TypeError(f"seq_offsets.pkl 期望是 list[int]，但得到 {type(seq_offsets)}")
        if not all(isinstance(x, int) for x in seq_offsets):
            raise TypeError("seq_offsets.pkl 内应全为 int 偏移。")

        self.seq_offsets = seq_offsets  # list[int]，行号即索引
        self.num_rows = len(self.seq_offsets)

        # 预计算 “下一条的起始偏移”，最后一条用文件大小
        file_size = os.path.getsize(self.data_file_path)
        self.seq_next_offsets = self.seq_offsets[1:] + [file_size]

        # 延迟打开（每个进程/worker 自己映射一份）
        self._file = None
        self._mm = None

    # --- 每个进程里按需 mmap 打开，只读，线程安全 ---
    def _ensure_open(self):
        if self._mm is None:
            f = open(self.data_file_path, "rb")
            self._file = f
            # 整个文件映射为只读
            self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    # --- 阻断把已打开的句柄带进子进程（fork/spawn 都安全）---
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_mm"] = None
        return state

    # --- 清理 ---
    def __del__(self):
        try:
            mm = getattr(self, "_mm", None)
            if mm is not None:
                mm.close()
            f = getattr(self, "_file", None)
            if f is not None:
                f.close()
        except Exception:
            pass

    # --- 精确切片读取一行并解析 JSON ---
    def _load_user_data(self, uid: int):
        """
        uid 为行号（与 seq_offsets 的索引一致）。
        通过 mmap[start:end] 精确切出该行 bytes，再用 orjson/json 解析。
        """
        if not (0 <= uid < self.num_rows):
            raise IndexError(f"uid 超界: {uid}（共有 {self.num_rows} 行）")

        self._ensure_open()
        start = self.seq_offsets[uid]
        end = self.seq_next_offsets[uid]
        # 直接切片，无需 readline/seek
        chunk = self._mm[start:end]

        # 兼容 UTF-8 BOM（极少见；若首行带 BOM）
        if chunk.startswith(b"\xef\xbb\xbf"):
            chunk = chunk[3:]

        # 去掉末尾换行（若有）
        if chunk and chunk[-1] in (10, 13):  # \n 或 \r
            # 处理 \r\n 的情况
            if len(chunk) >= 2 and chunk[-2] == 13 and chunk[-1] == 10:
                chunk = chunk[:-2]
            else:
                chunk = chunk[:-1]

        try:
            if _FASTJSON:
                # orjson 直接接收 bytes，速度更快（释放 GIL）
                return _fastjson.loads(chunk)
            else:
                # 回退：内置 json 需要 str
                return _fastjson.loads(chunk.decode("utf-8"))
        except Exception as e:
            head = chunk[:160]
            raise RuntimeError(
                f"Bad JSON at uid={uid}, start={start}, head={head!r}"
            ) from e

    def _random_prob(self):
        p = np.random.randint(0,100)
        return p

    
    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t
    
    def userfeat2tensor(self, u_feat, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [maxlen, max_array_len(if array)]
        """
        if k in self.user_feature_types['user_array']:
            return np.array(u_feat[k])
        else:
            return np.int64(u_feat[k])
    
    def itemfeat2tensor(self, seq_feature, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [maxlen, max_array_len(if array)]
        """

        if k in self.item_feature_types['item_array']:
            # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
            max_seq_len = len(seq_feature)
            # 每个用户每有一个item，就把他放进seq_data
            seq_data = [item[k] for item in seq_feature]
            # 最长的item
            max_array_len = max(len(item_data) for item_data in seq_data)
            # 初始化位置
            tensor_data = np.zeros((max_seq_len, max_array_len), dtype=np.int64)
            # padding 对齐长度
            for j, item_data in enumerate(seq_data):
                actual_len = min(len(item_data), max_array_len)
                tensor_data[j, :actual_len] = item_data[:actual_len]
            return tensor_data
        elif k in self.item_feature_types['item_emb']:
            seq_data = np.array([item[k] for item in seq_feature], dtype=np.float32)
            return seq_data
        else:
            # 如果特征是Sparse类型，直接转换为tensor
            seq_data = np.array([item[k] for item in seq_feature], dtype=np.int64)
            return seq_data

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户ID(reid)

        Returns:
            seq: 用户序列ID
            pos: 正样本ID（即下一个真实访问的item）
            neg: 负样本ID
            token_type: 用户序列类型，1表示item，2表示user
            next_token_type: 下一个token类型，1表示item，2表示user
            action_ts: 用户序列里对应的时间戳
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            pos_feat: 正样本特征，每个元素为字典，key为特征ID，value为特征值
            neg_feat: 负样本特征，每个元素为字典，key为特征ID，value为特征值
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        user_feat = {}
        user_id = 0

        for record_tuple in user_sequence:
            u, i, ufeat, item_feat, action_type, timestamp = record_tuple
            if u and ufeat:
                #ext_user_sequence.insert(0, (u, ufeat, 2, action_type, timestamp))
                user_id = u
                user_feat = ufeat
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type, timestamp))
        
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        action_ts = np.zeros([self.maxlen + 1], dtype=np.int32)
        t_from_prev = np.zeros([self.maxlen + 1], dtype=np.int32)
        t_from_last = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)

        nxt = ext_user_sequence[-1]
        lst_t = ext_user_sequence[-2][4]
        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        # left-padding, 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type, timestamp = record_tuple
            next_i, next_feat, next_type, next_act_type, next_timestamp = nxt
            feat = self.fill_missing_item_feat(feat, i)
            next_feat = self.fill_missing_item_feat(next_feat, next_i)
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            action_ts[idx] = timestamp
            t_from_last[idx] = lst_t - timestamp
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_item_feat(self.item_feat_dict[str(neg_id)], neg_id)
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break 

        t_prev = np.concatenate([action_ts[:1], action_ts[:-1]])
        t_from_prev = action_ts - t_prev
        t_from_prev[idx + 1] = 0 # 注意，因为第一个物品没有前一个物品，所以直接让他成为padding 不被考虑。未来可以单独分一个桶当作初始桶
        
        # 后续可更改为半衰期
        t_half = 24 * 3600  # 24h
        tau = t_half / (np.e - 1)  # ≈ 14h 
        t_from_last = np.where(token_type == 1, 1 / (1 + np.log1p(t_from_last / tau)), 0)
        seq_feat = np.where(seq_feat == None, self.item_feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.item_feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.item_feature_default_value, neg_feat)
        user_feat = {k: (v if user_feat.get(k) == None else user_feat.get(k))
            for k, v in self.user_feature_default_value.items()}
        seq_feat_dict = dict()
        pos_feat_dict = dict()
        neg_feat_dict = dict()
        user_feat_dict = dict()
        for k in self.item_feature_default_value.keys():
            seq_feat_dict[k] = self.itemfeat2tensor(seq_feat, k)
            pos_feat_dict[k] = self.itemfeat2tensor(pos_feat, k)
            neg_feat_dict[k] = self.itemfeat2tensor(neg_feat, k)
            
        for k in self.user_feature_default_value.keys():
            user_feat_dict[k] = self.userfeat2tensor(user_feat, k)

        return seq, pos, neg, user_id, token_type, next_token_type, action_ts, t_from_prev, t_from_last, next_action_type, seq_feat_dict, pos_feat_dict, neg_feat_dict, user_feat_dict
    
    def __len__(self):
        """
        返回数据集长度，即用户数量
 
        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def _init_item_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['item_sparse'] = [
            '100',
            '117',
            '111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]
        feat_types['item_array'] = []
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['item_continual'] = []

        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
                #ist(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float16
            )

        return feat_default_value, feat_types, feat_statistics

    def _init_user_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['user_continual'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_item_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        all_feat_ids = []
        for feat_type in self.item_feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.item_feature_default_value[feat_id]
        for feat_id in self.item_feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]
        return filled_feat
    
    def fill_missing_user_feat(self, feat):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        all_feat_ids = []
        for feat_type in self.user_feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.user_feature_default_value[feat_id]
        return filled_feat

    @staticmethod
    def collate_fn(batch):
        """
        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            pos: 正样本ID, torch.Tensor形式
            neg: 负样本ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            action_ts: 交互时间戳, torch.Tensor形式
            next_token_type: 下一个token类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            pos_feat: 正样本特征, list形式
            neg_feat: 负样本特征, list形式
        """
        seq, pos, neg, user_id, token_type, next_token_type, action_ts, t_from_prev, t_from_last, next_action_type, seq_feat, pos_feat, neg_feat, user_feat= zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        user_id = torch.from_numpy(np.array(user_id))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        action_ts = torch.from_numpy(np.array(action_ts))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        t_from_prev = torch.from_numpy(np.array(t_from_prev))
        t_from_last = torch.from_numpy(np.array(t_from_last))
        seq_dict_merged = dict()
        pos_feat_merged = dict()
        neg_feat_merged = dict()
        user_feat_merged = dict()
        batch_size = len(seq)
        max_seq_len = len(seq[0])
        item_feats = seq_feat[0].keys()
        emb_feats = [str(s) for s in range(81, 87)]
        user_feats = user_feat[0].keys()
        def padding(seq, k):
            k_type = len(seq[0][k][0].shape)
            if k_type == 1 and k not in emb_feats:
                max_array_len = 0
                # batch 里的每个用户
                # 每个用户每有一个item，就把他放进seq_data
                seq_data = [item[k] for item in seq]
                # 最长的item
                max_array_len = max(max_array_len, max(len(item_data[0]) for item_data in seq_data))
                # 初始化位置
                batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data[0]), max_array_len)
                    batch_data[j, :, :actual_len] = item_data
  
                return torch.from_numpy(batch_data)
            else:
                seq_data = [item[k] for item in seq]
                return torch.from_numpy(np.stack(seq_data, axis=0))
                #return torch.from_numpy(np.stack(seq_data, axis=0))

        def user_padding(seq, k):
            k_type = len(seq[0][k].shape)
            if k_type == 1 and k not in emb_feats:
                max_array_len = 0
                # batch 里的每个用户
                # 每个用户每有一个item，就把他放进seq_data
                seq_data = [item[k] for item in seq]
                # 最长的item
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))
                # 初始化位置
                batch_data = np.zeros((batch_size, max_array_len), dtype=np.int64)
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[j, :actual_len] = item_data
  
                return torch.from_numpy(batch_data)
            else:
                seq_data = [item[k] for item in seq]
                return torch.from_numpy(np.stack(seq_data, axis=0))
                #return torch.from_numpy(np.stack(seq_data, axis=0))
        for k in item_feats:
            seq_dict_merged[k] = padding(seq_feat, k)
            pos_feat_merged[k] = padding(pos_feat, k)
            neg_feat_merged[k] = padding(neg_feat, k)

        for k in user_feats:
            user_feat_merged[k] = user_padding(user_feat, k)

        return seq, pos, neg, user_id, token_type, next_token_type, action_ts, t_from_prev, t_from_last, next_action_type, seq_dict_merged, pos_feat_merged, neg_feat_merged, user_feat_merged


class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)
        #self._load_data_and_offsets()

    # 完全对齐基类：设置 data_file_path / seq_offsets / seq_next_offsets，并延迟 mmap
    def _load_data_and_offsets(self):
        self.data_file_path = self.data_dir / "predict_seq.jsonl"

        with open(self.data_dir / "predict_seq_offsets.pkl", "rb") as f:
            seq_offsets = pickle.load(f)

        if not isinstance(seq_offsets, list):
            raise TypeError(f"predict_seq_offsets.pkl 期望是 list[int]，但得到 {type(seq_offsets)}")
        if not all(isinstance(x, int) for x in seq_offsets):
            raise TypeError("predict_seq_offsets.pkl 内应全为 int 偏移。")

        self.seq_offsets = seq_offsets
        self.num_rows = len(self.seq_offsets)

        # 预计算下一条起始偏移（最后一条用文件大小）
        file_size = os.path.getsize(self.data_file_path)
        self.seq_next_offsets = self.seq_offsets[1:] + [file_size]

        # 与基类一致：延迟打开、每个 worker 自己 mmap
        self._file = None
        self._mm = None

    def _process_cold_start_feat(self, feat):
        """
        处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
        """
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if type(feat_value) == list:
                value_list = []
                for v in feat_value:
                    if type(v) == str:
                        value_list.append(0)
                    else:
                        value_list.append(v)
                processed_feat[feat_id] = value_list
            elif type(feat_value) == str:
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def userfeat2tensor(self, u_feat, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [maxlen, max_array_len(if array)]
        """
        if k in self.user_feature_types['user_array']:
            return np.array(u_feat[k])
        else:
            return np.int64(u_feat[k])
    
    def itemfeat2tensor(self, seq_feature, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [maxlen, max_array_len(if array)]
        """

        if k in self.item_feature_types['item_array']:
            # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
            max_seq_len = len(seq_feature)
            # 每个用户每有一个item，就把他放进seq_data
            seq_data = [item[k] for item in seq_feature]
            # 最长的item
            max_array_len = max(len(item_data) for item_data in seq_data)
            # 初始化位置
            tensor_data = np.zeros((max_seq_len, max_array_len), dtype=np.int64)
            # padding 对齐长度
            for j, item_data in enumerate(seq_data):
                actual_len = min(len(item_data), max_array_len)
                tensor_data[j, :actual_len] = item_data[:actual_len]
            return tensor_data
        elif k in self.item_feature_types['item_emb']:
            seq_data = np.array([item[k] for item in seq_feature], dtype=np.float32)
            return seq_data
        else:
            # 如果特征是Sparse类型，直接转换为tensor
            seq_data = np.array([item[k] for item in seq_feature], dtype=np.int64)
            return seq_data



    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        user_feat = {}
        user_re_id = 0
        for record_tuple in user_sequence:
            u, i, ufeat, item_feat, _, timestamp = record_tuple
            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]
                    user_re_id = u
            if u and ufeat:
                if type(u) == str:
                    u = 0
                if ufeat:
                    ufeat = self._process_cold_start_feat(ufeat)
                user_feat = ufeat
                #ext_user_sequence.insert(0, (u, user_feat, 2, timestamp))

            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1, timestamp))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        action_ts = np.zeros([self.maxlen + 1], dtype=np.int32)
        t_from_prev = np.zeros([self.maxlen + 1], dtype=np.int32)
        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])
        for record_tuple in reversed(ext_user_sequence[:]):
            i, feat, type_, timestamp = record_tuple
            feat = self.fill_missing_item_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            action_ts[idx] = timestamp
            idx -= 1
            if idx == -1:
                break
        t_prev = np.concatenate([action_ts[:1], action_ts[:-1]])
        t_from_prev = action_ts - t_prev
        t_from_prev[idx + 1] = 0 # 注意，因为第一个物品没有前一个物品，所以直接让他成为padding 不被考虑。未来可以单独分一个桶当作初始桶
        user_feat = {k: (v if user_feat.get(k) == None else user_feat.get(k))
            for k, v in self.user_feature_default_value.items()}
        user_feat_dict = dict()
        for k in self.user_feature_default_value.keys():
            user_feat_dict[k] = self.userfeat2tensor(user_feat, k)

        seq_feat = np.where(seq_feat == None, self.item_feature_default_value, seq_feat)
        seq_feat_dict = dict()
        for k in self.item_feature_default_value.keys():
            seq_feat_dict[k] = self.itemfeat2tensor(seq_feat, k)
    
        return seq, user_re_id, token_type, seq_feat_dict, user_id, action_ts, t_from_prev, user_feat_dict

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    @staticmethod
    def collate_fn(batch):
        """
        将多个__getitem__返回的数据拼接成一个batch

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            user_id: user_id, str
        """
        seq, user_re_id, token_type, seq_feat, user_id, action_ts, t_from_prev, user_feat = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        user_re_id = torch.from_numpy(np.array(user_re_id))
        action_ts = torch.from_numpy(np.array(action_ts))
        t_from_prev = torch.from_numpy(np.array(t_from_prev))
        seq_dict_merged = dict()
        user_feat_merged = dict()
        batch_size = len(seq)
        max_seq_len = len(seq[0])
        item_feats = seq_feat[0].keys()
        emb_feats = [str(s) for s in range(81, 87)]
        user_feats = user_feat[0].keys()

        def padding(seq, k):
            k_type = len(seq[0][k][0].shape)
            if k_type == 1 and k not in emb_feats:
                max_array_len = 0
                # batch 里的每个用户
                # 每个用户每有一个item，就把他放进seq_data
                seq_data = [item[k] for item in seq]
                # 最长的item
                max_array_len = max(max_array_len, max(len(item_data[0]) for item_data in seq_data))
                # 初始化位置
                batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data[0]), max_array_len)
                    batch_data[j, :, :actual_len] = item_data
  
                return torch.from_numpy(batch_data)
            else:
                seq_data = [item[k] for item in seq]
                return torch.from_numpy(np.stack(seq_data, axis=0))
            
        def user_padding(seq, k):
            k_type = len(seq[0][k].shape)
            if k_type == 1 and k not in emb_feats:
                max_array_len = 0
                # batch 里的每个用户
                # 每个用户每有一个item，就把他放进seq_data
                seq_data = [item[k] for item in seq]
                # 最长的item
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))
                # 初始化位置
                batch_data = np.zeros((batch_size, max_array_len), dtype=np.int64)
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[j, :actual_len] = item_data
  
                return torch.from_numpy(batch_data)
            else:
                seq_data = [item[k] for item in seq]
                return torch.from_numpy(np.stack(seq_data, axis=0))
                #return torch.from_numpy(np.stack(seq_data, axis=0))

        for k in item_feats:
            seq_dict_merged[k] = padding(seq_feat, k)
        for k in user_feats:
            user_feat_merged[k] = user_padding(user_feat, k)

        return seq, user_re_id, token_type, seq_dict_merged, user_id, action_ts, t_from_prev, user_feat_merged


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 1024}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('*.json'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                                #insert_emb = np.array(insert_emb, dtype=np.float16)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(feat_id, len(mm_emb_dict[feat_id].values()))
        #print(list(mm_emb_dict[feat_id].values())[0].shape)
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict
