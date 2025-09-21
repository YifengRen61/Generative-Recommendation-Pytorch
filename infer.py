import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # 添加F模块用于归一化
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset, save_emb
from hstu import HSTUModel


def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--complex_feat_emb', default=True, action='store_true')
    parser.add_argument('--info_tau', default=0.07, type=float)
    parser.add_argument('--lambda_info', default=1, type=float)

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args


def read_result_ids(file_path):
    with open(file_path, 'rb') as f:
        # Read the header (num_points_query and FLAGS_query_ann_top_k)
        num_points_query = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes
        query_ann_top_k = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes

        print(f"num_points_query: {num_points_query}, query_ann_top_k: {query_ann_top_k}")

        # Calculate how many result_ids there are (num_points_query * query_ann_top_k)
        num_result_ids = num_points_query * query_ann_top_k

        # Read result_ids (uint64_t, 8 bytes per value)
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)

        return result_ids.reshape((num_points_query, query_ann_top_k))


def process_cold_start_feat(feat):
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


# def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model):
def get_candidate_emb(
        indexer,  
        item_feat_default_value, 
        item_feat_types, 
        mm_emb_dict, 
        model, 
        batch_size=1024
        ):
    """
    生产候选库item的id和embedding，并返回嵌入张量、检索ID列表和映射字典。

    Args:
        indexer: 索引字典
        feat_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feature_default_value: 特征缺省值
        mm_emb_dict: 多模态特征字典
        model: 模型
        batch_size: 批次大小
    Returns:
        item_embs: 项目嵌入张量 [num_items, emb_dim]
        retrieval_ids_list: 检索ID列表，与item_embs顺序一致
        retrieve_id2creative_id: 检索ID->creative_id的映射字典
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}

    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            # 读取item特征，并补充缺失值
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer[creative_id] if creative_id in indexer else 0
            missing_fields = set(
                item_feat_types['item_sparse'] + item_feat_types['item_array'] + item_feat_types['item_continual']
            ) - set(feature.keys())
            feature = process_cold_start_feat(feature)
            for feat_id in missing_fields:
                feature[feat_id] = item_feat_default_value[feat_id]
            for feat_id in item_feat_types['item_emb']:
                if creative_id in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][creative_id]
                else:
                    feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    # 生成项目嵌入
    all_embs = []
    retrieval_ids_list = []  # 保存检索ID，与嵌入顺序一致
    for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Generating item embeddings"):
        end_idx = min(start_idx + batch_size, len(item_ids))
        item_seq = torch.tensor(item_ids[start_idx:end_idx], device=model.dev).unsqueeze(0)
        batch_feat = []
        for i in range(start_idx, end_idx):
            batch_feat.append(features[i])
        
        # 处理特征填充
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
                    max_array_len = 0
                    for i in range(batch_size):
                        seq_data = [item[k] for item in seq_feat[i]]
                        max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))
                    batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
                    for i in range(batch_size):
                        seq_data = [item[k] for item in seq_feat[i]]
                        for j, item_data in enumerate(seq_data):
                            actual_len = min(len(item_data), max_array_len)
                            batch_data[i, j, :actual_len] = item_data[:actual_len]
                    final_dict[k] = torch.from_numpy(batch_data)
                else:
                    batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)
                    for i in range(batch_size):
                        seq_data = [item[k] for item in seq_feat[i]]
                        batch_data[i] = seq_data
                    final_dict[k] = torch.from_numpy(batch_data)
            return final_dict

        batch_feat_padded = padding([batch_feat])  # 序列长度为1的填充
        with torch.no_grad():
            batch_emb = model.itemfeat2emb(item_seq, batch_feat_padded).squeeze(0)
            batch_emb = F.normalize(batch_emb, dim=-1, eps=1e-8)
            all_embs.append(batch_emb)
        retrieval_ids_list.extend(retrieval_ids[start_idx:end_idx])
    
    item_embs = torch.cat(all_embs, dim=0)  # [num_items, emb_dim]
    return item_embs, retrieval_ids_list, retrieve_id2creative_id



def infer():
    args = get_args()
    data_path = os.environ.get('EVAL_DATA_PATH')
    test_dataset = MyTestDataset(data_path, args)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn
    )
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    user_feat_statistics, user_feat_types = test_dataset.user_feat_statistics, test_dataset.user_feature_types
    item_feat_statistics, item_feat_types = test_dataset.item_feat_statistics, test_dataset.item_feature_types
    model = HSTUModel(
        usernum, 
        itemnum, 
        user_feat_statistics, 
        user_feat_types, 
        item_feat_statistics, 
        item_feat_types, 
        args
        ).to(args.device)
    model.eval()

    ckpt_path = get_ckpt_path()
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))

    # 生成候选库的embedding和映射字典
    # retrieve_id2creative_id = get_candidate_emb(
    item_embs, retrieval_ids_list, retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.item_feature_default_value, 
        test_dataset.item_feature_types,
        test_dataset.mm_emb_dict,
        model,
        args.batch_size
    )

    all_embs = []
    user_list = []
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        seq, user_re_id, token_type, seq_feat, user_id, action_ts, t_from_prev, user_feat_merged = batch
        seq = seq.to(args.device)
        action_ts = action_ts.to(args.device)
        t_from_prev = t_from_prev.to(args.device)
        logits = model.predict(seq, user_re_id, seq_feat, token_type, action_ts, t_from_prev, user_feat_merged)
        for i in range(logits.shape[0]):
            emb = logits[i].unsqueeze(0).detach()  # 保持在GPU上
            # emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
            all_embs.append(emb)
        user_list += user_id

    query_embs = torch.cat(all_embs, dim=0)  # [num_query, emb_dim]
    query_embs = F.normalize(query_embs, dim=-1, eps=1e-8)  # 归一化查询嵌入

    # 分批计算相似度矩阵以避免内存溢出
    batch_size_query = 512  # 每次处理的查询数量
    topk_indices_all = []
    for i in range(0, query_embs.shape[0], batch_size_query):
        query_batch = query_embs[i:i+batch_size_query]
        sim_batch = torch.matmul(query_batch, item_embs.t())  # [batch_query, num_items]
        topk_values_batch, topk_indices_batch = torch.topk(sim_batch, k=10, dim=1)
        topk_indices_all.append(topk_indices_batch)
    topk_indices = torch.cat(topk_indices_all, dim=0)  # [num_query, 10]

    # 映射到creative_id
    top10s = []
    for i in range(topk_indices.shape[0]):
        creative_list = []
        for idx in topk_indices[i]:
            retrieval_id = retrieval_ids_list[idx]  # 获取检索ID
            creative_id = retrieve_id2creative_id.get(retrieval_id, 0)
            creative_list.append(creative_id)
        top10s.append(creative_list)

    return top10s, user_list
