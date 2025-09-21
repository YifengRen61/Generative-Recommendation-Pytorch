import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import random

#from hstu_old_speed import HSTUModel
from hstu import HSTUModel
from dataset import MyDataset
from transformers import get_cosine_schedule_with_warmup
from InfoNCE import InfoNCELoss

def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=128, type=int)
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
    # 默认读取 81
    #parser.add_argument('--mm_emb_id', nargs='+', default=[str(s) for s in range(81, 84)], type=str, choices=[str(s) for s in range(81, 87)])
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 设置随机种子
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    return args

if __name__ == '__main__':
    # 手动设置路径
    os.environ['TRAIN_LOG_PATH'] = './logs'
    os.environ['TRAIN_TF_EVENTS_PATH'] = './tfevents'
    os.environ['TRAIN_DATA_PATH'] = './data/TencentGR_1k'
    os.environ['TRAIN_CKPT_PATH'] = './ckpt'

    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)

    # 日志上报到 TRAIN_LOG_PATH 下的 trian.log
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')

    # tensorboard 
    # 用于将训练过程中的指标（如损失、准确率等）写入日志文件
    # 这些日志可以被TensorBoard可视化工具读取和展示
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()

    # 读取数据集
    dataset = MyDataset(data_path, args)

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn, drop_last=True
    )

    usernum, itemnum = dataset.usernum, dataset.itemnum
    user_feat_statistics, user_feat_types = dataset.user_feat_statistics, dataset.user_feature_types
    item_feat_statistics, item_feat_types = dataset.item_feat_statistics, dataset.item_feature_types
    # 构建 Baseline 模型
    # TODO: 新增模型可以写到这里
    #model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model = HSTUModel(
        usernum, 
        itemnum, 
        user_feat_statistics, 
        user_feat_types,
        item_feat_statistics,
        item_feat_types,
        args
        ).to(args.device)
    
    # 初始化 神经网络 模型参数
    for name, param in model.named_parameters():
        try:
            if 'qkvu_proj' not in name and '_rel_attn_bias' not in name:
                torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    # 初始化 embedding
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    # 稀疏特征嵌入
    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    # default 是 None？
    if args.state_dict_path is not None:
        try:
            # 加载预训练模型状态字典：
            # 1. torch.load() 从指定路径加载模型参数
            # 2. map_location 确保参数加载到正确的设备(CPU/GPU)
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))

            # 从模型路径中解析出训练停止时的epoch值：
                # 1. 查找路径中'epoch='字符串的位置
                # 2. 提取'epoch='后面的数字部分
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            
            # 设置当前训练起始epoch：
                # 1. 从路径中提取的epoch值转换为整数
                # 2. +1 表示从上次训练结束的下一个epoch开始
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    # 初始化 BCEWithLogitsLoss 损失函数
    # TODO: 这是个什么损失函数？
    infoNce_criterion = InfoNCELoss(reduction='mean')
    # 初始化 Adam 优化器 和 AMP缩放器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98),weight_decay=1e-4)

    scalar = GradScaler()

    # 初始化 学习率调度器
    # 1. total_training_steps 总训练步数
    # 2. warmup_steps 预热步数
    total_training_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(0.05 * total_training_steps)

    # 3. get_cosine_schedule_with_warmup() 初始化一个余弦退火学习率调度器
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, user_id, token_type, next_token_type, action_ts, t_from_prev, t_from_last, next_action_type, seq_feat, pos_feat, neg_feat, user_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)

            optimizer.zero_grad()

            # 使用 自动混合精度（AMP）加快训练速度，减少显存使用
            with autocast(device_type=args.device, dtype=torch.float16):
                pos_embs, neg_embs, anchor = model(
                    seq, pos, neg, user_id, token_type, next_token_type, action_ts, t_from_prev, next_action_type, seq_feat, pos_feat, neg_feat, user_feat
                )
                mask = (next_token_type == 1).to(args.device)
                next_action_type = next_action_type.to(args.device)
                t_from_last = t_from_last.to(args.device)
                loss_cl, pos_logits, neg_logits, logits = infoNce_criterion(anchor, pos_embs, neg_embs, next_action_type, args.info_tau, t_from_last, mask)
                loss = args.lambda_info * loss_cl
            # 在训练循环中计算指标
            with torch.no_grad():
                # 计算训练指标
                logits_topK, idx = torch.topk(logits, 10, dim=1, largest=True)
                mask = (idx == 0).any(dim=1)
                count = mask.sum()
                acc_at_10 = count / logits.size(0)

            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            # 有没有必要加一些tensorboard的指标
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Loss/infoNCE', loss_cl.item(), global_step)
            writer.add_scalar("Model/nce_neg_logits", neg_logits.mean().item(), global_step)
            writer.add_scalar("Model/nce_pos_logits", pos_logits.mean().item(), global_step)

            writer.add_scalar('Metrics/ACC@10/train', acc_at_10, global_step)
                    
            global_step += 1

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)

            # 替换AMP所需的反向传播和更新缩放器
            # https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
            scalar.scale(loss).backward()
            scalar.unscale_(optimizer) # 需要取消缩放才能使用梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # 梯度裁剪，防止梯度爆炸
            scalar.step(optimizer)
            scalar.update()
            lr_scheduler.step() # 更新学习率

            # 在每次训练步骤后，记录学习率到 TensorBoard
            writer.add_scalar('Learning Rate/train', lr_scheduler.get_last_lr()[0], global_step)
            
            # loss.backward()
            # optimizer.step()

        model.eval()
        valid_loss_sum = 0
        valid_loss_sum = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                seq, pos, neg, user_id, token_type, next_token_type, action_ts, t_from_prev, t_from_last, next_action_type, seq_feat, pos_feat, neg_feat, user_feat = batch
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)

                with autocast(device_type=args.device, dtype=torch.float16):
                    pos_embs, neg_embs, anchor = model(
                        seq, pos, neg, user_id, token_type, next_token_type, action_ts, t_from_prev, next_action_type, seq_feat, pos_feat, neg_feat, user_feat
                    )
                    mask = (next_token_type == 1).to(args.device)
                    next_action_type = next_action_type.to(args.device)
                    t_from_last = t_from_last.to(args.device)
                    loss_cl, _, _, logits = infoNce_criterion(anchor, pos_embs, neg_embs, next_action_type, args.info_tau, t_from_last, mask)
                    loss = args.lambda_info * loss_cl
                    valid_loss_sum += loss.item()
                
                # 计算指标
                logits_topK, idx = torch.topk(logits, 10, dim=1, largest=True)
                mask = (idx == 0).any(dim=1)            
                count = mask.sum()
                acc_at_10 = count / logits.size(0)

        valid_loss_sum /= len(valid_loader)

        # 记录到TensorBoard
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)
        writer.add_scalar('ValidMetrics/ACC@10/valid', acc_at_10, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
