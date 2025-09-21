import torch
import torch.nn.functional as F

class InfoNCELoss(torch.nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, anchor, pos_emb, neg_emb, next_action_type, temperature, t_from_last, loss_mask=None):
        # anchor: [B, L, D]
        # pos_emb: [B, L, D]
        # neg_emb: [B, L, D]
        anchor = F.normalize(anchor, dim=-1, eps=1e-8)
        pos_emb = F.normalize(pos_emb, dim=-1, eps=1e-8)
        neg_emb = F.normalize(neg_emb, dim=-1, eps=1e-8)
        weights = torch.ones_like(next_action_type, dtype=torch.float32, device=next_action_type.device)
        weights[next_action_type == 0] = 0.5   # 曝光权重 0.5
        weights[next_action_type == 1] = 1.0   # 点击权重 1.0
        weights = weights * t_from_last # Log 时间衰减权重
        weights = weights.detach()
        
        #这是跨时间点的做法
        # 正样本
        pos_logits = torch.cosine_similarity(anchor, pos_emb, dim=-1).unsqueeze(-1)
        # 负样本
        neg_emb_all = neg_emb.reshape(-1, anchor.size(-1))
        neg_logits = torch.matmul(anchor, neg_emb_all.transpose(-1,-2))
        # 把正样本放最前面，方便索引
        # mask padding
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits = logits[loss_mask.bool()] / temperature
        weights = weights[loss_mask.bool()]
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64)
        loss = F.cross_entropy(logits, labels, reduction='none')
        weights = weights / (weights.mean() + 1e-8)
        loss = (loss * weights).sum() / (weights.sum() + 1e-8)
        return loss, pos_logits, neg_logits, logits

class reconInfoNCE(torch.nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, masked_log_feats, target_emb, temperature):
        masked_log_feats = F.normalize(masked_log_feats, dim = -1) 
        target_emb = F.normalize(target_emb, dim = -1)

        logits = torch.matmul(masked_log_feats, target_emb.transpose(0,1))
        logits = logits / temperature
        labels = torch.arange(target_emb.size(0)).to(device=logits.device) # 只有对角线上的才是正样本
        # 可能需要进行mask
        return F.cross_entropy(logits, labels, reduction=self.reduction)

class TripletLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, anchor, pos_emb, neg_emb, margin, loss_mask=None):
        anchor = F.normalize(anchor, dim=-1, eps=1e-8)
        pos_emb = F.normalize(pos_emb, dim=-1, eps=1e-8)
        neg_emb = F.normalize(neg_emb, dim=-1, eps=1e-8)
        pos_logits = (anchor * pos_emb).sum(dim=-1)
        neg_logits = (anchor * neg_emb).sum(dim=-1)
        loss = F.relu(neg_logits - pos_logits + margin)
        loss = loss * loss_mask
        return loss.mean()