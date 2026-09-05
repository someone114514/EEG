import copy
import os
import json
import re
from timeit import default_timer as timer

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast  # 引入混合精度工具
from tqdm import tqdm

from finetune_evaluator import Evaluator

class LoRALayer(nn.Module):
    """LoRA适配层，可以附加到任何线性层"""
    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA参数
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        
        # 添加dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """返回LoRA的输出: x @ A @ B * scaling，并应用dropout"""
        lora_output = self.dropout(x @ self.lora_A) @ self.lora_B
        return lora_output * self.scaling
    
def add_lora_to_multihead_attention(attention_module, r=4, alpha=1.0, dropout=0.0):
    """
    给现有的MultiheadAttention模块添加LoRA适配，不改变原有结构
    
    Args:
        attention_module: 原有的nn.MultiheadAttention实例
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout rate
    """
    embed_dim = attention_module.embed_dim
    
    # 获取原始模块的设备和数据类型
    device = next(attention_module.parameters()).device
    dtype = next(attention_module.parameters()).dtype
    
    # 冻结原始参数
    for param in attention_module.parameters():
        param.requires_grad = False
    
    # 为q、k、v和out_proj添加LoRA层，并移到正确设备
    attention_module.lora_q = LoRALayer(embed_dim, embed_dim, r=r, alpha=alpha, dropout=dropout).to(device=device, dtype=dtype)
    attention_module.lora_k = LoRALayer(embed_dim, embed_dim, r=r, alpha=alpha, dropout=dropout).to(device=device, dtype=dtype)
    attention_module.lora_v = LoRALayer(embed_dim, embed_dim, r=r, alpha=alpha, dropout=dropout).to(device=device, dtype=dtype)
    attention_module.lora_out_proj = LoRALayer(embed_dim, embed_dim, r=r, alpha=alpha, dropout=dropout).to(device=device, dtype=dtype)
    
    # 保存原始的forward方法
    attention_module._original_forward = attention_module.forward
    
    # 替换forward方法
    def lora_forward(query, key, value, key_padding_mask=None, need_weights=True, 
                    attn_mask=None, average_attn_weights=True, is_causal=False):
        
        # 检查输入格式
        is_batched = query.dim() == 3
        if not is_batched:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
        
        # 获取原始的qkv投影结果
        # 手动进行in_proj操作以便添加LoRA
        if attention_module.in_proj_weight is not None:
            # 使用in_proj_weight的情况
            w_q, w_k, w_v = attention_module.in_proj_weight.chunk(3, dim=0)
            
            if attention_module.in_proj_bias is not None:
                b_q, b_k, b_v = attention_module.in_proj_bias.chunk(3, dim=0)
            else:
                b_q = b_k = b_v = None
            
            # 原始投影
            q = F.linear(query, w_q, b_q)
            k = F.linear(key, w_k, b_k)  
            v = F.linear(value, w_v, b_v)
            
            # 添加LoRA适配
            q = q + attention_module.lora_q(query)
            k = k + attention_module.lora_k(key)
            v = v + attention_module.lora_v(value)
        else:
            # 使用独立qkv权重的情况（较少见）
            q = F.linear(query, attention_module.q_proj_weight, attention_module.in_proj_bias)
            k = F.linear(key, attention_module.k_proj_weight, None)
            v = F.linear(value, attention_module.v_proj_weight, None)
            
            q = q + attention_module.lora_q(query)
            k = k + attention_module.lora_k(key)
            v = v + attention_module.lora_v(value)
        
        # 执行注意力计算
        tgt_len, bsz, embed_dim = q.shape
        src_len = k.shape[0]
        
        head_dim = embed_dim // attention_module.num_heads
        scaling = float(head_dim) ** -0.5
        
        # reshape for multi-head
        q = q.contiguous().view(tgt_len, bsz * attention_module.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * attention_module.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * attention_module.num_heads, head_dim).transpose(0, 1)
        
        # attention weights
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        attn_output_weights = attn_output_weights * scaling
        
        if attn_mask is not None:
            attn_output_weights += attn_mask
        
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, attention_module.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
            attn_output_weights = attn_output_weights.view(bsz * attention_module.num_heads, tgt_len, src_len)
        
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=attention_module.dropout, training=attention_module.training)
        
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        
        # 输出投影+LoRA
        attn_output = F.linear(attn_output, attention_module.out_proj.weight, attention_module.out_proj.bias)
        attn_output = attn_output + attention_module.lora_out_proj(attn_output)

        if not is_batched:
            attn_output = attn_output.squeeze(1)
            if need_weights:
                attn_output_weights = attn_output_weights.squeeze(1)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, attention_module.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
            if not is_batched:
                attn_output_weights = attn_output_weights.squeeze(0)
            return attn_output, attn_output_weights
        else:
            return attn_output, None
    
    # 绑定新的forward方法
    attention_module.forward = lora_forward
    
    return attention_module
def add_lora_to_linear(linear_module, r=4, alpha=1.0, dropout=0.0):
    """给普通的Linear层添加LoRA"""
    in_features = linear_module.in_features
    out_features = linear_module.out_features
    
    # 获取原始模块的设备和数据类型
    device = next(linear_module.parameters()).device
    dtype = next(linear_module.parameters()).dtype
    
    # 冻结原始参数
    for param in linear_module.parameters():
        param.requires_grad = False
    
    # 添加LoRA层
    linear_module.lora_layer = LoRALayer(in_features, out_features, r=r, alpha=alpha, dropout=dropout).to(device=device, dtype=dtype)
    
    # 保存原始forward
    linear_module._original_forward = linear_module.forward
    
    # 替换forward方法
    def lora_forward(x):
        original_output = linear_module._original_forward(x)
        lora_output = linear_module.lora_layer(x)
        return original_output + lora_output
    
    linear_module.forward = lora_forward
    return linear_module

def apply_lora_to_model(model, r=4, alpha=1.0, dropout=0.0):
    """
    递归地给模型中所有的MultiheadAttention添加LoRA
    只针对QKV投影
    """
    # 确保模型在正确设备上
    device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')
    
    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention):
            add_lora_to_multihead_attention(module, r=r, alpha=alpha, dropout=dropout)
            # print(f"Added LoRA to MultiheadAttention: {name}")
        else:
            # 递归处理子模块
            apply_lora_to_model(module, r=r, alpha=alpha, dropout=dropout)
    
    return model


def get_lora_parameters(model):
    """获取所有LoRA参数"""
    lora_params = []
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            if hasattr(module, 'lora_q'):
                lora_params.extend([module.lora_q.lora_A, module.lora_q.lora_B])
                lora_params.extend([module.lora_k.lora_A, module.lora_k.lora_B])
                lora_params.extend([module.lora_v.lora_A, module.lora_v.lora_B])
    return lora_params

def freeze_non_lora_parameters(model):
    """冻结所有非LoRA参数"""
    for name, param in model.named_parameters():
        # 只保留LoRA(qkv)和out_proj权重为可训练
        if 'lora_' in name:
            param.requires_grad = True
        if 'backbone' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

       # self.scaler = GradScaler()

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        if self.params.downstream_dataset in ['FACED', 'SEED-V', 'PhysioNet-MI', 'ISRUC', 'BCIC2020-3', 'TUEV', 'BCIC-IV-2a', 'Chisco', 'cueless']:
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        elif self.params.downstream_dataset in ['SHU-MI', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB']:
            self.criterion = BCEWithLogitsLoss().cuda()
        elif self.params.downstream_dataset == 'SEED-VIG':
            self.criterion = MSELoss().cuda()
        else:
            raise ValueError(f"Criterion for {self.params.downstream_dataset} not implemented")

        self.best_model_states = None

        if self.params.use_lora:
            # 应用LoRA
            self.model = apply_lora_to_model(self.model, r=self.params.lora_r, alpha=self.params.lora_alpha, dropout=self.params.lora_dropout)
            freeze_non_lora_parameters(self.model)

        # 显示可训练参数统计
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        total = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(p.numel() for p in trainable)
        print(f"trainable: {trainable_count:,} / {total:,}")
        

        # 打印所有可训练参数的详细信息
        # print("\n=== Trainable Parameters ===")
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {param.shape} ({param.numel():,} params)")
        # print("=" * 30)


        # 如果使用LoRA，优化器只训练可训练参数
        if self.params.use_lora:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            trainable_params = self.model.parameters()

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr and not self.params.use_lora: 
                # 多学习率只在非LoRA模式下使用
                backbone_params = [p for name, p in self.model.named_parameters() if "backbone" in name]
                other_params = [p for name, p in self.model.named_parameters() if "backbone" not in name]
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(trainable_params, lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr and not self.params.use_lora:
                backbone_params = [p for name, p in self.model.named_parameters() if "backbone" in name]
                other_params = [p for name, p in self.model.named_parameters() if "backbone" not in name]
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ],  momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(trainable_params, lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )
        print(self.model)

    def train_for_multiclass(self, return_test_results=False):
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        cm_best = None
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                if self.params.downstream_dataset == 'ISRUC':
                    loss_class = self.criterion(pred.transpose(1, 2), y)
                else:
                    loss_class = self.criterion(pred, y)
                # Band
                if self.params.pretext == 'band':
                    if self.params.downstream_dataset == 'ISRUC':
                        B, S, C, T = x.shape
                        assert T == 30*200, f"epoch_size({T}) != 30*200"  # keep the sanity check

                        # 1) Build inputs for reject_band over time axis (C treated as channels)
                        x_flat = x.view(B*S, C, T)                         # (B*S, C, 30*200)
                        sfreq_tensor = torch.tensor([200], device=x.device)
                        x_rej, band_labels = self.model.band.reject_band(x_flat, sfreq_tensor)  # labels: (B*S,)

                        # 2) Extract features with backbone and average across channels
                        feats = self.model.backbone(x_rej.view(B*S, C, 30, 200))  # -> (B*S, C, 30, 200)
                        feats = feats.mean(dim=1).view(B*S, -1)                   # -> (B*S, 6000)

                        # 3) Band head loss (batch dims match!)
                        loss_band, _, _ = self.model.band(feats, band_labels)     # CE over (B*S, #bands)

                        # 4) Combine with main seq2seq loss
                        loss = loss_class + self.params.pretext_weight_band * loss_band
                    else:
                        # (batch, channels, total_time)
                        b, ch, seg, pts = x.shape  # seg=3, pts=200, ch=64
                        x_flat = x.view(b, ch, seg*pts)  # (b, 64, 600)
                        # 为每个样本生成带通滤波并计算Band损失
                        filtered_x = torch.zeros_like(x_flat)
                        band_labels = []
                        sfreq_tensor = torch.tensor([200], device=x.device)  # 采样率（200Hz）
                        for i in range(b):
                            # 随机去除一个频带
                            fx_i, band_idx = self.model.band.reject_band(x_flat[i].unsqueeze(0), sfreq_tensor)
                            filtered_x[i] = fx_i
                            band_labels.append(band_idx)
                        band_labels = torch.tensor(band_labels, device=x.device).long()
                        # 经backbone提取特征后计算Band任务预测
                        feats = self.model.backbone(filtered_x.view(b, ch, seg, pts))
                        band_feats = feats.mean(dim=1)               # 在通道维度上平均 (b, 3, 200)
                        band_feats_flat = band_feats.view(b, -1)     # (b, 600)
                        band_logits = self.model.band.classifier(band_feats_flat)  # (b, 7) 7个频带类别
                        loss_band = F.cross_entropy(band_logits, band_labels)
                        loss = loss_class + self.params.pretext_weight_band * loss_band
                elif self.params.pretext == 'amp':
                    # 将输入展开为 (batch, channels, total_time)
                    b, ch, seg, pts = x.shape
                    x_flat = x.view(b, ch, seg * pts)
                    # 为每个样本应用随机幅度缩放并计算 Amp 任务损失
                    scaled_x = torch.zeros_like(x_flat)
                    amp_labels = []
                    for i in range(b):
                        x_scaled_i, scale_label = self.model.amp.scale_amp(x_flat[i].unsqueeze(0))
                        scaled_x[i] = x_scaled_i
                        amp_labels.append(scale_label)
                    amp_labels = torch.tensor(amp_labels, device=x.device).long()
                    feats = self.model.backbone(scaled_x.view(b, ch, seg, pts))
                    amp_feats = feats.mean(dim=1)
                    amp_feats_flat = amp_feats.view(b, -1)
                    amp_logits = self.model.amp.classifier(amp_feats_flat)  # (b, 16)
                    loss_amp = F.cross_entropy(amp_logits, amp_labels)
                    loss = loss_class + self.params.pretext_weight_amp * loss_amp
                elif self.params.pretext == 'phase':
                    # 将输入展开为 (batch, channels, total_time)
                    b, ch, seg, pts = x.shape
                    x_flat = x.view(b, ch, seg * pts)
                    # 为每个样本应用随机相位偏移并计算 Phase 任务损失
                    shifted_x = torch.zeros_like(x_flat)
                    phase_labels = []
                    for i in range(b):
                        x_shifted_i, phase_label = self.model.phase.phase_shift(x_flat[i].unsqueeze(0))
                        shifted_x[i] = x_shifted_i
                        phase_labels.append(phase_label)
                    phase_labels = torch.tensor(phase_labels, device=x.device).long()
                    feats = self.model.backbone(shifted_x.view(b, ch, seg, pts))
                    phase_feats = feats.mean(dim=1)
                    phase_feats_flat = phase_feats.view(b, -1)
                    phase_logits = self.model.phase.classifier(phase_feats_flat)  # (b, 8)
                    loss_phase = F.cross_entropy(phase_logits, phase_labels)
                    loss = loss_class + self.params.pretext_weight_phase * loss_phase
                elif self.params.pretext == 'temporal':
                    b, ch, seg, pts = x.shape
                    x_flat = x.view(b, ch, seg * pts)

                    # 生成重排样本与标签（0 原序，1 打乱）
                    xr = torch.zeros_like(x_flat)
                    temporal_labels = []
                    for i in range(b):
                        xr_i, lbl = self.model.temporal.rearrange_or_not(x_flat[i].unsqueeze(0))
                        xr[i] = xr_i
                        temporal_labels.append(lbl)
                    temporal_labels = torch.tensor(temporal_labels, device=x.device).long()

                    # 过 backbone，与其它 pretext 一致的聚合方式
                    feats_temporal = self.model.backbone(xr.view(b, ch, seg, pts))
                    temporal_feats = feats_temporal.mean(dim=1).view(b, -1)

                    # 2 类 CE
                    temporal_logits = self.model.temporal.classifier(temporal_feats)
                    loss_temporal = F.cross_entropy(temporal_logits, temporal_labels)

                    loss = loss_class + self.params.pretext_weight_temporal * loss_temporal
                elif self.params.pretext == 'reverse':
                    # 将输入展开为 (batch, channels, total_time)
                    b, ch, seg, pts = x.shape
                    x_flat = x.view(b, ch, seg * pts)         # Flatten to (b, 64, total_time)
                    flipped_x = torch.zeros_like(x_flat)      # Placeholder for flipped data
                    reverse_labels = []
                    for i in range(b):
                        # Randomly flip all channels for sample i
                        x_flip_i, flip_label = self.model.reverse.flip_all_or_not(x_flat[i].unsqueeze(0))
                        flipped_x[i] = x_flip_i
                        reverse_labels.append(flip_label)
                    reverse_labels = torch.tensor(reverse_labels, device=x.device).long()
                    # Forward through backbone and reverse classifier
                    feats = self.model.backbone(flipped_x.view(b, ch, seg, pts))
                    reverse_feats = feats.mean(dim=1)                     # average over channel dim
                    reverse_feats_flat = reverse_feats.view(b, -1)
                    reverse_logits = self.model.reverse.classifier(reverse_feats_flat)  # (b, 2)
                    loss_reverse = F.cross_entropy(reverse_logits, reverse_labels)
                    # Combine main task loss and reverse loss
                    loss = loss_class + self.params.pretext_weight_reverse * loss_reverse
                elif self.params.pretext == 'all':
                    if self.params.downstream_dataset == 'BCIC2020-3':
                        # 自监督任务步：同时计算 Band、Amp 三个任务的损失
                        b, ch, seg, pts = x.shape
                        x_flat = x.view(b, ch, seg * pts)
                        # Band 任务
                        filtered_x = torch.zeros_like(x_flat)
                        band_labels = []
                        sfreq_tensor = torch.tensor([200], device=x.device)
                        for i in range(b):
                            fx_i, band_idx = self.model.band.reject_band(x_flat[i].unsqueeze(0), sfreq_tensor)
                            filtered_x[i] = fx_i
                            band_labels.append(band_idx)
                        band_labels = torch.tensor(band_labels, device=x.device).long()
                        feats_band = self.model.backbone(filtered_x.view(b, ch, seg, pts))
                        band_feats = feats_band.mean(dim=1)
                        band_feats_flat = band_feats.view(b, -1)
                        band_logits = self.model.band.classifier(band_feats_flat)
                        loss_band = F.cross_entropy(band_logits, band_labels)
                        # Amp 任务
                        scaled_x = torch.zeros_like(x_flat)
                        amp_labels = []
                        for i in range(b):
                            x_scaled_i, scale_label = self.model.amp.scale_amp(x_flat[i].unsqueeze(0))
                            scaled_x[i] = x_scaled_i
                            amp_labels.append(scale_label)
                        amp_labels = torch.tensor(amp_labels, device=x.device).long()
                        feats_amp = self.model.backbone(scaled_x.view(b, ch, seg, pts))
                        amp_feats = feats_amp.mean(dim=1)
                        amp_feats_flat = amp_feats.view(b, -1)
                        amp_logits = self.model.amp.classifier(amp_feats_flat)
                        loss_amp = F.cross_entropy(amp_logits, amp_labels)
                        # # Phase 任务
                        # shifted_x = torch.zeros_like(x_flat)
                        # phase_labels = []
                        # for i in range(b):
                        #     x_shifted_i, phase_label = self.model.phase.phase_shift(x_flat[i].unsqueeze(0))
                        #     shifted_x[i] = x_shifted_i
                        #     phase_labels.append(phase_label)
                        # phase_labels = torch.tensor(phase_labels, device=x.device).long()
                        # feats_phase = self.model.backbone(shifted_x.view(b, ch, seg, pts))
                        # phase_feats = feats_phase.mean(dim=1)
                        # phase_feats_flat = phase_feats.view(b, -1)
                        # phase_logits = self.model.phase.classifier(phase_feats_flat)
                        # loss_phase = F.cross_entropy(phase_logits, phase_labels)
                        # # 总自监督损失加权求和
                        total_pretext_loss = (
                            loss_class +
                            self.params.pretext_weight_band * loss_band +
                            self.params.pretext_weight_amp * loss_amp
                            # self.params.pretext_weight_phase * loss_phase
                            )
                    elif self.params.downstream_dataset == 'BCIC-IV-2a':
                        b, ch, seg, pts = x.shape
                        x_flat = x.view(b, ch, seg * pts)
                        # Band 任务
                        filtered_x = torch.zeros_like(x_flat)
                        band_labels = []
                        sfreq_tensor = torch.tensor([200], device=x.device)
                        for i in range(b):
                            fx_i, band_idx = self.model.band.reject_band(x_flat[i].unsqueeze(0), sfreq_tensor)
                            filtered_x[i] = fx_i
                            band_labels.append(band_idx)
                        band_labels = torch.tensor(band_labels, device=x.device).long()
                        feats_band = self.model.backbone(filtered_x.view(b, ch, seg, pts))
                        band_feats = feats_band.mean(dim=1)
                        band_feats_flat = band_feats.view(b, -1)
                        band_logits = self.model.band.classifier(band_feats_flat)
                        loss_band = F.cross_entropy(band_logits, band_labels)
                        # Temporal 任务
                        xr = torch.zeros_like(x_flat)
                        temporal_labels = []
                        for i in range(b):
                            xr_i, lbl = self.model.temporal.rearrange_or_not(x_flat[i].unsqueeze(0))
                            xr[i] = xr_i
                            temporal_labels.append(lbl)
                        temporal_labels = torch.tensor(temporal_labels, device=x.device).long()

                        feats_temporal = self.model.backbone(xr.view(b, ch, seg, pts))
                        temporal_feats = feats_temporal.mean(dim=1).view(b, -1)
                        # 2 类 CE
                        temporal_logits = self.model.temporal.classifier(temporal_feats)
                        loss_temporal = F.cross_entropy(temporal_logits, temporal_labels)
                        # # 总自监督损失加权求和
                        total_pretext_loss = (
                            loss_class +
                            self.params.pretext_weight_band * loss_band +
                            self.params.pretext_weight_temporal * loss_temporal
                            # self.params.pretext_weight_phase * loss_phase
                            )
                    loss = total_pretext_loss  # 本步不包含主任务损失
                else:
                    # 未使用任何自监督任务
                    loss = loss_class

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        kappa,
                        f1,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                if kappa > kappa_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                        acc,
                        kappa,
                        f1,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    kappa_best = kappa
                    f1_best = f1
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        if self.best_model_states:
            self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                    acc,
                    kappa,
                    f1,
                )
            )
            print(cm)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(best_f1_epoch, acc, kappa, f1)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)
            if return_test_results:
                path = self.params.results_dir + "test_result.json"
                os.makedirs(self.params.results_dir, exist_ok=True)
                # Create result dictionary
                result = {
                    "dataset": self.params.downstream_dataset,
                    "acc": float(acc),
                    "kappa": float(kappa),
                    "f1": float(f1),
                    "subject": getattr(self.params, 'test_subj', None),  # if you have subject info
                }
                with open(path, 'a') as f:
                    f.write(json.dumps(result) + '\n')
    def train_for_binaryclass(self, return_test_results=False):
        acc_best = 0
        roc_auc_best = 0
        pr_auc_best = 0
        cm_best = None
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)

                loss_class = self.criterion(pred, y)
                # 根据预训练任务配置，计算自监督任务损失
                if self.params.pretext == 'band':
                    b, ch, seg, pts = x.shape
                    x_flat = x.view(b, ch, seg * pts)
                    # Band 任务
                    filtered_x = torch.zeros_like(x_flat)
                    band_labels = []
                    sfreq_tensor = torch.tensor([200], device=x.device)
                    for i in range(b):
                        fx_i, band_idx = self.model.band.reject_band(x_flat[i].unsqueeze(0), sfreq_tensor)
                        filtered_x[i] = fx_i
                        band_labels.append(band_idx)
                    band_labels = torch.tensor(band_labels, device=x.device).long()
                    feats_band = self.model.backbone(filtered_x.view(b, ch, seg, pts))
                    band_feats = feats_band.mean(dim=1).view(b, -1)
                    band_logits = self.model.band.classifier(band_feats)
                    loss_band = F.cross_entropy(band_logits, band_labels)
                    loss = loss_class + self.params.pretext_weight_band * loss_band
                elif self.params.pretext == 'amp':
                    b, ch, seg, pts = x.shape
                    x_flat = x.view(b, ch, seg * pts)
                    scaled_x = torch.zeros_like(x_flat)
                    amp_labels = []
                    for i in range(b):
                        x_scaled_i, scale_label = self.model.amp.scale_amp(x_flat[i].unsqueeze(0))
                        scaled_x[i] = x_scaled_i
                        amp_labels.append(scale_label)
                    amp_labels = torch.tensor(amp_labels, device=x.device).long()
                    feats_amp = self.model.backbone(scaled_x.view(b, ch, seg, pts))
                    amp_feats = feats_amp.mean(dim=1)
                    amp_feats_flat = amp_feats.view(b, -1)
                    amp_logits = self.model.amp.classifier(amp_feats_flat)
                    loss_amp = F.cross_entropy(amp_logits, amp_labels)
                    loss = loss_class + self.params.pretext_weight_amp * loss_amp
                elif self.params.pretext == 'phase':
                    b, ch, seg, pts = x.shape
                    x_flat = x.view(b, ch, seg * pts)
                    shifted_x = torch.zeros_like(x_flat)
                    phase_labels = []
                    for i in range(b):
                        x_shifted_i, phase_label = self.model.phase.phase_shift(x_flat[i].unsqueeze(0))
                        shifted_x[i] = x_shifted_i
                        phase_labels.append(phase_label)
                    phase_labels = torch.tensor(phase_labels, device=x.device).long()
                    feats = self.model.backbone(shifted_x.view(b, ch, seg, pts))
                    phase_feats = feats.mean(dim=1)
                    phase_feats_flat = phase_feats.view(b, -1)
                    phase_logits = self.model.phase.classifier(phase_feats_flat)
                    loss_phase = F.cross_entropy(phase_logits, phase_labels)
                    loss = loss_class + self.params.pretext_weight_phase * loss_phase
                elif self.params.pretext == 'reverse':
                    b, ch, seg, pts = x.shape
                    x_flat = x.view(b, ch, seg * pts)
                    flipped_x = torch.zeros_like(x_flat)
                    reverse_labels = []
                    for i in range(b):
                        # Randomly flip all channels for sample i
                        x_flip_i, flip_label = self.model.reverse.flip_all_or_not(x_flat[i].unsqueeze(0))
                        flipped_x[i] = x_flip_i
                        reverse_labels.append(flip_label)
                    reverse_labels = torch.tensor(reverse_labels, device=x.device).long()
                    feats = self.model.backbone(flipped_x.view(b, ch, seg, pts))
                    reverse_feats = feats.mean(dim=1)
                    reverse_feats_flat = reverse_feats.view(b, -1)
                    reverse_logits = self.model.reverse.classifier(reverse_feats_flat)
                    loss_reverse = F.cross_entropy(reverse_logits, reverse_labels)
                    loss = loss_class + self.params.pretext_weight_reverse * loss_reverse
                elif self.params.pretext == 'temporal':
                    b, ch, seg, pts = x.shape
                    x_flat = x.view(b, ch, seg * pts)

                    # 生成重排样本与标签（0 原序，1 打乱）
                    xr = torch.zeros_like(x_flat)
                    temporal_labels = []
                    for i in range(b):
                        xr_i, lbl = self.model.temporal.rearrange_or_not(x_flat[i].unsqueeze(0))
                        xr[i] = xr_i
                        temporal_labels.append(lbl)
                    temporal_labels = torch.tensor(temporal_labels, device=x.device).long()

                    # 过 backbone，与其它 pretext 一致的聚合方式
                    feats_temporal = self.model.backbone(xr.view(b, ch, seg, pts))
                    temporal_feats = feats_temporal.mean(dim=1).view(b, -1)

                    # 2 类 CE
                    temporal_logits = self.model.temporal.classifier(temporal_feats)
                    loss_temporal = F.cross_entropy(temporal_logits, temporal_labels)

                    loss = loss_class + self.params.pretext_weight_temporal * loss_temporal
                elif self.params.pretext == 'channel':
                    b, ch, seg, pts = x.shape
                    x_flat = x.view(b, ch, seg * pts)
                    channel_labels = []
                    channel_shuffled_x = torch.zeros_like(x_flat)
                    for i in range(b):
                        x_ch_shuffle_i, ch_label = self.model.channel.swap_one_ap_pair(x_flat[i].unsqueeze(0))
                        channel_shuffled_x[i] = x_ch_shuffle_i
                        channel_labels.append(ch_label)
                    channel_labels = torch.tensor(channel_labels, device=x.device).long()
                    feats = self.model.backbone(channel_shuffled_x.view(b, ch, seg, pts))
                    channel_feats = feats.mean(dim=1)
                    channel_feats_flat = channel_feats.view(b, -1)
                    channel_logits = self.model.channel.classifier(channel_feats_flat)
                    loss_channel = F.cross_entropy(channel_logits, channel_labels)
                    loss = loss_class + self.params.pretext_weight_channel * loss_channel
                elif self.params.pretext == 'all':
                    # 自监督任务步：同时计算 Band、Amp、Phase 三个任务的损失
                    b, ch, seg, pts = x.shape
                    x_flat = x.view(b, ch, seg * pts)
                    # Band 任务
                    filtered_x = torch.zeros_like(x_flat)
                    band_labels = []
                    sfreq_tensor = torch.tensor([200], device=x.device)
                    for i in range(b):
                        fx_i, band_idx = self.model.band.reject_band(x_flat[i].unsqueeze(0), sfreq_tensor)
                        filtered_x[i] = fx_i
                        band_labels.append(band_idx)
                    band_labels = torch.tensor(band_labels, device=x.device).long()
                    feats_band = self.model.backbone(filtered_x.view(b, ch, seg, pts))
                    band_feats = feats_band.mean(dim=1)
                    band_feats_flat = band_feats.view(b, -1)
                    band_logits = self.model.band.classifier(band_feats_flat)
                    loss_band = F.cross_entropy(band_logits, band_labels)
                    # Channel 任务
                    channel_labels = []
                    channel_shuffled_x = torch.zeros_like(x_flat)
                    for i in range(b):
                        x_ch_shuffle_i, ch_label = self.model.channel.swap_one_ap_pair(x_flat[i].unsqueeze(0))
                        channel_shuffled_x[i] = x_ch_shuffle_i
                        channel_labels.append(ch_label)
                    channel_labels = torch.tensor(channel_labels, device=x.device).long()
                    feats = self.model.backbone(channel_shuffled_x.view(b, ch, seg, pts))
                    channel_feats = feats.mean(dim=1)
                    channel_feats_flat = channel_feats.view(b, -1)
                    channel_logits = self.model.channel.classifier(channel_feats_flat)
                    loss_channel = F.cross_entropy(channel_logits, channel_labels)
                    # 总自监督损失加权求和
                    loss = (
                        loss_class +
                        self.params.pretext_weight_band * loss_band +
                        self.params.pretext_weight_channel * loss_channel
                    )
                else:
                    loss = loss_class
                    
                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        pr_auc,
                        roc_auc,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                # if acc > acc_best:
                #     print("acc increasing....saving weights !! ")
                if roc_auc > roc_auc_best:
                    print("roc_auc increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                        acc,
                        pr_auc,
                        roc_auc,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    pr_auc_best = pr_auc
                    roc_auc_best = roc_auc
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                    acc,
                    pr_auc,
                    roc_auc,
                )
            )
            print(cm)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(best_f1_epoch, acc, pr_auc, roc_auc)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)
            if return_test_results:
                path = self.params.results_dir + "test_result.json"
                os.makedirs(self.params.results_dir, exist_ok=True)
                # Create result dictionary
                result = {
                    "dataset": self.params.downstream_dataset,
                    "acc": float(acc),
                    "pr_auc": float(pr_auc),
                    "roc_auc": float(roc_auc),
                    "subject": getattr(self.params, 'test_subj', None),  # if you have subject info
                }
                with open(path, 'a') as f:
                    f.write(json.dumps(result) + '\n')

    def train_for_regression(self):
        corrcoef_best = 0
        r2_best = 0
        rmse_best = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        corrcoef,
                        r2,
                        rmse,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                if r2 > r2_best:
                    print("r2 increasing....saving weights !! ")
                    print("Val Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                        corrcoef,
                        r2,
                        rmse,
                    ))
                    best_r2_epoch = epoch + 1
                    corrcoef_best = corrcoef
                    r2_best = r2
                    rmse_best = rmse
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                    corrcoef,
                    r2,
                    rmse,
                )
            )

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_corrcoef_{:.5f}_r2_{:.5f}_rmse_{:.5f}.pth".format(best_r2_epoch, corrcoef, r2, rmse)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)