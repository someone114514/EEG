import argparse
import copy
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

# 引入所需的数据集加载模块和模型模块
from datasets import speech_dataset, stress_dataset, bciciv2a_dataset
from datasets.config import SplitConfig
from models import model_for_speech, model_for_stress, model_for_bciciv2a

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', 'y', '1')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test-Time Training Adaptation")
    parser.add_argument('--cuda', type=int, default=0, help='cuda device index')
    parser.add_argument('--downstream_dataset', type=str, default='BCIC2020-3',
                        help='name of the downstream dataset')
    parser.add_argument('--datasets_dir', type=str, default='/work/projects/project02629/datasets/processed/BCIC2020-3/processed',
                        help='path to the processed dataset directory')
    parser.add_argument('--num_of_classes', type=int, default=5, help='number of classes (use correct value for dataset)')
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        help='classifier head type (should match model trained configuration)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate in classifier head')
    parser.add_argument('--use_pretrained_weights', type=str2bool, default=True,
                        help='whether to load foundation model weights before fine-tuned weights')
    parser.add_argument('--foundation_dir', type=str, default='pretrained_weights/pretrained_weights.pth',
                        help='path to foundation model weights')
    parser.add_argument('--model_path', type=str, required=True, help='path to fine-tuned model weights (.pth file)')
    parser.add_argument('--split_config', type=str, help='split_config')
    parser.add_argument('--test_split', type=int, default=5, help='test_split for cueless')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for test loader (default 1 for TTT)')
    parser.add_argument('--ttt_lr', type=float, default=1e-4, help='learning rate for test-time training step')
    parser.add_argument('--ttt_steps', type=int, default=1,
                        help='Number of Test-Time Training self-supervised update steps per test sample (default: 1).')
    parser.add_argument('--pretext', type=str, default='all', help='self-supervised task(s) for TTT [none, band, channel, temporal, all]')
    parser.add_argument('--online', type=str2bool, default=False, help='whether to use online (cumulative) adaptation across test samples')
    parser.add_argument('--pretext_weight_band', type=float, default=0.1)
    parser.add_argument('--pretext_weight_amp', type=float, default=0.6)
    parser.add_argument('--pretext_weight_phase', type=float, default=0.6)
    parser.add_argument('--pretext_weight_reverse', type=float, default=0.6)
    parser.add_argument('--pretext_weight_temporal', type=float, default=0.3)
    parser.add_argument('--pretext_weight_channel', type=float, default=0.2)

    args = parser.parse_args()

    # 设置下游数据集正确的类别数
    if args.downstream_dataset == 'BCIC2020-3':
        args.num_of_classes = 5
    elif args.downstream_dataset == 'MentalArithmetic':
        args.num_of_classes = 2
    elif args.downstream_dataset == 'BCIC-IV-2a':
        args.num_of_classes = 4

    # 固定CUDA设备
    torch.cuda.set_device(args.cuda)

    # 根据数据集选择正确的模型和数据加载器
    if args.downstream_dataset == 'MentalArithmetic':
        load_dataset = stress_dataset.LoadDataset(args)
        data_loader = load_dataset.get_data_loader()
        model = model_for_stress.Model(args)
    elif args.downstream_dataset == 'BCIC2020-3':
        config = yaml.safe_load(open(args.split_config, "r"))
        config = SplitConfig(**config)
        load_dataset = speech_dataset.LoadDataset(args)
        data_loader = load_dataset.get_data_loader(config)
        model = model_for_speech.Model(args)
    elif args.downstream_dataset == 'BCIC-IV-2a':
        # BCIC-IV-2a 不需要 SplitConfig，直接加载数据集
        load_dataset = bciciv2a_dataset.LoadDataset(args)
        data_loader = load_dataset.get_data_loader()
        model = model_for_bciciv2a.Model(args)
    else:
        raise ValueError(f"Dataset {args.downstream_dataset} not supported in TTT")

    # DataLoader的batch_size为1，逐样本迭代
    test_loader = DataLoader(
        data_loader['test'].dataset,
        batch_size=1, shuffle=False,
        collate_fn=data_loader['test'].dataset.collate
    )

    # 不加载预训练基础权重（直接用finetuned权重）
    args.use_pretrained_weights = False
    model = model.cuda()
    # 加载微调后的模型参数
    model.load_state_dict(torch.load(args.model_path, map_location=f'cuda:{args.cuda}'))
    model.eval()  # 模型设为eval模式以禁用Dropout等
    print("Loaded model weights from", args.model_path)

    # 准备评估指标收集
    truths = []
    preds = []
    
    # 确定要使用的自监督任务列表
    if args.pretext == 'none':
        task_list = []
    elif args.pretext == 'all':
        if args.downstream_dataset == 'MentalArithmetic':
            task_list = ['band', 'channel']  # MentalArithmetic只支持band和channel
        elif args.downstream_dataset == 'BCIC2020-3':
            task_list = ['band', 'amp']
        elif args.downstream_dataset == 'BCIC-IV-2a':
            task_list = ['band', 'temporal']  # BCIC-IV-2a支持band和temporal
    else:
        task_list = [args.pretext]

    print(f"Using pretext tasks: {task_list}")

    # 遍历测试集中每个样本
    for data, label in test_loader:
        x = data.cuda()
        y_true = label.item()
        truths.append(y_true)

        if len(task_list) == 0 or args.ttt_steps <= 0:
            # 无自监督任务或步数为0，直接进行推理
            model.eval()
            with torch.no_grad():
                logits = model(x)
                if args.downstream_dataset == 'MentalArithmetic':
                    # 二分类任务，使用sigmoid激活函数处理标量输出
                    pred_label = (torch.sigmoid(logits) > 0.5).int().item()
                else:
                    # 多分类任务，使用argmax
                    pred_label = torch.argmax(logits, dim=-1).item()
            preds.append(pred_label)
            continue
        
        # 保存模型当前权重，以便后续恢复（如果非 online 模式）
        if not args.online:
            original_state = copy.deepcopy(model.state_dict())

        # 冻结主任务分类头，仅更新骨干网络和自监督任务头
        for param in model.classifier.parameters():
            param.requires_grad = False
        
        # 启用训练模式
        model.train()
        
        # 准备优化器
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.ttt_lr)

        # 多步更新        
        steps = max(int(args.ttt_steps), 0)
        for _ in range(steps):
            optimizer.zero_grad()
            b, ch, seg, pts = x.shape  # b=1
            x_flat = x.view(b, ch, seg * pts)  # 展平
            
            if args.pretext == 'band':
                # Band任务
                sfreq_tensor = torch.tensor([200], device=x.device)
                fx, band_idx = model.band.reject_band(x_flat, sfreq_tensor)
                band_label = torch.tensor([band_idx], device=x.device).long()
                feats = model.backbone(fx.view(b, ch, seg, pts))
                band_feats = feats.mean(dim=1)
                band_feats_flat = band_feats.view(b, -1)
                band_logits = model.band.classifier(band_feats_flat)
                loss = F.cross_entropy(band_logits, band_label)
            elif args.pretext == 'channel':
                # Channel任务
                x_ch_shuffle, ch_label = model.channel.swap_one_ap_pair(x_flat)
                channel_label = torch.tensor([ch_label], device=x.device).long()
                feats = model.backbone(x_ch_shuffle.view(b, ch, seg, pts))
                channel_feats = feats.mean(dim=1)
                channel_feats_flat = channel_feats.view(b, -1)
                channel_logits = model.channel.classifier(channel_feats_flat)
                loss = F.cross_entropy(channel_logits, channel_label)
            elif args.pretext == 'temporal':
                # Temporal任务
                xr = torch.zeros_like(x_flat)
                temporal_labels = []
                for i in range(b):
                    xr_i, lbl = model.temporal.rearrange_or_not(x_flat[i].unsqueeze(0))
                    xr[i] = xr_i
                    temporal_labels.append(lbl)
                temporal_labels = torch.tensor(temporal_labels, device=x.device).long()
                
                feats_temporal = model.backbone(xr.view(b, ch, seg, pts))
                temporal_feats = feats_temporal.mean(dim=1).view(b, -1)
                temporal_logits = model.temporal.classifier(temporal_feats)
                loss = F.cross_entropy(temporal_logits, temporal_labels)
            elif args.pretext == 'all':
                if args.downstream_dataset == 'MentalArithmetic':
                    # Band任务
                    sfreq_tensor = torch.tensor([200], device=x.device)
                    fx, band_idx = model.band.reject_band(x_flat, sfreq_tensor)
                    band_label = torch.tensor([band_idx], device=x.device).long()
                    feats_band = model.backbone(fx.view(b, ch, seg, pts))
                    band_feats = feats_band.mean(dim=1)
                    band_feats_flat = band_feats.view(b, -1)
                    band_logits = model.band.classifier(band_feats_flat)
                    loss_band = F.cross_entropy(band_logits, band_label)
                    
                    # Channel任务
                    x_ch_shuffle, ch_label = model.channel.swap_one_ap_pair(x_flat)
                    channel_label = torch.tensor([ch_label], device=x.device).long()
                    feats_channel = model.backbone(x_ch_shuffle.view(b, ch, seg, pts))
                    channel_feats = feats_channel.mean(dim=1)
                    channel_feats_flat = channel_feats.view(b, -1)
                    channel_logits = model.channel.classifier(channel_feats_flat)
                    loss_channel = F.cross_entropy(channel_logits, channel_label)
                    
                    # 总损失
                    loss = (args.pretext_weight_band * loss_band + 
                            args.pretext_weight_channel * loss_channel)
                elif args.downstream_dataset == 'BCIC2020-3':
                    # Band任务
                    sfreq_tensor = torch.tensor([200], device=x.device)
                    fx, band_idx = model.band.reject_band(x_flat, sfreq_tensor)
                    band_label = torch.tensor([band_idx], device=x.device).long()
                    feats = model.backbone(fx.view(b, ch, seg, pts))
                    band_feats = feats.mean(dim=1)
                    band_feats_flat = band_feats.view(b, -1)
                    band_logits = model.band.classifier(band_feats_flat)
                    loss_band = F.cross_entropy(band_logits, band_label)
                    
                    # Amp任务
                    x_scaled, scale_label = model.amp.scale_amp(x_flat)
                    amp_label = torch.tensor([scale_label], device=x.device).long()
                    feats = model.backbone(x_scaled.view(b, ch, seg, pts))
                    amp_feats = feats.mean(dim=1)
                    amp_feats_flat = amp_feats.view(b, -1)
                    amp_logits = model.amp.classifier(amp_feats_flat)
                    loss_amp = F.cross_entropy(amp_logits, amp_label)
                    
                    loss = (args.pretext_weight_band * loss_band +
                            args.pretext_weight_amp * loss_amp)
                elif args.downstream_dataset == 'BCIC-IV-2a':
                    # Band任务
                    sfreq_tensor = torch.tensor([200], device=x.device)
                    fx, band_idx = model.band.reject_band(x_flat, sfreq_tensor)
                    band_label = torch.tensor([band_idx], device=x.device).long()
                    feats_band = model.backbone(fx.view(b, ch, seg, pts))
                    band_feats = feats_band.mean(dim=1)
                    band_feats_flat = band_feats.view(b, -1)
                    band_logits = model.band.classifier(band_feats_flat)
                    loss_band = F.cross_entropy(band_logits, band_label)
                    
                    # Temporal任务
                    xr = torch.zeros_like(x_flat)
                    temporal_labels = []
                    for i in range(b):
                        xr_i, lbl = model.temporal.rearrange_or_not(x_flat[i].unsqueeze(0))
                        xr[i] = xr_i
                        temporal_labels.append(lbl)
                    temporal_labels = torch.tensor(temporal_labels, device=x.device).long()

                    feats_temporal = model.backbone(xr.view(b, ch, seg, pts))
                    temporal_feats = feats_temporal.mean(dim=1).view(b, -1)
                    temporal_logits = model.temporal.classifier(temporal_feats)
                    loss_temporal = F.cross_entropy(temporal_logits, temporal_labels)
                    
                    # 总损失
                    loss = (args.pretext_weight_band * loss_band +
                            args.pretext_weight_temporal * loss_temporal)
            else:
                # 未指定已知的自监督任务，跳过
                continue

            # 反向传播并单步更新
            loss.backward()
            optimizer.step()

        # 恢复模型评估模式
        model.eval()
        # 用更新后的模型对原始样本进行主任务预测
        with torch.no_grad():
            logits = model(x)
            if args.downstream_dataset == 'MentalArithmetic':
                # 二分类任务，使用sigmoid激活函数处理标量输出
                pred_label = (torch.sigmoid(logits) > 0.5).int().item()
            else:
                # 多分类任务，使用argmax
                pred_label = torch.argmax(logits, dim=-1).item()
        preds.append(pred_label)
        
        # 恢复模型参数到初始状态（如果不是online模式）
        if not args.online:
            model.load_state_dict(original_state)
        
        # 重置参数的requires_grad状态（解冻分类头）
        for param in model.classifier.parameters():
            param.requires_grad = True

    # 计算指标：Balanced Accuracy、Cohen's Kappa、加权F1等
    truths_arr = np.array(truths)
    preds_arr = np.array(preds)
    acc = balanced_accuracy_score(truths_arr, preds_arr)
    kappa = cohen_kappa_score(truths_arr, preds_arr)
    f1 = f1_score(truths_arr, preds_arr, average='weighted')
    cm = confusion_matrix(truths_arr, preds_arr)
    
    # 输出结果
    print(f"******** Results STEPS:{args.ttt_steps}, LR: {args.ttt_lr}, B: {args.batch_size} ********")
    print(f"Balanced Accuracy: {acc:.5f}, Cohen Kappa: {kappa:.5f}, Weighted F1: {f1:.5f}")
    print("Confusion Matrix:")
    print(cm)
