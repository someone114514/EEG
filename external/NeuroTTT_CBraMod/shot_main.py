import argparse
import copy
import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc  
import json
import time
from datetime import datetime

# 引入所需的数据集加载模块和模型模块
from datasets import speech_dataset
from models import model_for_speech
from datasets import faced_dataset, seedv_dataset, physio_dataset, shu_dataset, isruc_dataset, chb_dataset, \
    speech_dataset, mumtaz_dataset, seedvig_dataset, stress_dataset, tuev_dataset, tuab_dataset, bciciv2a_dataset, chisco_dataset
from finetune_trainer import Trainer
from models import model_for_faced, model_for_seedv, model_for_physio, model_for_shu, model_for_isruc, model_for_chb, \
    model_for_speech, model_for_mumtaz, model_for_seedvig, model_for_stress, model_for_tuev, model_for_tuab, \
    model_for_bciciv2a, model_for_chisco


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', 'y', '1')


def main():
    parser = argparse.ArgumentParser(description="SHOT target domain adaptation training script")
    
    # Basic settings
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device index')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--downstream_dataset', type=str, default='BCIC2020-3',
                        help='name of the downstream dataset')
    parser.add_argument('--results_dir', type=str, default='./results/', help='output directory')

    # Training settings
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')

    # Data settings
    parser.add_argument('--datasets_dir', type=str, default='/work/projects/project02629/datasets/processed/BCIC2020-3/processed',
                        help='path to the processed dataset directory')
    parser.add_argument('--num_of_classes', type=int, default=5, help='number of classes (use correct value for dataset)')
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        help='classifier head type (should match model trained configuration)')

    # Model settings
    parser.add_argument('--use_pretrained_weights', type=str2bool, default=True,
                        help='whether to load foundation model weights before fine-tuned weights')
    parser.add_argument('--foundation_dir', type=str, default='pretrained_weights/pretrained_weights.pth',
                        help='path to foundation model weights')
    parser.add_argument('--model_path', type=str, required=True, help='path to fine-tuned model weights (.pth file)')


    # SHOT-specific parameters
    parser.add_argument('--mi_lambda', type=float, default=1.0, help='marginal entropy weight')
    parser.add_argument('--pl_weight', type=float, default=1.0, help='pseudo-label loss weight')
    parser.add_argument('--conf_threshold', type=float, default=0.9, help='pseudo-label confidence threshold')
    parser.add_argument('--use_sharpen', action='store_true', help='whether to use temperature sharpening')
    parser.add_argument('--temperature', type=float, default=0.5, help='sharpening temperature')

    # Logging and evaluation
    parser.add_argument('--log_interval', type=int, default=20, help='logging print interval')
    parser.add_argument('--eval_interval', type=int, default=1, help='evaluation interval (in epochs)')

    args = parser.parse_args()


    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_dir = Path(args.results_dir)          # 比如 './experiments'
    exp_dir.mkdir(parents=True, exist_ok=True)   # 确保目录存在

    # 如果下游数据集是BCIC2020-3，设置正确的类别数（5 类）
    if args.downstream_dataset == 'BCIC2020-3':
        args.num_of_classes = 5
        load_dataset = speech_dataset.LoadDataset(args)
        model = model_for_speech.Model(args)
    elif args.downstream_dataset == 'MentalArithmetic':
        args.num_of_classes = 2
        load_dataset = stress_dataset.LoadDataset(args)
        model = model_for_stress.Model(args)
    elif args.downstream_dataset == 'BCICIV2a':
        args.num_of_classes = 4
        load_dataset = bciciv2a_dataset.LoadDataset(args)
        model = model_for_bciciv2a.Model(args)

    # 固定CUDA设备
    if device == "cuda":
        torch.cuda.set_device(args.cuda)
        torch.cuda.set_device(args.cuda)

    # 获取目标域的无标签训练集和有标签测试集
    data_loaders = load_dataset.get_data_loader()
        
    
    # 不加载预训练基础权重（直接用finetuned权重）
    model = model.cuda()
    #args.use_pretrained_weights = False
    # 加载微调后的模型参数
    if args.model_path != "None":
        model.load_state_dict(torch.load(args.model_path, map_location=f'cuda:{args.cuda}'))
    model.eval()  # 模型设为eval模式以禁用Dropout等
    print("Loaded model weights from", args.model_path)

    layers = list(model.children())
    print (layers)

    if len(layers) > 1:
        feature_extractor = nn.Sequential(*layers[:-1])
        classifier = layers[-1]
        print(f"Using generic split: first {len(layers)-1} layers as feature extractor, last layer as classifier")

        feature_extractor.to(device)
        classifier.to(device)
    else:
        raise ValueError("Model structure does not match expectation, cannot split into feature extractor and classifier")
            

    # 【SHOT关键】冻结分类器参数
    for param in classifier.parameters():
        param.requires_grad = False
    classifier.eval()
        
    # 只优化特征提取器
    optimizer = SGD(
        [p for p in feature_extractor.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
        
    print("SHOT training environment setup completed:")
    print(f"- Classifier has been frozen")
    # print(f"- Optimizer: AdamW, lr={args.lr}")
    print(f"- Trainable parameters: {sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad):,}")

    """Execute SHOT training"""
    print("="*60)
    print("Start SHOT target domain adaptation training")
    print("="*60)
        
    # 设定模型为训练模式
    feature_extractor.train()
    # 冻结分类器参数
    classifier.eval()
    
    best_acc = 0.0
    best_model_state = None
    train_stats = []
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        epoch_stats = {
            'epoch': epoch,
            'losses': [],
            'mi_losses': [],
            'pl_losses': [],
            #'conf_coverages': []
        }

        # 1) 预计算全局质心：一次/每轮
        feature_extractor.eval()
        classifier.eval()

        # 整域/整轮 Self-supervised PL 
        with torch.no_grad():
             # 动态确定特征维度
            first_batch = next(iter(data_loaders['train']))
            if isinstance(first_batch, (list, tuple)):
                first_batch = first_batch[0].to(device)  # 只取数据部分
            else:
                first_batch = first_batch.to(device)
            f0 = feature_extractor(first_batch)
            D  = f0.flatten(1).size(1)                    

            sum_w   = torch.zeros(args.num_of_classes, 1, device=device)
            sum_feat= torch.zeros(args.num_of_classes, D, device=device)

            for batch in data_loaders['train']:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)  # 只取数据，忽略标签
                else:
                    x = batch.to(device)
                f = feature_extractor(x)
                feat_vec = f.flatten(1)
                logits = classifier(f)
                
                # 处理二分类情况：如果输出是1维的，转换为2维概率
                if logits.dim() == 1:
                    # 对于二分类，logits是单个值，应用sigmoid后转为2类概率
                    probs_pos = torch.sigmoid(logits)  # 正类概率
                    probs_neg = 1 - probs_pos  # 负类概率
                    p = torch.stack([probs_neg, probs_pos], dim=1)  # [N, 2]
                else:
                    p = F.softmax(logits, dim=1)

                for k in range(args.num_of_classes):
                    w = p[:, k].unsqueeze(1)          # [N,1]
                    sum_w[k] += w.sum(0)
                    sum_feat[k] += (w * feat_vec).sum(0)

            centroids = sum_feat / (sum_w + 1e-6)        # Eq.(4)

            # 2) 最近质心 -> 伪标签（硬）
            all_feats  = []
            all_yhat   = []
            for batch in data_loaders['train']:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)  # 只取数据，忽略标签
                else:
                    x = batch.to(device)
                f = feature_extractor(x)
                feat_vec = f.flatten(1)                        # [N, D]

                # 用余弦相似度做最近质心分配
                cos = F.normalize(feat_vec, dim=1) @ F.normalize(centroids, dim=1).T  # [N, K]
                y_hat = cos.argmax(1)                          # [N]

                all_feats.append(feat_vec)
                all_yhat.append(y_hat)

            feats = torch.cat(all_feats, dim=0)                # [M, D]
            yhat  = torch.cat(all_yhat, dim=0)                 # [M]

            # 3) 按硬伪标签再均值更新一次质心（Eq.(6)）
            K = args.num_of_classes
            new_sum_feat = torch.zeros(K, D, device=device)
            # 对每一类把对应样本特征相加（index_add_ 向量化）
            new_sum_feat.index_add_(0, yhat, feats)

            # 统计每类样本数（用于除法）
            counts = torch.zeros(K, device=device).scatter_add_(0, yhat, torch.ones(yhat.size(0), device=device))
            counts = counts.clamp_min(1e-6).unsqueeze(1)       # [K,1]

            new_centroids = new_sum_feat / counts              # [K, D]

            # 4) 用新质心覆盖旧质心
            centroids = new_centroids
            # 可选：用 y_hat 再均值更新一次 centroids，然后再跑一次最近质心分配

        feature_extractor.train()
            
        for batch_idx, batch in enumerate(data_loaders['train']):
        
            optimizer.zero_grad()

            # 正确处理数据：只取数据部分，忽略标签
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)  # 只取数据，忽略batch[1]（标签）
            else:
                inputs = batch.to(device)
            # 提取特征
            features = feature_extractor(inputs)
            
            # 分类预测
            logits = classifier(features)
            
            # 处理二分类情况：如果输出是1维的，转换为2维概率
            if logits.dim() == 1:
                # 对于二分类，logits是单个值，应用sigmoid后转为2类概率
                probs_pos = torch.sigmoid(logits)  # 正类概率
                probs_neg = 1 - probs_pos  # 负类概率
                probabilities = torch.stack([probs_neg, probs_pos], dim=1)  # [N, 2]
                # 为了计算损失，我们需要将logits也转为2维
                logits_2d = torch.stack([-logits, logits], dim=1)  # [N, 2]
                logits = logits_2d
            else:
                probabilities = F.softmax(logits, dim=1)
            
            # 计算互信息损失
            cond_entropy = conditional_entropy(probabilities)
            marg_entropy = marginal_entropy(probabilities)
            mi_loss = cond_entropy - args.mi_lambda * marg_entropy
            
            # 计算伪标签损失
            # 生成伪标签
            # naive pseudo-labeling table 6
            # with torch.no_grad():
            #     p = F.softmax(logits.detach(), dim=1)
                
            #     if args.use_sharpen:
            #         p = sharpen(p, args.temperature)
                
            #     confidences, pseudo_labels = p.max(dim=1)
            #     conf_mask = confidences.ge(args.conf_threshold)
                

            # if conf_mask.any():
            #     pl_loss = F.cross_entropy(logits[conf_mask], pseudo_labels[conf_mask])
            # else:
            #     pl_loss = torch.tensor(0.0, device=device)
            
   
            # 用固定centroids 生成当批伪标签
            with torch.no_grad():
                feat_vec = features.flatten(1)
                cos = F.normalize(feat_vec, dim=1) @ F.normalize(centroids, dim=1).T
                pseudo_labels = cos.argmax(1)

                # （可选）相似度覆盖率：最大相似度 ≥ 阈值的比例
                # 例如阈值 0.6，可根据数据分布调整
                # max_sim, _ = cos.max(dim=1)
                # cov = (max_sim >= getattr(args, 'centroid_sim_threshold', 0.6)).float().mean().item() * 100

            # 5. 计算伪标签损失
            pl_loss = F.cross_entropy(logits, pseudo_labels)

            # 总损失
            total_loss = mi_loss + args.pl_weight * pl_loss 
            #+ args.pretext_weight_channel * loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 记录统计信息
            epoch_stats['losses'].append(total_loss.item())
            epoch_stats['mi_losses'].append(mi_loss.item())
            epoch_stats['pl_losses'].append(pl_loss.item())
            #epoch_stats['conf_coverages'].append(conf_mask.float().mean().item() * 100)
            
            # 打印训练日志
            if (batch_idx + 1) % args.log_interval == 0:
                #conf_cov = conf_mask.float().mean().item() * 100
                print(f"Epoch [{epoch}/{args.epochs}] "
                        f"Batch [{batch_idx+1}/{len(data_loaders['train'])}] "
                        f"Loss: {total_loss.item():.4f} "
                        f"MI: {mi_loss.item():.4f} "
                        f"PL: {pl_loss.item():.4f} ")
                        #f"ConfCov: {conf_cov:.1f}%")
        
        # 每个epoch结束后的统计
        epoch_time = time.time() - epoch_start_time
        avg_loss = np.mean(epoch_stats['losses'])
        #avg_conf_cov = np.mean(epoch_stats['conf_coverages'])
        
        print(f"Epoch {epoch} completed | "
            f"Average Loss: {avg_loss:.4f} | "
            #f"Average Confidence Coverage: {avg_conf_cov:.1f}% | "
            f"Time Elapsed: {epoch_time:.1f}s"
        )
        
        # 定期评估
        if epoch % args.eval_interval == 0:
            metrics = evaluate_model(feature_extractor, classifier, data_loaders['test'], device)
            accuracy = metrics['accuracy']
            print(f"Epoch {epoch} test results: {metrics}")

            # 恢复训练模式
            feature_extractor.train()
            # classifier 若冻结且含 BN/Dropout，建议继续保持 eval()；否则这里自行决定

            # 保存最佳模型
            if accuracy > best_acc:
                best_acc = accuracy
                best_model_state = {
                    'feature_extractor': copy.deepcopy(feature_extractor.state_dict()),
                    'classifier': copy.deepcopy(classifier.state_dict()),
                    'epoch': epoch,
                    'accuracy': accuracy,
                    'metrics': metrics
                }
                save_path = exp_dir / 'best_model.pth'
                torch.save(best_model_state, save_path)
                print(f"保存最佳模型 (准确率: {best_acc:.4f})")

        #  train_stats.append(epoch_stats)

    # 保存训练统计
    # output_path = exp_dir / 'train_stats.json' 
    # with open(output_path, 'w') as f:
    #     json.dump(train_stats, f, indent=2)
    


    # 加载最佳模型进行最终评估
    if best_model_state is not None:
        feature_extractor.load_state_dict(best_model_state['feature_extractor'])
        classifier.load_state_dict(best_model_state['classifier'])

    final_metrics = evaluate_model(feature_extractor, classifier, data_loaders['test'], device)
    final_accuracy = final_metrics['accuracy']

    
    # 保存结果
    results = {
            'experiment_info': {
                'name': 'SHOT-Experiment',
                'timestamp': datetime.now().isoformat(),
                'device': str(device)
            },
            'config': vars(args),
            'training_results': {
                'best_epoch': best_model_state['epoch'] if best_model_state else -1,
                'best_accuracy': best_acc,
                'total_epochs': args.epochs
            },
            'final_evaluation': {
                'accuracy': final_accuracy,
                'balanced_accuracy': final_metrics['balanced_accuracy'],
                'f1_score': final_metrics['f1_score'],
                'cohen_kappa': final_metrics['cohen_kappa'],
                'roc_auc': final_metrics['roc_auc'],
                'pr_auc': final_metrics['pr_auc'],
                'mean_confidence': final_metrics['mean_confidence'],
                'confusion_matrix': final_metrics['confusion_matrix']
            }
    }

    results_serializable = convert(results)

    results_path =  exp_dir / 'test_results.json'
    with open(results_path, 'a') as f:
        json.dump(results_serializable, f, indent=2)

    # Print final result summary
    print("=" * 70)
    print("🎉 SHOT target domain adaptation training completed!")
    print("=" * 70)
    print(f"Experiment Name: SHOT-Experiment")
    # print(f"Source Domain -> Target Domain: {args.source_domain} -> {args.target_domain}")
    print(f"Best Epoch: {best_model_state['epoch'] if best_model_state else 'N/A'}")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Final Accuracy: {final_accuracy:.4f}")
    print(f"Balanced Accuracy: {final_metrics['balanced_accuracy']:.4f}")
    print(f"F1 Score: {final_metrics['f1_score']:.4f}")
    print(f"Cohen's Kappa: {final_metrics['cohen_kappa']:.4f}")
    print(f"Mean Confidence: {final_metrics['mean_confidence']:.4f}")
    print(f"Results saved to: {results_path}")
    print("=" * 70)



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 计算条件熵 H(Y|X)
def conditional_entropy(p_softmax):
    """Compute conditional entropy H(Y|X)"""
    eps = 1e-8
    ent = -torch.sum(p_softmax * torch.log(p_softmax + eps), dim=1)
    return ent.mean()

# 计算边际熵 H(Y)
def marginal_entropy(p_softmax):
    """Compute marginal entropy H(Y)"""
    eps = 1e-8
    p_mean = p_softmax.mean(dim=0)
    ent = -torch.sum(p_mean * torch.log(p_mean + eps))
    return ent

# 温度锐化函数
def sharpen(p_softmax, temperature=0.5):
    """Temperature sharpening"""
    p = p_softmax ** (1.0 / temperature)
    return p / p.sum(dim=1, keepdim=True)


def convert(o):
    if isinstance(o, dict):
        return {k: convert(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [convert(v) for v in o]
    elif isinstance(o, np.generic):
        return o.item()
    elif isinstance(o, np.ndarray):
        return o.tolist()
    else:
        return o

def evaluate_model(feature_extractor, classifier, data_loader, device):
    feature_extractor.eval()
    classifier.eval()

    all_predictions, all_labels, all_confidences, all_scores = [], [], [], []

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)

            features = feature_extractor(inputs)
            logits   = classifier(features)
            
            # 处理二分类输出（logits是1维的）
            if logits.dim() == 1:
                # 对于二分类，将1维logits转换为2维概率
                probs_class1 = torch.sigmoid(logits)  # 使用sigmoid而不是softmax
                probs_class0 = 1 - probs_class1
                probs = torch.stack([probs_class0, probs_class1], dim=1)  # [batch_size, 2]
            else:
                probs = F.softmax(logits, dim=1)

            conf, preds = probs.max(dim=1)

            all_predictions.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_confidences.extend(conf.detach().cpu().numpy())
            if logits.dim() == 1:
                all_scores.extend(probs_class1.detach().cpu().numpy())

    # 计算指标
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    all_scores = np.array(all_scores)

    roc_auc = None
    pr_auc = None

    accuracy = float(np.mean(all_predictions == all_labels))
    balanced_acc = float(balanced_accuracy_score(all_labels, all_predictions))
    f1 = float(f1_score(all_labels, all_predictions, average='weighted'))
    kappa = float(cohen_kappa_score(all_labels, all_predictions))
    if len(all_scores) > 0:
        roc_auc = roc_auc_score(all_labels, all_scores)
        precision, recall, thresholds = precision_recall_curve(all_labels, all_scores, pos_label=1)
        pr_auc = auc(recall, precision)
    cm = confusion_matrix(all_labels, all_predictions).tolist()

    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1,
        'cohen_kappa': kappa,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'mean_confidence': float(all_confidences.mean()),
        'confusion_matrix': cm,
    }
    return metrics



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        
    print("Program execution completed!")