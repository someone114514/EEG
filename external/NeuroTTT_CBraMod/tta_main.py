from tent import tent
import argparse
import torch.optim as optim
import logging
import torch
import numpy as np
import random
from CBraMod.datasets import shu_dataset, chb_dataset, mumtaz_dataset, stress_dataset, tuab_dataset
from CBraMod.models import model_for_shu, model_for_chb, model_for_mumtaz, model_for_stress, model_for_tuab
from CBraMod.finetune_evaluator import Evaluator
import CBraMod.utils as utils
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'CBraMod'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tent'))


logger = logging.getLogger(__name__)

# def binary_entropy_from_logits(logits, eps=1e-8):
#     """Calculate the entropy loss for binary classification"""
#     p = torch.sigmoid(logits)
#     ent = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
#     return ent   # shape [B]

# class BinaryTent(tent.Tent):
#     """Tent for binary classification: use binary entropy for adaptation"""
#     def forward(self, x):
#         # 1. Forward
#         logits = self.model(x)

#         # 2. Calculate unsupervised loss
#         loss = binary_entropy_from_logits(logits).mean()

#         # 3. One-step gradient update
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         # 4. If episodic=True, restore parameters after each batch
#         if self.episodic:
#             self.reset()

#         # 5. Return logits for external evaluation
#         return logits

# def setup_tent_binary(model, args):
#     """Set up the Tent model for binary classification"""
#     model = tent.configure_model(model)
#     params, param_names = tent.collect_params(model)
#     optimizer = setup_optimizer(params, args)
#     tent_model = BinaryTent(model, optimizer, steps=args.steps, episodic=args.episodic)
#     tent.check_model(tent_model)
#     logger.info(f"model for adaptation: %s", model)
#     logger.info(f"params for adaptation: %s", param_names)
#     logger.info(f"optimizer for adaptation: %s", optimizer)
#     return tent_model

def setup_tent(model, args):
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params, args)
    tent_model = tent.Tent(model, optimizer, steps=args.steps, episodic=args.episodic)
    tent.check_model(tent_model)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model

def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model

def setup_optimizer(params, args):
    # TODO: Multi-lr is not supported now
    if args.optimizer == 'AdamW':
        return optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        return optim.SGD(params, momentum=0.9, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        return optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    else: 
        raise ValueError(f"Optimizer {args.optimizer} not supported")

def run_binary_classification(args, load_dataset_class, model_class):
    """Run tests on binary classification datasets"""
    load_dataset = load_dataset_class(args)
    data_loader = load_dataset.get_data_loader()
    model = model_class(args)
    
    if args.test_time_method == 'tent':
        model = setup_tent(model, args)
    elif args.test_time_method == 'source':
        model = setup_source(model)
    else:
        raise ValueError(f"Test time method {args.test_time_method} not supported")
    
    model = model.cuda()

    # Test evaluation
    test_eval = Evaluator(args, data_loader['test'])
    acc, pr_auc, roc_auc, cm = test_eval.get_metrics_for_binaryclass(model)

    print(f"Dataset: {args.downstream_dataset}")
    print(f"Method: {args.test_time_method} result in test set:")
    print(f"  acc: {acc:.5f}, pr_auc: {pr_auc:.5f}, roc_auc: {roc_auc:.5f}")
    print(f"  confusion matrix: {cm}")
    
    return acc, pr_auc, roc_auc, cm

def main():
    print("test_time_main.py")
    parser = argparse.ArgumentParser(description='Tent Test Time Adaptation for Binary Classification')
    parser.add_argument('--test_time_method', type=str, default='source', help='test time method (tent, source)')

    # test time settings
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer (AdamW, SGD, Adam)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    parser.add_argument('--steps', type=int, default=1, help='steps (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size (default: 32)')
    parser.add_argument('--episodic', action='store_true', help='episodic adaptation')

    # model settings
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        help='[all_patch_reps, all_patch_reps_twolayer, '
                             'all_patch_reps_onelayer, avgpooling_patch_reps]')

    # system settings
    parser.add_argument('--seed', type=int, default=3407, help='random seed (default: 3407)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 0)')
    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    
    # downstream dataset settings
    parser.add_argument('--downstream_dataset', type=str, required=True,
                        help='[FACED, SEED-V, PhysioNet-MI, SHU-MI, ISRUC, CHB-MIT, BCIC2020-3, Mumtaz2016, '
                             'SEED-VIG, MentalArithmetic, TUEV, TUAB, BCIC-IV-2a]')
    parser.add_argument('--datasets_dir', type=str, required=True, help='datasets_dir')
    parser.add_argument('--foundation_dir', type=str, required=True, help='foundation_dir')

    # The pretrained weights is disabled, instead, foundation_dir refers to the finetuned weights
    parser.add_argument('--use_pretrained_weights', type=bool, default=False, help='use_pretrained_weights')
    parser.add_argument('--use_finetuned_weights', type=bool, default=True, help='use_finetuned_weights')
    args = parser.parse_args()
    print(args)

    setup_seed(args.seed)
    torch.cuda.set_device(args.cuda)
    print(f'The downstream dataset is {args.downstream_dataset}')

    # set up dataset and model
    dataset_configs = {
        'SHU-MI': (shu_dataset.LoadDataset, model_for_shu.Model),
        'CHB-MIT': (chb_dataset.LoadDataset, model_for_chb.Model),
        'Mumtaz2016': (mumtaz_dataset.LoadDataset, model_for_mumtaz.Model),
        'MentalArithmetic': (stress_dataset.LoadDataset, model_for_stress.Model),
        'TUAB': (tuab_dataset.LoadDataset, model_for_tuab.Model),
    }

    if args.downstream_dataset not in dataset_configs:
        raise ValueError(f"Dataset {args.downstream_dataset} not supported. "
                        f"Supported datasets: {list(dataset_configs.keys())}")

    load_dataset_class, model_class = dataset_configs[args.downstream_dataset]

    # Test
    acc, pr_auc, roc_auc, cm = run_binary_classification(args, load_dataset_class, model_class)
    
    print(f"\n=== Final Results ===")
    print(f"Dataset: {args.downstream_dataset}")
    print(f"Method: {args.test_time_method}")
    print(f"Accuracy: {acc:.5f}")
    print(f"PR-AUC: {pr_auc:.5f}")
    print(f"ROC-AUC: {roc_auc:.5f}")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    main()