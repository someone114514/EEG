from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import re
import lmdb
import pickle
from datasets.config import SplitConfig
import yaml

def extract_mode_from_key(key):
    # example: test-Data_Sample01-13
    mode = key.split('-')[0]
    return mode

def extract_subj_from_key(key):
    # example: test-Data_Sample01-13
    subj = key.split('-')[1][-2:]
    return int(subj)

def extract_trial_from_key(key):    
    # example: test-Data_Sample01-13
    trial = key.split('-')[2]
    return int(trial)

def to_global_trial_id(trial_id, mode):
    if mode == 'train':
        return trial_id
    elif mode == 'val':
        return trial_id + 300
    elif mode == 'test':
        return trial_id + 350

class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
    ):
        super(CustomDataset, self).__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']
        # print(key, data.shape, label)
        return data/100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label).long()

class CustomDatasetWithSplit(Dataset):
    def __init__(self, data_dir, keys):
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        self.keys = keys
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']
        # print(key, data.shape, label)
        return data/100, label
    
    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label).long()

# 把有标签数据集包装成“无标签视图”
class UnlabeledWrapper(Dataset):
    def __init__(self, base_ds):
        self.base = base_ds
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        x, _ = self.base[idx]   # 丢弃 label
        return x

    # 只拼接 x 的 collate（用于无标签训练）
    def collate_unlabeled(self, batch):
        # batch: List[x]，其中 x 是 numpy 数组或张量
        x_data = np.array(batch)
        return to_tensor(x_data)    # 返回仅包含 x 的张量

class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    # def get_data_loaders_by_subj(self, config = None):
    #     if config is None:
    #         print("Using default config for subject split")
    #         config = {
    #             "train": range(1, 12), 
    #             "val": range(12, 14), 
    #             "test": range(14, 16)
    #         }

    #     db = lmdb.open(self.datasets_dir, readonly=True, lock=False, readahead=True, meminit=False)
    #     with db.begin(write=False) as txn:
    #         all_keys = pickle.loads(txn.get('__keys__'.encode()))
    #     all_keys = all_keys['train'] + all_keys['val'] + all_keys['test']

    #     for key in all_keys:
    #         mode = extract_mode_from_key(key)
    #         subj = extract_subj_from_key(key)
        
    #     new_train_keys = []
    #     new_val_keys = []
    #     new_test_keys = []
    #     for key in all_keys:
    #         subj = extract_subj_from_key(key)
    #         if subj in config['train']:
    #             new_train_keys.append(key)
    #         elif subj in config['val']:
    #             new_val_keys.append(key)
    #         elif subj in config['test']:
    #             new_test_keys.append(key)

    #     new_train_set = CustomDatasetWithSplit(self.datasets_dir, new_train_keys)
    #     new_val_set = CustomDatasetWithSplit(self.datasets_dir, new_val_keys)
    #     new_test_set = CustomDatasetWithSplit(self.datasets_dir, new_test_keys)

    #     print(len(new_train_set), len(new_val_set), len(new_test_set))

    #     data_loader = {
    #         'train': DataLoader(
    #             new_train_set,
    #             batch_size=self.params.batch_size,
    #             collate_fn=new_train_set.collate,
    #             shuffle=True,
    #         ),
    #         'val': DataLoader(
    #             new_val_set,
    #             batch_size=self.params.batch_size,
    #             collate_fn=new_val_set.collate,
    #             shuffle=False,
    #         ),
    #         'test': DataLoader(
    #             new_test_set,
    #             batch_size=self.params.batch_size,
    #             collate_fn=new_test_set.collate,
    #             shuffle=False,
    #         ),
    #     }
    #     return data_loader
    
    def get_data_loader(self, config: SplitConfig):
        print("Using new data loader by config")
        db = lmdb.open(self.datasets_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with db.begin(write=False) as txn:
            all_keys = pickle.loads(txn.get('__keys__'.encode()))
        all_keys = all_keys['train'] + all_keys['val'] + all_keys['test']
        
        test_keys = []
        val_keys = []
        train_keys = []

        for key in all_keys:
            mode = extract_mode_from_key(key)
            subj = extract_subj_from_key(key)
            trial = to_global_trial_id(extract_trial_from_key(key), mode)
            sess = None

            if config.is_test(sess, subj, trial):
                test_keys.append(key)
            elif config.is_val(sess, subj, trial):
                val_keys.append(key)
            else:
                train_keys.append(key)
        
        test_set = CustomDatasetWithSplit(self.datasets_dir, test_keys)
        val_set = CustomDatasetWithSplit(self.datasets_dir, val_keys)
        train_set = CustomDatasetWithSplit(self.datasets_dir, train_keys)

        print(len(train_set), len(val_set), len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        return data_loader
        
    def get_data_loader_legacy(self):
        print("Using legacy data loader")
        train_set = CustomDataset(self.datasets_dir, mode='train')
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
        print(len(train_set), len(val_set), len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        return data_loader


    def get_target_loaders(self, split_by : str = 'default'):
        # TODO: support splitting via subj ids
        # 目标域无标签训练：基于 train split，但抹掉标签
        if split_by != 'default':
            raise NotImplementedError(f"Splitting by {split_by} is not supported for speech dataset")

        target_train_set = UnlabeledWrapper(CustomDataset(self.datasets_dir, mode='train'))
        self.target_train_loader = DataLoader(
            target_train_set,
            batch_size=self.params.batch_size,
            collate_fn=target_train_set.collate_unlabeled,  # 只拼 x
            shuffle=True,
        )

        # 目标域有标签测试：直接用 test split，保持原 collate（含 x 和 y）
        target_test_set = CustomDataset(self.datasets_dir, mode='test')
        self.target_test_loader = DataLoader(
            target_test_set,
            batch_size=self.params.batch_size,
            collate_fn=target_test_set.collate,
            shuffle=False,
        )

        return self.target_train_loader, self.target_test_loader

if __name__ == "__main__":
    datasets_dir = "/work/projects/project02629/datasets/processed/BCIC2020-3/processed"
    config_file = "/work/projects/project02629/yangshen/B2T-TTT/configs/speech_dataset_test_subj_val_trial.yaml"
    # config_file = "/work/projects/project02629/yangshen/B2T-TTT/configs/speech_dataset_default.yaml"

    params = {
        "datasets_dir": datasets_dir,
        "batch_size": 128,
    }
    from types import SimpleNamespace
    params = SimpleNamespace(**params)

    config = yaml.safe_load(open(config_file, "r"))
    config = SplitConfig(**config)
    config.print_config()

    load_dataset = LoadDataset(params)
    data_loader = load_dataset.get_data_loader(config)

    print(len(data_loader['train']), len(data_loader['val']), len(data_loader['test']))

    for data, label in data_loader['val']:
        pass