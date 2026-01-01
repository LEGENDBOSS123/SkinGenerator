import torch
from torch.utils.data import Dataset

class SkinDataset(Dataset):
    def __init__(self, path='skins_dataset.pt'):
        try:
            self.data = torch.load(path)
        except Exception as e:
            print(f"Failed to load dataset from {path}. Run preprocess.py first.")
            raise e

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Data is [Seq_Len, 10]
        # 0: ID, 1-9: Continuous Features
        return self.data[idx]