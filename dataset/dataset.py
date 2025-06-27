# -*- coding: utf-8 -*-
"""
Created on

@author:
"""
import os
import numpy as np
import scipy.io
import random
import torch

from torch.utils.data.dataset import Dataset


class HCodeDataset(Dataset):
    def __init__(self, split, H_path, codeword_dir=None, subset_size=None, use_percentage=False, encoder=None, device='cpu'):
        self.split = split
        self.H_path = H_path
        self.subset_size = subset_size
        self.use_percentage = use_percentage
        self.encoder = encoder.to(device) if encoder else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.H_flat = self.load_data(H_path)

        if subset_size:
            self.select_subset()

    def load_data(self, H_path):
        assert os.path.exists(H_path), f"H matrix file does not exist at {H_path}"
        mat_data = scipy.io.loadmat(H_path)
        if 'HT' in mat_data:
            H = mat_data['HT']
        else:
            raise KeyError("Expected 'HT' in the .mat file, but not found.")
        H_flat = H.reshape(H.shape[0], -1)  # (N, 2048)
        print(f"Loaded {H_flat.shape[0]} CSI samples.")
        return H_flat

    def select_subset(self):
        if self.use_percentage:
            subset_size = int(len(self.H_flat) * self.subset_size)
        else:
            subset_size = self.subset_size
        indices = random.sample(range(len(self.H_flat)), subset_size)
        self.H_flat = self.H_flat[indices]

    def __len__(self):
        return len(self.H_flat)

    def __getitem__(self, idx):
        H_i = self.H_flat[idx].reshape(2, 32, 32)
        H_tensor = torch.tensor(H_i, dtype=torch.float32).to(self.device)
        if self.encoder:
            with torch.no_grad():
                codeword = self.encoder(H_tensor.unsqueeze(0)).squeeze(0).cpu()
        else:
            codeword = torch.zeros(128)  # Fallback
        return H_i, codeword


# ─── Convenience Sampling ─────────────────────────────────────────────────────

def sample_random(dataset: HCodeDataset, k: int = 1, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    idxs = np.random.randint(0, len(dataset), size=k)
    return [dataset[i] for i in idxs]

def get_train_dataset():
    H_path = "C:/Users/rb/Desktop/DM_compression/GDMOPT_sp/Dataset/TSF Vision model/Data_custom(80in_20out)/train.mat"
    return HCodeDataset(split='train', H_path=H_path)

def get_val_dataset():
    H_path = "C:/Users/rb/Desktop/DM_compression/GDMOPT_sp/Dataset/TSF Vision model/Data_custom(80in_20out)/test.mat"
    return HCodeDataset(split='test', H_path=H_path)

def sample_train(k: int = 1, seed: int = None):
    return sample_random(get_train_dataset(), k=k, seed=seed)

def sample_val(k: int = 1, seed: int = None):
    return sample_random(get_val_dataset(), k=k, seed=seed)
