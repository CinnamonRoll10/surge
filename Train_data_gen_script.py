# -*- coding: utf-8 -*-
"""

@author: Chethan
This script generates the train data pickle file, It will read the train data 
stored in the dataset folder and the create its pickle file.
"""
import scipy.io
import os
import mat73
import random
import numpy as np
import h5py
from sklearn.decomposition import IncrementalPCA
import pickle 
os.getcwd()

# log_path = os.path.join(args.logdir, args.log_prefix, "diffusion", time_now)
log_dir = os.getcwd()

# # COST2100 Data
# log_path = os.path.join(log_dir,"Dataset", "Benchmark_Data_COST2100", "Train")
# Train_dict            = scipy.io.loadmat(os.path.join(log_path,"DATA_Htrainin.mat"))

# Custom Data
log_path = os.path.join(log_dir,"Dataset", "TSF Vision model", "Data_custom(80in_20out)")
# log_path = os.path.join(
#     "C:/Users/rb/Desktop/DM_compression/GDMOPT_sp/Dataset/TSF Vision model/Data_custom(80in_20out)"
# )
Train_dict            = scipy.io.loadmat(os.path.join(log_path,"train.mat"))
  

Train_data = Train_dict['HT']
No_of_Dtpts = Train_data.shape[0]
Train_data = np.reshape(Train_data, (No_of_Dtpts, 2, 32, 32))

# Parameters
n_samples  = No_of_Dtpts
batch_size = 2000     # tune to your RAM
orig_shape = (2, 32, 32)
real_dim   = 2 * 32 * 32  # =2048
target_dim = 128


# X = np.load("H_split_real_imag.npy", mmap_mode="r")   # optional memmap
X = Train_data

def batch_generator(X, batch_size):
    N = X.shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        Xb = X[start:end]               # (b,2,32,32)
        Xb = Xb.reshape(Xb.shape[0], -1)  # (b,2048)
        yield Xb

# Set up IncrementalPCA
ipca = IncrementalPCA(n_components=target_dim)


print("Fitting the batch values...")

# 1) Fit in batches
for Xb in batch_generator(X, batch_size):
    ipca.partial_fit(Xb)

print("transforming in batches...")

# 2) Transform in batches to get compressed features
Y = np.zeros((n_samples, target_dim), dtype=np.float32)
idx = 0
for Xb in batch_generator(X, batch_size):
    nb = Xb.shape[0]
    Y[idx:idx+nb] = ipca.transform(Xb)
    idx += nb
    
# print("Saving compressed vectors...")

# Y now has shape (100000, 512), real‑valued
# np.save("H_compressed_COST2100_train_512.npy", Y)
# np.save("H_compressed_Quadriga_train_128.npy", Y)


# Create an output directory for the pickled vectors
codeword_dir = "pickle_vectors"
# SAVE_DIR = os.path.join(log_dir,"Dataset", "Benchmark_Data_COST2100", "Train",codeword_dir)
SAVE_DIR = os.path.join(log_dir,"Dataset", "TSF Vision model", "Data_custom(80in_20out)",codeword_dir)
os.makedirs(SAVE_DIR, exist_ok=True)

print("Saving compressed vectors as pickle files...")

# Save each 512-length vector into its own pickle file
for idx, vec in enumerate(Y):
    filename = os.path.join(SAVE_DIR, f"vector_{idx:06d}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(vec, f)

print(f"Saved {Y.shape[0]} pickle files to '{codeword_dir}/'.")

# Create an output directory for the HDF5 file
codeword_dir = "codewords_hdf5"
SAVE_DIR = os.path.join(log_dir, "Dataset", "TSF Vision model", "Data_custom(80in_20out)", codeword_dir)
os.makedirs(SAVE_DIR, exist_ok=True)

# Save as HDF5 file
h5file_path = os.path.join(SAVE_DIR, 'codewords.h5')
print(f"Saving compressed codewords to HDF5 file: {h5file_path}")

with h5py.File(h5file_path, 'w') as f:
    # Save the dataset as a compressed HDF5 dataset
    f.create_dataset('codewords', data=Y, compression='gzip', compression_opts=9)

print(f"Saved {Y.shape[0]} codewords to '{h5file_path}'.")

