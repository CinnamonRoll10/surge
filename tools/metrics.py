import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NMSE Calculation
def compute_nmse(Pred_matrix, GndTrth_matrix):
    """ 
    Input:  (B, C, H, W)
    Output: NMSE
    Args:
        Pred_matrix: (B, C, H, W)
        GndTrth_matrix:   (B, C, H, W)
    Returns:
        x (float): NMSE
    """
    # Convert numpy arrays to PyTorch tensors
    if isinstance(GndTrth_matrix, np.ndarray):
        sparse_gt = torch.tensor(GndTrth_matrix, dtype=torch.float32)
        sparse_gt = sparse_gt.to(device)
    if isinstance(Pred_matrix, np.ndarray):
        sparse_pred = torch.tensor(Pred_matrix, dtype=torch.float32)
        sparse_pred = sparse_pred.to(device)
        
    # moving to gpu    
    sparse_gt = GndTrth_matrix.to(device)
    sparse_pred = Pred_matrix.to(device)
    
    with torch.no_grad():
        # De-centralize
        sparse_gt = sparse_gt - 0.5
        sparse_pred = sparse_pred - 0.5
        # Calculate the NMSE
        power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
        difference = sparse_gt - sparse_pred
        mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
        nmse = 10 * torch.log10((mse.sum(dim=(1, 2)) / power_gt.sum(dim=(1, 2))).mean())
        
    return nmse
        
def compute_cosine_similarity(pred, target):
    """ 
    Compute the Cosine Similarity (CS) per sample, then average over the batch.
    
    Args:
        pred (torch.Tensor): The predicted tensor (model output), shape (B, C, H, W)
        target (torch.Tensor): The ground truth tensor, shape (B, C, H, W)
    
    Returns:
        float: The Cosine Similarity value.
    """
    # Ensure the tensors are on the same device
    pred = pred.to(device)
    target = target.to(device)

    with torch.no_grad():
        # Flatten the tensors to vectors (B, C*H*W)
        pred_flat = pred.flatten(start_dim=1)  # Flatten from the second dimension
        target_flat = target.flatten(start_dim=1)

        # Calculate the Frobenius dot product of predicted and target tensors
        dot_product = torch.sum(pred_flat * target_flat, dim=1)

        # Compute the Frobenius norms of the predicted and target tensors
        norm_pred = torch.norm(pred_flat, p=2, dim=1)
        norm_target = torch.norm(target_flat, p=2, dim=1)

        # Compute the cosine similarity for each sample in the batch
        cos_sim_per_sample = dot_product / (norm_pred * norm_target + 1e-8)  # Avoid division by zero

    # Average cosine similarity over the batch
    avg_cos_sim = cos_sim_per_sample.mean()

    return avg_cos_sim.item()  # Return the average Cosine Similarity