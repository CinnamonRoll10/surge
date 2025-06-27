import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dataset.dataset import HCodeDataset  # Assuming the custom dataset is in dataset.py
from metrics import compute_nmse, compute_cosine_similarity  # Assuming these functions are defined

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, test_config, model_config, diffusion_config, test_loader, writer):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions (denoised H matrices) and compute NMSE and CS.
    """
    # Loop over test dataset
    for idx, (H_i, codeword) in enumerate(tqdm(test_loader, desc="Sampling", total=len(test_loader))):
        H_i, codeword = H_i.to(device), codeword.to(device)

        # Initialize random noise xt
        xt = torch.randn_like(H_i).to(device)  # Initialize xt with noise same shape as H matrix

        # Ground truth H matrix for NMSE and CS computation
        ground_truth = H_i

        # Save the original ground truth for comparison
        if not os.path.exists(os.path.join(test_config['task_name'], 'samples')):
            os.makedirs(os.path.join(test_config['task_name'], 'samples'))

        np.save(os.path.join(test_config['task_name'], 'samples', f'ground_truth_{idx}.npy'), ground_truth.cpu().numpy())

        for t in tqdm(reversed(range(diffusion_config['num_timesteps'])), desc=f"Sampling {idx}/{len(test_loader)}"):
            # Get prediction of noise at time t
            noise_pred = model(xt, torch.as_tensor(t).unsqueeze(0).to(device), codeword)
            
            # Use scheduler to get xt-1 and x0 (denoised H matrix)
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(t).to(device))
            
            # Save denoised H matrix (x0_pred)
            ims = torch.clamp(xt, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2  # Scale to [0, 1] for saving/displaying
            
            # Save at each timestep (optional, can be skipped for performance)
            np.save(os.path.join(test_config['task_name'], 'samples', f'x0_{idx}_t{t}.npy'), ims.numpy())

            # Display/save the comparison images (optional)
            if t == 0:  # Save final x0 (denoised)
                np.save(os.path.join(test_config['task_name'], 'samples', f'x0_final_{idx}.npy'), ims.numpy())
        
        # Compute NMSE and CS between final denoised H matrix and ground truth
        final_denoised = torch.clamp(xt, -1., 1.).detach().cpu().numpy()
        nmse = compute_nmse(final_denoised, ground_truth.cpu().numpy())
        cs = compute_cosine_similarity(final_denoised, ground_truth.cpu().numpy())

        # Log and print NMSE and CS
        print(f"Sample {idx} - NMSE: {nmse:.4f}, CS: {cs:.4f}")

        # # Log the final NMSE and CS to TensorBoard       
        # writer.add_scalar('Test/NMSE', nmse, idx)
        # writer.add_scalar('Test/CS', cs, idx)


def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    test_config = config['test_params']
    
    # Load model with checkpoint
    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(os.path.join(test_config['task_name'],
                                                  test_config['ckpt_name']), map_location=device))
    model.eval()
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    # Load the test dataset
    test_dataset = HCodeDataset(split='test', H_path='data/test/H_matrices.mat', codeword_dir='data/test/pickle_vectors')
    test_loader = DataLoader(test_dataset, batch_size=test_config['batch_size'], shuffle=False)
    
    # TensorBoard Logging
    writer = SummaryWriter(log_dir=config['test_params']['task_name'])

    with torch.no_grad():
        sample(model, scheduler, test_config, model_config, diffusion_config, test_loader, writer)
        
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm CSI generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    infer(args)
