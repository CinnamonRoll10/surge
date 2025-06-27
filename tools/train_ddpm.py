import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm
import csv
import time, gc
from fvcore.nn import parameter_count, FlopCountAnalysis
from prettytable import PrettyTable
from ptflops import get_model_complexity_info

# Add the parent directory to sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now try importing from models
from models.unet_base import CondUNet2D
from dataset.dataset import HCodeDataset  # Assuming the custom dataset is in dataset.py
from .metrics import compute_nmse, compute_cosine_similarity  # Assume these functions are defined
from models.encoder import CSIEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device1 = "cuda" if torch.cuda.is_available() else "cpu"

# Timing utilities
start_time = None

def start_timer():
    global start_time
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    if torch.cuda.is_available():
        print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))

# Load configuration from default.yaml
def load_config(config_path='default.yaml'):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config



def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
class CondUNet2DWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x, t, codeword):
        # Directly pass codeword as raw_input and encoder
        return self.model(x, t, raw_input=codeword, encoder=self.model.encoder)

def flopcount_ptflops(model, spatial_dim=32):
    print("🧮 ptflops: FLOP and Parameter Count Report")

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            B = x.shape[0]
            t = torch.randint(0, 1000, (B,), device=x.device)
            code = torch.randn(B, 2, spatial_dim, spatial_dim, device=x.device)
            return self.model(x, t, raw_input=code, encoder=self.model.encoder)

    wrapped = Wrapper(model).to('cuda')
    wrapped.eval()

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            wrapped,
            (2, spatial_dim, spatial_dim),  # input shape (channels, H, W)
            as_strings=True,
            print_per_layer_stat=True,
            verbose=True
        )
        print(f"✅ MACs: {macs}")
        print(f"✅ Params: {params}")

# Training function
def train_ddpm(model, train_dataset, test_dataset, config):
    # Set device
    model.to(device)
    print("[CHECK] Type of train_dataset:", type(train_dataset))
    print("[CHECK] Sample item (train_dataset[0]):", train_dataset[0])

    # DataLoader for training and validation (test) dataset
    sample_loader = DataLoader(train_dataset, batch_size=1)
    H_i_sample, _ = next(iter(sample_loader))

    print("Sample shape:", H_i_sample.shape)

    train_loader = DataLoader(train_dataset, batch_size=config['train_params']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['train_params']['batch_size_test'], shuffle=False)
    # Fetch one real batch
    #real_batch = next(iter(train_loader))
   #H_i_real, _ = real_batch  # You can use H_i_real as both input and encoder input
   # H_i_real = H_i_real.to(device)

    # Sample timestep
    #t_real = torch.randint(0, config['diffusion_params']['num_timesteps'], (H_i_real.shape[0],)).to(device)

    # Optimizer and learning rate scheduler
    
    optimizer = optim.AdamW(model.parameters(), lr=config['train_params']['lr'])
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # Loss function (MSE loss)
    criterion = torch.nn.MSELoss()
    # Extract num_timesteps once from config
    num_timesteps = config['diffusion_params']['num_timesteps']


    # TensorBoard Logging
    writer = SummaryWriter(log_dir=config['train_params']['task_name'])

    # --- Parameter and FLOP count ---
    # Calculate the number of parameters and FLOPs in the model
    # Calculate the total number of parameters in the model
    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    count_parameters(model)
   ## wrapped_model = CondUNet2DWrapper(model)

    #dummy_x = torch.randn(1, 2, 32, 32).to(device)
    #dummy_t = torch.tensor([config['diffusion_params']['num_timesteps'] // 2]).to(device)
    #dummy_codeword = torch.randn(1, 2, 32, 32).to(device)  # This mimics raw_input for encoder

    ##try:
    ##   flops = FlopCountAnalysis(wrapped_model, (H_i_real, t_real, H_i_real))  # raw_input = H_i_real
    ##    total_flops = flops.total()
    ##    print(f"Actual FLOPs for one real batch: {total_flops:.2e}")
    ##    print(f"Per-sample FLOPs: {total_flops / H_i_real.shape[0]:.2e}")
    ##    writer.add_scalar('Model/FLOPs', total_flops, 0)
    ##except Exception as e:
    ##    print(f"⚠️ FLOP analysis failed on real batch: {e}")


    # Print the total number of parameters in millions
    # print(f"Model Parameter Count: {num_params / 1e6:.2f} million")
# =============================================================================
#     flops = FlopCountAnalysis(model, (torch.randn(1, 2, 32, 32).to(device), torch.tensor(config['diffusion_params']['num_timesteps'],dtype=torch.int16).to(device), torch.randn(config['model_params']['code_dim'], 1).to(device)))  # Dummy input to calculate FLOPs
# 
#     
#     print(f"Model FLOPs: {flops.total()}")
# =============================================================================

# =============================================================================
#     writer.add_scalar('Model/Parameter Count', num_params / 1e6, 0)  # Log parameters in millions
#     writer.add_scalar('Model/FLOPs', flops.total(), 0)  # Log FLOPs 
# =============================================================================

    # Prepare CSV file for storing losses and metrics
    csv_filename = os.path.join(config['train_params']['task_name'], 'metrics.csv')
    with open(csv_filename, mode='w', newline='') as file:
        writer_csv = csv.writer(file)
        writer_csv.writerow(['epoch', 'train_loss', 'train_nmse', 'train_cs', 'test_loss', 'test_nmse', 'test_cs'])
        
    
    
    use_amp = True
    """ define loss scaler for automatic mixed precision """
    # scaler = torch.cuda.amp.GradScaler()
    scaler = torch.amp.GradScaler("cuda" ,enabled=use_amp)
    # Perform FLOP count before training begins
    flopcount_ptflops(model, spatial_dim=32)


    
    start_timer()

    # Training Loop
    total_start_time = time.time()  # Start total training time measurement
    for epoch in range(config['train_params']['num_epochs']):
        epoch_start_time = time.time()  # Start current epoch time measurement

        model.train()
        train_loss = 0.0
        train_nmse = 0.0
        train_cs = 0.0
        
        # Training epoch
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train_params']['num_epochs']}", total=len(train_loader)) as pbar:
            print(device)
            for batch_idx, batch in enumerate(pbar):
                H_i, codeword = batch
                #print(f"[DEBUG] batch type: {type(batch)}")
                #print(f"[DEBUG] batch content: {batch}")

                #  Add this only for first batch
                #if batch_idx == 0:
                    #print(f"[DEBUG] H_i shape: {H_i.shape}")         # Should be (B, 2, 32, 32)
                    #print(f"[DEBUG] codeword shape: {codeword.shape}") # Should be (B, 128) or similar
                
                
                num_timesteps = config['diffusion_params']['num_timesteps']  
                
                batch_size = H_i.shape[0]
                # Assuming H_i is the input tensor and codeword is the PCA codeword
                H_i = H_i.to(torch.float32)  # Convert to float32 before passing to the model
                codeword = codeword.to(torch.float32)  # Ensure codeword is also float32

                # Sample a random timestep for each batch
                timestep = torch.randint(0, num_timesteps, (batch_size,)).to(device)
                timestep = timestep.to(device)
                
                
                # Forward pass
                optimizer.zero_grad()
                H_i = H_i.to(device)  # move input to GPU
                timestep = timestep.to(device)

                
                with torch.autocast(device_type=device1, dtype=torch.float16, enabled=use_amp):
                    H_i = H_i.to(dtype=torch.float16)

                # Mixed-precision training
                    output = model(H_i, timestep, raw_input=H_i, encoder=model.encoder)

                # Compute loss
                    loss = criterion(output, H_i)
                    
                # Scales the loss, and calls backward() 
                # to create scaled gradients 
                scaler.scale(loss).backward()
                # Unscales gradients and calls 
                # or skips optimizer.step() 
                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()
                
                # # Normal training without mixed-precision
                # output = model(H_i, timestep, codeword)

                # # Compute loss
                # loss = criterion(output, H_i)
                # loss.backward()
                # optimizer.step()
                
                
                optimizer.zero_grad() # set_to_none=True here can modestly improve performance
                
                # Compute NMSE and Cosine Similarity for training data
                nmse = compute_nmse(output, H_i)
                cs = compute_cosine_similarity(output, H_i)
                
                train_loss += loss.item()
                train_nmse += nmse
                train_cs += cs

                # Update progress bar with loss information
                pbar.set_postfix(loss=train_loss / (batch_idx + 1), nmse=train_nmse / (batch_idx + 1), cs=train_cs / (batch_idx + 1))

        # Average training metrics
        train_loss /= len(train_loader)
        train_nmse /= len(train_loader)
        train_cs /= len(train_loader)

        # Log training metrics to TensorBoard
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/NMSE', train_nmse, epoch)
        writer.add_scalar('Train/CS', train_cs, epoch)

        # Evaluate on the test dataset after every epoch
        model.eval()
        test_loss = 0.0
        test_nmse = 0.0
        test_cs = 0.0
        total_nmse = 0.0
        num_samples = 0
        best_nmse = float('inf')  # Initialize best NMSE to a large number
        checkpoint_path = os.path.join(config['train_params']['task_name'], f"checkpoint_{epoch+1}.pth")

        with torch.no_grad():
            with tqdm(test_loader, desc=f"Evaluating Epoch {epoch+1}", total=len(test_loader)) as pbar:
                for batch_idx, (H_i, _) in enumerate(pbar):  # discard codeword
                    H_i = H_i.to(device=device, dtype=torch.float16)

                    
                    batch_size = H_i.shape[0]
                    # Assuming H_i is the input tensor and codeword is the PCA codeword
                    H_i = H_i.to(torch.float32)  # Convert to float32 before passing to the model
                    codeword = codeword.to(torch.float32)  # Ensure codeword is also float32
                    
                    # timestep = config['diffusion_params']['num_timesteps']
                    timestep = torch.randint(0, num_timesteps, (batch_size,)).to(device)
                    timestep = timestep.to(device)
                    
                    with torch.autocast(device_type=str(device), dtype=torch.float16, enabled=use_amp):

                        # Forward pass for testing
                        output = model(H_i, timestep, raw_input=H_i, encoder=model.encoder)
                        # Compute loss
                        loss = criterion(output, H_i)
                    
                    # Compute NMSE and Cosine Similarity for test data
                    nmse = compute_nmse(output, H_i)
                    total_nmse += nmse * H_i.size(0)  # Multiply by batch size to sum NMSE across batches
                    num_samples += H_i.size(0)
                    cs = compute_cosine_similarity(output, H_i)
                    
                    test_loss += loss.item()
                    test_nmse += nmse
                    test_cs += cs
                    best_model_state = model.state_dict()

                    # Check if current model has the best NMSE
                    if nmse < best_nmse:
                        best_nmse = nmse
                        best_model_state = model.state_dict()  # Save the best model state

                    
                
                    # Save the best model
                    
                    torch.save(best_model_state, checkpoint_path)
                    # print("Best model saved as 'best_model.pth'")

                    # Update progress bar with loss information
                    pbar.set_postfix(loss=test_loss / (batch_idx + 1), nmse=test_nmse / (batch_idx + 1), cs=test_cs / (batch_idx + 1))


        # Calculate average NMSE across all batches
        avg_nmse = total_nmse / num_samples
        # print(f'Best Test NMSE: {best_nmse:.4f}')
        # print(f'Average Test NMSE: {avg_nmse:.4f}')
        # Average test metrics
        test_loss /= len(test_loader)
        test_nmse /= len(test_loader)
        test_cs /= len(test_loader)

        # Log test metrics to TensorBoard
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Test/NMSE', test_nmse, epoch)
        writer.add_scalar('Test/CS', test_cs, epoch)

        # Print Epoch Metrics
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - total_start_time
        est_remaining_time = (total_time / (epoch + 1)) * (config['train_params']['num_epochs'] - (epoch + 1))

        print(f"Epoch [{epoch+1}/{config['train_params']['num_epochs']}] "
              f"Train Loss: {train_loss:.4f}, Train NMSE: {train_nmse:.4f}, Train CS: {train_cs:.4f} "
              f"Test Loss: {test_loss:.4f}, Test NMSE: {test_nmse:.4f}, Test CS: {test_cs:.4f}"
              f'Best Test NMSE: {best_nmse:.4f}, Average Test NMSE: {avg_nmse:.4f}')
        print(f"Epoch Time: {epoch_time:.2f}s | Estimated Time Left: {est_remaining_time:.2f}s")

        # Step the learning rate scheduler
        lr_scheduler.step()

        # Optionally, save model checkpoints
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs (for example)
            checkpoint_path = os.path.join(config['train_params']['task_name'], f"checkpoint_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        # Save metrics to CSV
        with open(csv_filename, mode='a', newline='') as file:
            writer_csv = csv.writer(file)
            writer_csv.writerow([epoch+1, train_loss, train_nmse, train_cs, test_loss, test_nmse, test_cs])
            
            
        torch.cuda.empty_cache()   # Empty GPU cache to speed up 
        
        
    writer.close()


# Main function to run the training
if __name__ == '__main__':
    # Load configuration from default.yaml
    
    

    # Load the config file using an absolute path
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'default.yaml')
    # config_path = 'c:/users/rb/desktop/dm_compression/cddpm_csi_cmpr/config/default.yaml'

    
    config = load_config(config_path=config_path)


    # Initialize the model
    #model = CondUNet2D(**config['model_params'])  # Unpack the model parameters from the config dictionary
    encoder = CSIEncoder(seq=64, dim=32, heads=4, codeword=128).to(device)
    model = CondUNet2D(**config['model_params'], encoder=encoder).to(device)


    encoder_model = CSIEncoder().to(device)
    encoder_model.eval()

    # Load previously trained model
    load_prev_model = 1
    train_config = config['train_params']

# if load_prev_model:
#     ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
#     if os.path.exists(ckpt_path):
#         model.load_state_dict(torch.load(ckpt_path, map_location=device))
#         print(f"Loaded model from {ckpt_path}")
#     else:
#         print(f"⚠️ Warning: Checkpoint not found at {ckpt_path}, training from scratch.")
if load_prev_model:
    ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    
    if os.path.exists(ckpt_path):
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
            print(f"✅ Loaded model from {ckpt_path}")
        except RuntimeError as e:
            print(f"❌ Checkpoint mismatch error:\n{e}")
            print("⚠️ Model architecture has changed. Training from scratch.")
            # Optionally log or archive the bad checkpoint
    else:
        print(f"⚠️ Warning: Checkpoint not found at {ckpt_path}, training from scratch.")


        # checkpoint = torch.load(os.path.join(train_config['task_name'],
        #                                               train_config['ckpt_name']),
        #                         map_location = lambda storage, loc: storage.cuda(dev))
        
        # model.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # scaler.load_state_dict(checkpoint["scaler"])
    
    # # Shyam Quadriga data
    # H_train_path = 'C:/Users/rb/Desktop/DM_compression/GDMOPT_sp/Dataset/TSF Vision model/Data_custom(80in_20out)/train.mat'
    # train_cdw = 'C:/Users/rb/Desktop/DM_compression/GDMOPT_sp/Dataset/TSF Vision model/Data_custom(80in_20out)/codewords_hdf5'
    # H_test_path = 'C:/Users/rb/Desktop/DM_compression/GDMOPT_sp/Dataset/TSF Vision model/Data_custom(80in_20out)/test.mat'
    # test_cdw = 'C:/Users/rb/Desktop/DM_compression/GDMOPT_sp/Dataset/TSF Vision model/Data_custom(80in_20out)/codewords_hdf5_test'
    
    # COST2100 data
    H_train_path = '/scratch/work/prakritip23/diffusion_model/Benchmark_Data_COST2100/DATA_Htrainout.mat'
    train_cdw = '/scratch/work/prakritip23/diffusion_model/Benchmark_Data_COST2100/codewords_hdf5_out'
    H_test_path = '/scratch/work/prakritip23/diffusion_model/Benchmark_Data_COST2100/DATA_Htestout.mat'
    test_cdw = '/scratch/work/prakritip23/diffusion_model/Benchmark_Data_COST2100/codewords_hdf5_test_out'

    test_subset_size = config['test_params']['test_subset_size']

    # Get the dataset
    train_dataset = HCodeDataset(
        split='train',
        H_path=H_train_path,
        codeword_dir=train_cdw,
        encoder=encoder_model,
        device=device
    )

    test_dataset = HCodeDataset(
       split='test',
       H_path=H_test_path,
       codeword_dir=test_cdw,
       subset_size=test_subset_size,
       use_percentage=False,
       encoder=encoder_model,
       device=device
    )
    #test_dataset = HCodeDataset(
    #split='test',
    #H_path=H_test_path,
    #codeword_dir=test_cdw,
    #subset_size=test_subset_size,  # Pass the subset size here
    #use_percentage=False  # Set this to True if you want a percentage of the dataset
#)
    # Train the model
    train_ddpm(model, train_dataset, test_dataset, config)
