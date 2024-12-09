import torch
import pathlib
import hydra
from tqdm.auto import tqdm
import wandb
import torch.nn.functional as F
import pdb
from models.ColdDiffusion.scheduler import *
from torch.optim import Adam
from models.ColdDiffusion.utils.model import Unet1D
import torch.optim.lr_scheduler as lr_scheduler
import models.ColdDiffusion.utils.testing as testing

### Model Parameters 
def create_model_and_optimizer(cfg):
    '''
    Creates the Unet1D model and the Adam optimizer using the given arguments.
    
    Args:
        cfg: The arguments containing model and optimizer parameters.
        
    Returns:
        model (Unet1D): The instantiated model.
        optimizer (Adam): The instantiated optimizer.
    '''
    model = Unet1D(
        dim = cfg.model.dim,
        dim_mults = cfg.model.dim_multiples,
        channels = cfg.model.channels,
        learned_sinusoidal_cond = cfg.model.learned_sinusoidal_cond,
        attn_dim_head = cfg.model.attn_dim_head,
        attn_heads = cfg.model.attn_heads,
    )
    if cfg.model.continue_from_pretrained:
        model.load_state_dict(torch.load(cfg.model.pretrained_path, map_location=device, weights_only=True))

    optimizer = Adam(model.parameters(), lr= cfg.model.lr)
    return model, optimizer

def load_model_and_weights(path_model, cfg):
    '''
    Loads the Unet1D model and its weights from the specified path.
    
    Args:
        cfg: The arguments containing model and optimizer parameters.
        
    Returns:
        model (Unet1D): The model with loaded weights.
    '''
    model = Unet1D(
        dim = cfg.model.dim,
        dim_mults = cfg.model.dim_multiples,
        channels = cfg.model.channels,
        learned_sinusoidal_cond = cfg.model.learned_sinusoidal_cond,
        attn_dim_head = cfg.model.attn_dim_head,
        attn_heads = cfg.model.attn_heads,
    )
    model.load_state_dict(torch.load(path_model, map_location=device, weights_only=True))
    return model

### Loss
def p_losses(cfg, denoise_model, eq_in, noise_real, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device, loss_type="l1"):
    '''
    Computes the loss for the denoising model.
    
    Args:
        cfg: The arguments containing model and optimizer parameters.
        denoise_model (Unet1D): The denoising model.
        eq_in (torch.Tensor): The input tensor.
        noise_real (torch.Tensor): The real noise tensor.
        t (torch.Tensor): The time steps tensor.
        sqrt_alphas_cumprod (torch.Tensor): The cumulative product of alphas.
        sqrt_one_minus_alphas_cumprod (torch.Tensor): The square root of one minus the cumulative product of alphas.
        device (torch.device): The device to run the model on.
        loss_type (str): The type of loss to use ('l1', 'l2', 'huber').
        
    Returns:
        final_loss (torch.Tensor): The computed loss.
    '''
    x_start = eq_in
    x_end = eq_in + noise_real
    x_noisy = forward_diffusion_sample(x_start, x_end, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod).to(torch.float32)
    predicted_eq = denoise_model(x_noisy.to(torch.float32), t.to(torch.float32))
    x = x_noisy
    new_x_start = predicted_eq
    predicted_noise = ((x - get_index_from_list(sqrt_alphas_cumprod, t, x.shape) * predicted_eq) / get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape))
    new_x_end = predicted_noise + predicted_eq
    new_t = torch.randint(0, cfg.model.T, (x.shape[0],), device=device).long()
    new_x_noisy = forward_diffusion_sample(new_x_start, new_x_end, new_t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod).to(device)
    new_predicted_eq = denoise_model(new_x_noisy.to(torch.float32), new_t.to(torch.float32))
    if loss_type == 'l1':
        loss = F.l1_loss(eq_in, predicted_eq)
        loss2 = F.l1_loss(eq_in, new_predicted_eq)
        final_loss = (loss + cfg.model.penalization * loss2)
    elif loss_type == 'l2':
        loss = F.mse_loss(eq_in, predicted_eq)
        loss2 = F.mse_loss(eq_in, new_predicted_eq)
        final_loss = (loss + cfg.model.penalization * loss2)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(eq_in, predicted_eq)
        loss2 = F.smooth_l1_loss(eq_in, new_predicted_eq)
        final_loss = (loss + cfg.model.penalization * loss2)
    else:
        raise NotImplementedError()

    return final_loss

def train_one_epoch(model, optimizer, tr_dl, tr_dl_noise, cfg, device):
    '''
    Trains the model for one epoch.
    
    Args:
        model (Unet1D): The model to train.
        optimizer (Adam): The optimizer.
        tr_dl (DataLoader): The training data loader.
        tr_dl_noise (DataLoader): The noisy training data loader.
        cfg: The arguments containing training parameters.
        device (torch.device): The device to run the model on.
        
    Returns:
        curr_train_loss (float): The average training loss for the epoch.
    '''
    model.train()
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = compute_beta_schedule(cfg)
    sum_train_loss = 0
    for step, (eq_in, noise_in) in tqdm(enumerate(zip(tr_dl, tr_dl_noise)), total=len(tr_dl)):
        optimizer.zero_grad()
        # eq_in = eq_in[1][:, args.channel_type, :].unsqueeze(dim=1).to(device)
        # reduce_noise = random.randint(*args.Range_RNF) * 0.01
        # noise_real = (noise_in[1][:, args.channel_type, :].unsqueeze(dim=1) * reduce_noise).to(device)
        eq_in = eq_in.to(device)
        noise_real = noise_in.to(device)
        t = torch.randint(0, cfg.model.T, (eq_in.shape[0],), device=device).long()
        loss = p_losses(cfg, model.to(device), eq_in, noise_real, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
        sum_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        # with torch.no_grad():
        #     if cfg.user.wandb:
        #         wandb.log({"Train Loss Batch": loss.item()})
    curr_train_loss = sum_train_loss / len(tr_dl)
    return curr_train_loss

def validate_model(model, val_dl, val_dl_noise, cfg, device=device):
    '''
    Validates the model on the validation dataset.
    
    Args:
        model (Unet1D): The model to validate.
        val_dl (DataLoader): The validation data loader.
        val_dl_noise (DataLoader): The noisy validation data loader.
        cfg: The arguments containing validation parameters.
        device (torch.device): The device to run the model on.
        
    Returns:
        curr_val_loss (float): The average validation loss.
    '''
    model.eval()
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = compute_beta_schedule(cfg)
    sum_val_loss = 0
    with torch.no_grad():
        for step, (eq_in, noise_in) in tqdm(enumerate(zip(val_dl, val_dl_noise)), total=len(val_dl)):
            # eq_in = eq_in[1][:, cfg.channel_type, :].unsqueeze(dim=1).to(device)
            # reduce_noise = random.randint(*args.Range_RNF) * 0.01
            # noise_real = (noise_in[1][:, args.channel_type, :].unsqueeze(dim=1) * reduce_noise).to(device)
            eq_in = eq_in.to(device)
            noise_real = noise_in.to(device)
            t = torch.randint(0, cfg.model.T, (eq_in.shape[0],), device=device).long()
            loss = p_losses(cfg, model.to(device), eq_in, noise_real, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
            sum_val_loss += loss.item()
        curr_val_loss = sum_val_loss / len(val_dl)
    return curr_val_loss

def train_model(cfg, tr_dl, tr_dl_noise, val_dl, val_dl_noise):
    '''
    Trains the model for multiple epochs and validates it.
    
    Args:
        cfg: The arguments containing training parameters.
        tr_dl (DataLoader): The training data loader.
        tr_dl_noise (DataLoader): The noisy training data loader.
        val_dl (DataLoader): The validation data loader.
        val_dl_noise (DataLoader): The noisy validation data loader.
        
    Returns:
        min_loss (float): The minimum validation loss achieved during training.
    '''
    output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    print(f"Trial: T={cfg.model.T}, scheduler_type = {cfg.model.scheduler_type}, s={cfg.model.s}, Range_RNF=None")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, optimizer = create_model_and_optimizer(cfg)
    # print(model)
    model = model.to(device)
    min_loss = np.inf
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    for epoch in range(cfg.model.epochs):
        train_loss = train_one_epoch(model, optimizer, tr_dl, tr_dl_noise, cfg, device)
        print(f'Epoch: {epoch}, Train Loss: {train_loss}')
        if cfg.user.wandb:
            wandb.log({"Train Loss": train_loss}, step=epoch)  # Log train loss to wandb
        val_loss = validate_model(model, val_dl, val_dl_noise, cfg, device)
        print(f'Epoch: {epoch}, Val Loss: {val_loss}')
        if cfg.user.wandb:
            wandb.log({"Val Loss": val_loss}, step=epoch)  # Log validation loss to wandb
        scheduler.step()
        if val_loss < min_loss:
            min_loss = val_loss
            save_path = f'{output_dir}/chkpt_epoch_{epoch}_{cfg.model.T}_{cfg.model.scheduler_type}_cold_diffusion.pth' 
            torch.save(model.state_dict(), save_path)
            # if cfg.user.wandb:
            #     wandb.save(save_path)  # Log the model checkpoint to wandb
            print(f"Best Epoch (so far): {epoch+1}")
    if cfg.user.wandb:
        wandb.finish()  # End the wandb run after training is complete
    return min_loss

def test_model(cfg, test_loader, noise_test_loader):
    '''
    Tests the model on the test dataset and saves the results.
    
    Args:
        args (argparse.Namespace): The arguments containing test parameters.
        test_loader (DataLoader): The test data loader.
        noise_test_loader (DataLoader): The noisy test data loader.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testing.initialize_parameters(cfg.model.T)
    model = load_model_and_weights(cfg.user.path_model)
    model = model.to(device)

    Original, restored_direct, restored_sampling, Noised = [], [], [], []
    
    T = cfg.model.T

    with torch.no_grad():
        model.eval()
        for eq_in, noise_in in tqdm(zip(test_loader, noise_test_loader), total=len(test_loader)):
            # eq_in = eq_in[1][:,args.channel_type,:].unsqueeze(dim=1).to(device)
            # reduce_noise = random.randint(*args.Range_RNF) * 0.01
            # noise_real = (noise_in[1][:,args.channel_type,:].unsqueeze(dim=1) * reduce_noise).to(device)
            eq_in = eq_in.to(device)
            noise_real = noise_in.to(device)
            signal_noisy = eq_in + noise_real
            t = torch.Tensor([T-1]).long().to(device)
            
            restored_dir = testing.direct_denoising(model, signal_noisy.to(device).float(), t)
            restored_direct.extend([x[0].cpu().numpy() for x in restored_dir])

            t = T-1
            restored_sample = testing.sample(
                                            model,
                                            signal_noisy.float(),
                                            t,
                                            batch_size=signal_noisy.shape[0]
                                            )
            restored_sampling.extend([x[0].cpu().numpy() for x in restored_sample[-1]])
            Original.extend(eq_in.squeeze().cpu().numpy())
            Noised.extend(signal_noisy.squeeze().cpu().numpy())

    return Original, restored_direct, restored_sampling, Noised

    # np.save(f"./Restored/Restored_direct_0.npy", np.array(restored_direct))
    # np.save(f"./Restored/Restored_sampling_0.npy", np.array(restored_sampling))
    # np.save(f"./Restored/Original.npy", np.array(Original))
    # np.save(f"./Restored/Noised.npy", np.array(Noised))