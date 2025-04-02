from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import Adam

# Model is here
from cls_model import Unet
# Data is here
from cls_dataset import get_tiny_imagenet_data_loader
# Forward and backward difussion functions are here
from funct_diffusion import get_params, diffusion_loss, linear_beta_schedule

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == "__main__":
    # Manual seed for reproducibility
    torch.manual_seed(0)

    # Diffusion parameters
    tr_epochs = 8000
    timesteps = 500
    selected_beta_scheduler = linear_beta_schedule
    diffusion_param_dict = get_params(timesteps, selected_beta_scheduler)

    # Select one of the losses
    selected_loss = F.l1_loss
    # selected_loss = F.mse_loss
    # selected_loss = F.smooth_l1_loss

    # Dataset/Dataloader parameters
    channels = 3
    image_size = 64
    batch_size = 64
    selected_class = 0
    # Get data loder from cls_dataset
    data_loader = get_tiny_imagenet_data_loader(image_size,
                                                batch_size,
                                                selected_class)

    # Define model
    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4, 8)
    )
    model.cuda()
    device = next(model.parameters()).device
    # Optimizer
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Main training loop
    for epoch in range(tr_epochs):
        print(f'Epoch:{epoch} starts')
        for index, images in enumerate(data_loader):
            current_batch_size = images.shape[0]

            # Send images to GPU
            images = images.cuda(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            sampled_timesteps = torch.randint(0, timesteps, (current_batch_size,), device=device).long()

            # Zero grads
            optimizer.zero_grad()

            # Calculate loss based on the selected loss
            loss = diffusion_loss(diffusion_param_dict,
                                  model,
                                  images,
                                  sampled_timesteps,
                                  selected_loss)

            # Calculate gradients and update
            loss.backward()
            optimizer.step()

            print(f'Loss at epoch:{epoch} batch:{index} is {loss.item()}')

    torch.save(model.state_dict(), 'trained_model.pth')
