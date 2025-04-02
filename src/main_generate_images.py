import pdb
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Lambda, ToPILImage

# Model is here
from cls_model import Unet
# Forward and backward difussion functions are here
from funct_diffusion import get_params, generate_images, linear_beta_schedule

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def sample_images_from_diffusion(diffusion_steps, nth_step, reverse_transform):
    sampled_images = []
    no_of_generations = len(diffusion_steps[0])

    # Get images at the nth step and the final one
    # 500 steps means index 499 is the final image, adjust this accordingly
    step_indices = [0, 200, 400, 450, 480, 499]
    print(step_indices)
    # For each image
    for im_id in range(no_of_generations):
        sampled_images.append([])
        for idx in step_indices:
            selected_image = diffusion_steps[idx][im_id]
            # Take the final image from the N images at that step
            pil_image = reverse_transform(selected_image.cpu())
            sampled_images[-1].append(pil_image)

    return sampled_images


def save_images_side_by_side(images, save_path, gap=5, gap_color=(0, 0, 0)):
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths) + gap * (len(images) - 1)
    max_height = max(heights)

    stitched_image = Image.new('RGB', (total_width, max_height), color=gap_color)

    x_offset = 0
    for img in images:
        stitched_image.paste(img, (x_offset, 0))
        x_offset += img.width + gap

    stitched_image.save(save_path)


if __name__ == "__main__":
    torch.manual_seed(99)
    # Output saving parameters
    output_folder = Path("../results/")
    output_folder.mkdir(exist_ok=True)

    # Diffusion parameters
    timesteps = 500
    selected_beta_scheduler = linear_beta_schedule
    diffusion_param_dict = get_params(timesteps, selected_beta_scheduler)

    # Dataset/Dataloader parameters
    channels = 3
    image_size = 64
    image_cnt = 100

    # Load model
    model_path = '../models/trained_model.pth'
    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4, 8)
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))  # Load weights
    model.eval()
    model.cuda()
    device = next(model.parameters()).device

    # Optimizer
    reverse_transform = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage()
    ])

    # Generate images
    generated_ims = generate_images(diffusion_param_dict,
                                    model,
                                    image_size=image_size,
                                    image_cnt=image_cnt,
                                    channels=channels)

    # Sample images to see denoising
    sample_every_nth_step = 100
    selected_ims = sample_images_from_diffusion(generated_ims,
                                                sample_every_nth_step,
                                                reverse_transform)

    for image_id in range(image_cnt):
        save_images_side_by_side(selected_ims[image_id], output_folder / f'stitched_{image_id}.png')
        selected_ims[image_id][-1].save(output_folder / f'final_{image_id}.png')
