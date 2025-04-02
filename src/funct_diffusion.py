import torch
import torch.nn.functional as F


def get_params(timesteps, selected_beta_schedule):
    # Define beta schedule
    betas = selected_beta_schedule(timesteps=timesteps)

    # Define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # Calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # Calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    # Put all params into a dict and return that
    param_dict = {}
    param_dict['timesteps'] = timesteps
    param_dict['betas'] = betas
    param_dict['sqrt_recip_alpha'] = sqrt_recip_alphas
    param_dict['sqrt_alphas_cumprod'] = sqrt_alphas_cumprod
    param_dict['sqrt_one_minus_alphas_cumprod'] = sqrt_one_minus_alphas_cumprod
    param_dict['posterior_variance'] = posterior_variance
    return param_dict


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def diffusion_loss(param_dict, model, x_start, t, loss_funct):
    # Extract params
    sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance = \
        (param_dict[k] for k in ['sqrt_recip_alpha',
                                 'sqrt_alphas_cumprod',
                                 'sqrt_one_minus_alphas_cumprod',
                                 'posterior_variance'])
    # Generate noise
    noise = torch.randn_like(x_start)

    # Algo 1:
    # Forward diffusion with additive Gaussian property
    # Broadcast to time
    sqrt_alphas_cumprod_t = broadcast_timestep(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = broadcast_timestep(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    # Create noisy imges
    x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    # Predict noise
    out = model(x_noisy, t)
    # Calculate loss
    loss = loss_funct(noise, out)

    return loss


def broadcast_timestep(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


@torch.no_grad()
def generate_images(param_dict, model, image_size, image_cnt, channels=3):
    # Extract params
    timesteps, betas, sqrt_recip_alphas, sqrt_alphas_cumprod, \
        sqrt_one_minus_alphas_cumprod, posterior_variance = \
        (param_dict[k] for k in ['timesteps',
                                 'betas',
                                 'sqrt_recip_alpha',
                                 'sqrt_alphas_cumprod',
                                 'sqrt_one_minus_alphas_cumprod',
                                 'posterior_variance'])

    # Generate images with algorithm 2 (including returning all images)
    @torch.no_grad()
    def iterative_generation_step(x, generative_timestep, t_index):
        # Broadcast params based on t
        betas_t = broadcast_timestep(betas, generative_timestep, x.shape)
        sqrt_one_minus_alphas_cumprod_t = broadcast_timestep(
            sqrt_one_minus_alphas_cumprod, generative_timestep, x.shape)
        sqrt_recip_alphas_t = broadcast_timestep(sqrt_recip_alphas, generative_timestep, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, generative_timestep) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = broadcast_timestep(posterior_variance, generative_timestep, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # List to add generated images
    generated_imgs = []

    # Generate images from noise
    shape = (image_cnt, channels, image_size, image_size)
    device = next(model.parameters()).device

    # Start from noise (for each example in the batch)
    img = torch.randn(shape, device=device)

    for t_index in range(timesteps - 1, -1, -1):
        g_timestep = torch.full((image_cnt,), t_index, device=device, dtype=torch.long)
        # Iteratively denoise
        img = iterative_generation_step(img, g_timestep, t_index)
        generated_imgs.append(img.cpu())

    # [steps, image_cnt, image_CWH]
    return generated_imgs
