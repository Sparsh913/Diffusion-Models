import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(prompt: str, uncond_prompt: str, input_image = None, strength = 0.8, do_cfg = True, cfg_scale = 7.5,
             sampler_name = "ddpm", n_inference_steps = 50, models = {}, seed = None, device = None, idle_device = None,
             tokenizer = None):
    
    # cfg_scale is the measure of how much do we want to pay attention to the prompt (conditioned vs unconditioned)
    # cfg is classifier free guidance
    # strength is the measure of how much attention we want to pay to the input image
    # uncond_prompt is the negative prompt (or can be empty)
    
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("Strength must be in (0, 1]")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
            
        generator = torch.Generator(device = device)
        if seed is None:
            generate.seed()
        else:
            generator.manual_seed(seed)
            
        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            # convert the prompt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding = "max_length", max_length = 77).input_ids
            # (Batch Size, Seq_Length)
            cond_tokens = torch.tensor(cond_tokens, dtype = torch.long, device=device)
            # (Batch Size, Seq_Length) -> (Batch Size, Seq_Length, Dim)
            cond_context = clip(cond_tokens)
            
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding = "max_length", max_length = 77).input_ids
            uncond_tokens = torch.tensor(uncond_prompt, dtype = torch.long, device=device)
            # (Batch Size, Seq_Length) -> (Batch Size, Seq_Length, Dim)
            uncond_context = clip(uncond_tokens)
            
            # (2, Seq Length, Dim) = (2, 77, 768)
            context = torch.cat([cond_context, uncond_context]) # Concatenating the two prompts
            
        else:
            # Convert it into a list of tokens
            token = tokenizer.batch_encode_plus([prompt], padding = "max_length", max_length = 77).input_ids
            tokens = torch.tensor(tokens, dtype = torch.long, device=device)
            
            # (1, 77, 768)
            context = clip(tokens)
        to_idle(clip)
        
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
            
        else:
            raise ValueError(f"Unknown sampler {sampler_name}")
        
        latent_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)
            
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            
            # (Height, Width, Channels)
            input_image_tensor = torch.tensor(input_image_tensor, dtype = torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            
            # (Height, Width, Channels) -> (Batch Size, Height, Width, Channels)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            
            # (Batch Size, Height, Width, Channels) -> (Batch Size, Channels, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
            
            encoder_noise = torch.randn(latent_shape, generator=generator, device = device)
            # run the image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise)
            
            sampler.set_strength(strength = strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0]) # Add noise to the latents
            
            to_idle(encoder)
            
        else:
            # If we're doing text-to-image, start from random noise N(0,1)
            latents = torch.randn(latent_shape, generator = generator, device = device)
        
        diffusion = models["diffusion"] # UNET
        diffusion.to(device)
        
        timesteps = tqdm(sampler.timesteps)
        
        for i, temestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)
            
            # (Batch Size, 4, Latents Height, Latents Width)
            model_input = latents
            
            if do_cfg:
                # (Batch Size, 4, Latents Height, Latents Width) -> (2 * Batch Size, 4, Latents Height, Latents Width)
                model_input = model_input.repeat(2, 1, 1, 1)
                
            # model_output is the predicted noise by the UNET
            model_output = diffusion(model_input, context, time_embedding)
            
            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond # Standard formula for CFG -> See slides
                
            # Remove the noise predicted from UNET
            latents = sampler.step(timestep, latents, model_output)
            
        to_idle(diffusion)
        
        decoder = models["decoder"]
        decoder.to(device)
        
        images = decoder(latents)
        to_idle(decoder)
        
        images = rescale(images, (-1, 1), (0, 255), clamp = True)
        
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        
        return images[0]
    
def rescale(x, old_range, new_range, clamp = False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    
    if clamp:
        x = x.clamp_(min = new_min, max = new_max)
    return x

def get_time_embedding(timestep):
    # (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype = torch.float32) / 160)
    # (1, 160)
    x = torch.tensor([timestep], dtype = torch.float32)[:, None] * freqs[None] # Indexing a tensor with None adds a new dimension, same as unsqueeze
    
    # (1, 320)
    return torch.cat([torch.sin(x), torch.cos(x)], dim = -1)