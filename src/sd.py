from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch


model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "A unity render of planet earth in hyper real detail"
image = pipe(prompt).images[0]
    
image.save("hello_world.png")
