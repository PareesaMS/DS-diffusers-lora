#from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from local_pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline as StableDiffusionPipelineOld
import torch
import pdb

seed = 12345001
new_model = "/home/pagolnar/clones/DS-diffusers-lora/examples/dreambooth/model-lora-distill/"
old_model = "runwayml/stable-diffusion-v1-5"
image_out_dir = "/home/pagolnar/clones/DS-diffusers-lora/examples/dreambooth/out1/"

prompt = "A person holding a cat"

#--- old image no optimization                                                                               
pipe = StableDiffusionPipeline.from_pretrained(old_model, torch_dtype=torch.float16).to("cuda")
pipe_old = StableDiffusionPipelineOld.from_pretrained(old_model, torch_dtype=torch.float16).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe_old.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
generator = torch.Generator("cuda").manual_seed(seed)                                                        
image_old_no_opt = pipe_old(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]             
image_old_no_opt.save(image_out_dir+"OLD_NO_OPT_seed_"+str(seed)+"_"+prompt[0:100]+".png")


# #--- new image
#pdb.set_trace()
pipe.unet.load_attn_procs(new_model)
generator = torch.Generator("cuda").manual_seed(seed)
#pdb.set_trace()
image_new = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, scale=0).images[0]
image_new.save(image_out_dir+"NEW__seed_"+str(seed)+"_"+prompt[0:100]+".png")



    

    
