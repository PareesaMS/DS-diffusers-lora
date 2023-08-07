from local_pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline as StableDiffusionPipelineOld
import torch
import pdb
import os

seed = 12345001

prompts = ["A boy is watching TV",
           "A photo of a person dancing in the rain",
           "A photo of a boy jumping over a fence",
           "A photo of a boy is kicking a ball",
           "A beach with a lot of waves on it",
           "A road that is going down a hill",
           "Abraham Lincoln touches his toes while George Washington does chin-ups  Lincoln is barefoot",
           "A snowy forest with trees covered in snow",
           "A path through a forest with fog and tall trees",
           "A waterfall with a tree in the middle of it",
           "A foggy sunrise over a valley with trees and hills",
           "A black and white photo of a mountain range",
           "A picture of a one-dollar money bill",
           "Supreme Court Justices play a baseball game with the FBI",
           "A picture of Coco Cola can",
           "A picture of Costco store",
           "A profile photo for a smart, engaging digital assistant",
           "A picture of a multilingual Bert hanging out with Elmo and Ernie",
           "A picture of water pouring out of a jar in outer space",
           "Futuristic view of Delhi when India becomes a developed country as digital art",
           "A silver dragon head",
           "A pear cut into seven pieces arranged in a ring",
           "An armchair in the shape of an avocado",
           "An old man is talking to his parents",
           "A grocery store refrigerator has pint cartons of milk on the top shelf, quart cartons on the middle shelf, and gallon plastic jugs on the bottom shelf",
           "An oil painting of a couple in formal evening wear going home get caught in a heavy downpour with no umbrellas",
           "Paying for a pizza with quarters",
           "Wild turkeys in a garden seen from inside the house through a screen door",
           "A watercolor of a silver dragon head with colorful flowers growing out of the top on a colorful smooth gradient background",
           "A red basketball with flowers on it, in front of blue one with a similar pattern",
           "A Hokusai painting of a happy dragon head with flowers growing out of the top",
           "3d rendering of 5 tennis balls on top of a cake",
           "A person holding a drink of soda",
           "A person is squeezing a lemon",
           "A person holding a cat"]

new_model = "/home/pagolnar/clones/DS-diffusers-lora/examples/dreambooth/model-lora-distill-2/"
old_model = "runwayml/stable-diffusion-v1-5"                                                   
image_out_dir = "/home/pagolnar/clones/DS-diffusers-lora/examples/dreambooth/out3-loop/"

if not os.path.exists(image_out_dir):
    os.makedirs(image_out_dir)


for prompt in prompts:
    pipe_new = StableDiffusionPipeline.from_pretrained(old_model, torch_dtype=torch.float16).to("cuda")
    pipe_old = StableDiffusionPipelineOld.from_pretrained(old_model, torch_dtype=torch.float16).to("cuda")

    pipe_new.scheduler = DPMSolverMultistepScheduler.from_config(pipe_new.scheduler.config)
    pipe_old.scheduler = DPMSolverMultistepScheduler.from_config(pipe_old.scheduler.config)

    #--- baseline image
    generator = torch.Generator("cuda").manual_seed(seed)                                              
    image_old = pipe_old(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image_old.save(image_out_dir+"BASELINE_seed_"+str(seed)+"_"+prompt[0:100]+".png")

    #--- new image
    pipe_new.unet.load_attn_procs(new_model)
    #--- No LoRA
    generator = torch.Generator("cuda").manual_seed(seed)                                          
    image_new = pipe_new(prompt, num_inference_steps=50, guidance_scale=7.5, scale=0).images[0]    
    image_new.save(image_out_dir+"NEW_NoLoRA_seed_"+str(seed)+"_"+prompt[0:100]+".png")
    #--- LoRA
    generator = torch.Generator("cuda").manual_seed(seed)                                          
    image_new = pipe_new(prompt, num_inference_steps=50, guidance_scale=7.5, scale=1).images[0]    
    image_new.save(image_out_dir+"NEW_LoRA_seed_"+str(seed)+"_"+prompt[0:100]+".png")                    
