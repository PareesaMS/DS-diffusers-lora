#from diffusers import StableDiffusionPipeline
from local_pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers import StableDiffusionPipeline as StableDiffusionPipelineOld
import torch
import pdb

seed = 123450011
#new_model = "/home/pagolnar/myclones/sd-ft-model-nikki"
#old_model = "/home/pagolnar/myclones/clone23-hf/diffusers/tests/p_tests/stable-diffusion-v1-5"
new_model = "/home/pagolnar/clones/clone-distill/DS-diffusers/examples/dreambooth/sd-distill_alaki"
old_model = "CompVis/stable-diffusion-v1-4"
image_out_dir = "/home/pagolnar/clones/clone-distill/DS-diffusers/examples/dreambooth/out5/"

prompt = "A person holding a cat"
#pdb.set_trace()


#--- new image
pipe_new = StableDiffusionPipeline.from_pretrained(new_model, torch_dtype=torch.float16).to("cuda")
generator = torch.Generator("cuda").manual_seed(seed)
image_new = pipe_new(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]


#image_new.save("/home/pagolnar/myclones/clone23-hf/diffusers/tests/p_tests/cmp_output/"+"NEW5__seed_"+str(seed)+"_"+prompt[1:100]+".png")
image_new.save(image_out_dir+"NEW__seed_"+str(seed)+"_"+prompt[0:100]+".png")

#pdb.set_trace()
#--- old image and optimization
pipe_old = StableDiffusionPipeline.from_pretrained(old_model, torch_dtype=torch.float16).to("cuda")
generator = torch.Generator("cuda").manual_seed(seed)                                              
image_old = pipe_old(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]                 
                                                                                                   
image_old.save(image_out_dir+"OLD_seed_"+str(seed)+"_"+prompt[0:100]+".png")

#pdb.set_trace()
#--- old image no optimization                                                                    
pipe_old_no_opt = StableDiffusionPipelineOld.from_pretrained(old_model, torch_dtype=torch.float16).to("cuda")
generator = torch.Generator("cuda").manual_seed(seed)                                              
image_old_no_opt = pipe_old_no_opt(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]                 
                                                                                                   
image_old_no_opt.save(image_out_dir+"OLD_NO_OPT_seed_"+str(seed)+"_"+prompt[0:100]+".png")                       
