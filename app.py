import gradio as gr
import torch
import os
import glob
import numpy as np

from datetime import datetime
from PIL import Image
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from pipline_StableDiffusion_ConsistentID import ConsistentIDStableDiffusionPipeline
from huggingface_hub import hf_hub_download
### Model can be imported from https://github.com/zllrunning/face-parsing.PyTorch?tab=readme-ov-file
### We use the ckpt of 79999_iter.pth: https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812
### Thanks for the open source of face-parsing model.
from models.BiSeNet.model import BiSeNet

# zero = torch.Tensor([0]).cuda()
# print(zero.device) # <-- 'cpu' 🤔
# device = zero.device # "cuda"
device = "cuda"

# Gets the absolute path of the current script
script_directory = os.path.dirname(os.path.realpath(__file__))

# download ConsistentID checkpoint to cache
base_model_path = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
consistentID_path = hf_hub_download(repo_id="JackAILab/ConsistentID", filename="ConsistentID-v1.bin", repo_type="model")

### Load base model
pipe = ConsistentIDStableDiffusionPipeline.from_pretrained(
    base_model_path, 
    torch_dtype=torch.float16, 
    safety_checker=None, # use_safetensors=True, 
    variant="fp16"
).to(device)

### Load other pretrained models
## BiSenet
bise_net_cp_path = hf_hub_download(repo_id="JackAILab/ConsistentID", filename="face_parsing.pth", local_dir="./checkpoints")
bise_net = BiSeNet(n_classes = 19)
bise_net.load_state_dict(torch.load(bise_net_cp_path, map_location="cpu")) # device fail
bise_net.cuda()

"""
import sys
sys.path.append("models\\LLaVA")
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# Load Llava for prompt enhancement
llva_model_path = "liuhaotian/llava-v1.5-7b"  
llva_tokenizer, llva_model, llva_image_processor, llva_context_len = load_pretrained_model(
    model_path=llva_model_path,
    model_base=None,
    model_name=get_model_name_from_path(llva_model_path),
    offload_folder="offload"
)
# llva_tokenizer.cuda()
# llva_model.to(device)
# llva_image_processor.to(device)
"""
### Load consistentID_model checkpoint
pipe.load_ConsistentID_model(
    os.path.dirname(consistentID_path),
    bise_net,
    subfolder="",
    weight_name=os.path.basename(consistentID_path),
    trigger_word="img",
)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

### Load to cuda
pipe.to(device)
pipe.image_encoder.to(device)
pipe.image_proj_model.to(device)
pipe.FacialEncoder.to(device)


    
def process(selected_template_images,costum_image,prompt
        ,negative_prompt,prompt_selected,retouching,model_selected_tab,prompt_selected_tab,width,height,merge_steps,seed_set):
    
    if model_selected_tab==0:
        select_images = load_image(Image.open(selected_template_images))
    else:
        select_images = load_image(Image.fromarray(costum_image))

    if prompt_selected_tab==0:
        prompt = prompt_selected
        negative_prompt = ""
        need_safetycheck = False
    else:
        need_safetycheck = True

    # hyper-parameter
    num_steps = 50
    seed_set = torch.randint(0, 1000, (1,)).item()
    # merge_steps = 30
    """            
    @torch.inference_mode()
    def Enhance_prompt(prompt,select_images):
        
        llva_prompt = f'Please ignore the image. Enhance the following text prompt for me. You can associate more details with the character\'s gesture, environment, and decent clothing:"{prompt}".' 
        args = type('Args', (), {
            "model_path": llva_model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(llva_model_path),
            "query": llva_prompt,
            "conv_mode": None,
            "image_file": select_images,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })() 
        Enhanced_prompt = eval_model(args, llva_tokenizer, llva_model, llva_image_processor)
    
        return Enhanced_prompt
    """
    if prompt == "":
        prompt = "A man, in a forest"
        prompt = "A man, with backpack, in a raining tropical forest, adventuring, holding a flashlight, in mist, seeking animals"
        prompt = "A person, in a sowm, wearing santa hat and a scarf, with a cottage behind"
    else:
        # prompt=Enhance_prompt(prompt,Image.new('RGB', (200, 200), color = 'white'))
        print(prompt)
        pass

    if negative_prompt == "":
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

    #Extend Prompt
    prompt = "cinematic photo," + prompt + ", 50mm photograph, half-length portrait, film, bokeh, professional, 4k, highly detailed"

    negtive_prompt_group="((cross-eye)),((cross-eyed)),((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))). out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))"
    negative_prompt = negative_prompt + negtive_prompt_group
    
    # seed = torch.randint(0, 1000, (1,)).item()
    generator = torch.Generator(device=device).manual_seed(seed_set)

    images = pipe(
        prompt=prompt,
        width=width,    
        height=height,
        input_id_images=select_images,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        num_inference_steps=num_steps,
        start_merge_step=merge_steps,
        generator=generator,
        retouching=retouching,
        need_safetycheck=need_safetycheck,
    ).images[0]

    current_date = datetime.today()
    return np.array(images)

# Gets the templates
script_directory = os.path.dirname(os.path.realpath(__file__))
preset_template = glob.glob("./images/templates/*.png")
preset_template = preset_template + glob.glob("./images/templates/*.jpg")


with gr.Blocks(title="ConsistentID Demo") as demo:
    gr.Markdown("# ConsistentID Demo")
    gr.Markdown("\
        Put the reference figure to be redrawn into the box below (There is a small probability of referensing failure. You can submit it repeatedly)")
    gr.Markdown("\
        If you find our work interesting, please leave a star in GitHub for us!<br>\
        https://github.com/JackAILab/ConsistentID")
    with gr.Row():
        with gr.Column():
            model_selected_tab = gr.State(0)
            with gr.TabItem("template images") as template_images_tab:
                template_gallery_list = [(i, i) for i in preset_template]
                gallery = gr.Gallery(template_gallery_list,columns=[4], rows=[2], object_fit="contain", height="auto",show_label=False)
                
                def select_function(evt: gr.SelectData):
                    return preset_template[evt.index]

                selected_template_images = gr.Text(show_label=False, visible=False, placeholder="Selected")
                gallery.select(select_function, None, selected_template_images)
            with gr.TabItem("Upload Image") as upload_image_tab:
                costum_image = gr.Image(label="Upload Image")

            model_selected_tabs = [template_images_tab, upload_image_tab]
            for i, tab in enumerate(model_selected_tabs):
                tab.select(fn=lambda tabnum=i: tabnum, inputs=[], outputs=[model_selected_tab])

            with gr.Column():
                prompt_selected_tab = gr.State(0)
                with gr.TabItem("template prompts") as template_prompts_tab:
                    prompt_selected = gr.Dropdown(value="A person, police officer, half body shot", elem_id='dropdown', choices=[
                        "A woman in a wedding dress",
                        "A woman, queen, in a gorgeous palace",
                        "A man sitting at the beach with sunset", 
                        "A person, police officer, half body shot", 
                        "A man, sailor, in a boat above ocean",
                        "A women wearing headphone, listening music", 
                        "A man, firefighter, half body shot"], label=f"prepared prompts")

                with gr.TabItem("custom prompt") as custom_prompt_tab:
                    prompt = gr.Textbox(label="prompt",placeholder="A man/woman wearing a santa hat")
                    nagetive_prompt = gr.Textbox(label="negative prompt",placeholder="monochrome, lowres, bad anatomy, worst quality, low quality, blurry")
            
                prompt_selected_tabs = [template_prompts_tab, custom_prompt_tab]
                for i, tab in enumerate(prompt_selected_tabs):
                    tab.select(fn=lambda tabnum=i: tabnum, inputs=[], outputs=[prompt_selected_tab])
            
            retouching = gr.Checkbox(label="face retouching",value=False,visible=False)
            width = gr.Slider(label="image width",minimum=256,maximum=768,value=512,step=8)
            height = gr.Slider(label="image height",minimum=256,maximum=768,value=768,step=8)
            width.release(lambda x,y: min(1280-x,y), inputs=[width,height], outputs=[height])
            height.release(lambda x,y: min(1280-y,x), inputs=[width,height], outputs=[width])
            merge_steps = gr.Slider(label="step starting to merge facial details(30 is recommended)",minimum=10,maximum=50,value=30,step=1)
            seed_set = gr.Slider(label="set the random seed for different results",minimum=1,maximum=2147483647,value=2024,step=1)
            
            btn = gr.Button("Run")
        with gr.Column():
            out = gr.Image(label="Output")
            gr.Markdown('''
                N.B.:<br/>
                - If the proportion of face in the image is too small, the probability of an error will be slightly higher, and the similarity will also significantly decrease.)
                - At the same time, use prompt with \"man\" or \"woman\" instead of \"person\" as much as possible, as that may cause the model to be confused whether the protagonist is male or female.
                ''')
        btn.click(fn=process, inputs=[selected_template_images,costum_image,prompt,nagetive_prompt,prompt_selected,retouching
            ,model_selected_tab,prompt_selected_tab,width,height,merge_steps,seed_set], outputs=out)

demo.launch()