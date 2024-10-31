import os
import json
import numpy as np
from PIL import Image
import torch
from pipeline_flux_regional import RegionalFluxPipeline, RegionalFluxAttnProcessor2_0
from pipeline_flux_controlnet_regional import RegionalFluxControlNetPipeline
from diffusers import FluxControlNetModel, FluxMultiControlNetModel

if __name__ == "__main__":
    
    model_path = "black-forest-labs/FLUX.1-dev"
    pipeline = RegionalFluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
    
    use_lora = False
    use_controlnet = False

    if use_controlnet:
        # READ https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro for detailed usage tutorial
        controlnet_model_union = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro'
        controlnet_union = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
        controlnet = FluxMultiControlNetModel([controlnet_union])
        pipeline = RegionalFluxControlNetPipeline.from_pretrained(model_path, controlnet=controlnet, torch_dtype=torch.bfloat16).to("cuda")
    else:
        pipeline = RegionalFluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
    
    if use_lora:
        # READ https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-Children-Simple-Sketch for detailed usage tutorial
        pipeline.load_lora_weights("Shakker-Labs/FLUX.1-dev-LoRA-Children-Simple-Sketch", weight_name="FLUX-dev-lora-children-simple-sketch.safetensors")
    
    attn_procs = {}
    for name in pipeline.transformer.attn_processors.keys():
        if 'transformer_blocks' in name and name.endswith("attn.processor"):
            attn_procs[name] = RegionalFluxAttnProcessor2_0()
        else:
            attn_procs[name] = pipeline.transformer.attn_processors[name]
    pipeline.transformer.set_attn_processor(attn_procs)

    ## generation settings
    image_width = 1280
    image_height = 968
    num_samples = 1
    num_inference_steps = 24
    guidance_scale = 3.5
    seed = 124

    # controlnet settings
    if use_controlnet:
        control_image = [Image.open("control-depth.png")]
        control_mode = [2] # (2) stands for depth control
        controlnet_conditioning_scale = [0.7]
    
    # lora settings
    if use_lora:
        pipeline.fuse_lora(lora_scale=0.8) # recommended 0.8~1.0, trigger words: [Sketch-style]

    ## regional settings

    # example regional prompt and mask pairs
    base_prompt = "A striking scene of a woman with vibrant red hair stepping out of a sleek black sports car, surrounded by city lights and urban ambiance."
    background_prompt = "city lights" # needed if regional masks don't cover the whole image
    regional_prompt_mask_pairs = {
        "0": {
            "description": "The woman is wearing a black elegant dress, her vivid red hair cascades down, catching the city lights and contrasting against the sleek, polished black surface of the sports car.",
            "mask": [640, 0, 1280, 768]
        },
        "1": {
            "description": "The glossy black sports car reflects the surrounding lights and shadows, highlighting its aerodynamic curves and bold, modern design.",
            "mask": [0, 0, 640, 768]
        }
    }
    mask_inject_steps = 10
    double_inject_blocks = 18
    single_inject_blocks = 39
    base_ratio = 0.2

    # example input with controlnet enabled
    # base_prompt = "Three high-performance sports cars, red, blue, and yellow, are racing side by side on a city street"
    # background_prompt = "city street" # needed if regional masks don't cover the whole image
    # regional_prompt_mask_pairs = {
    #         "0": {
    #             "description": "A sleek red sports car in the lead position, with aggressive aerodynamic styling and gleaming paint that catches the light. The car appears to be moving at high speed with motion blur effects.",
    #             "mask": [0, 0, 426, 968]
    #         },
    #         "1": {
    #             "description": "A powerful blue sports car in the middle position, neck-and-neck with its competitors. Its metallic paint shimmers as it races forward, with visible speed lines and dynamic movement.",
    #             "mask": [426, 0, 853, 968]
    #         },
    #         "2": {
    #             "description": "A striking yellow sports car in the third position, its bold color standing out against the street. The car's aggressive stance and aerodynamic profile emphasize its racing performance.",
    #             "mask": [853, 0, 1280, 968]
    #         }
    # }
    # mask_inject_steps = 10
    # double_inject_blocks = 18
    # single_inject_blocks = 20
    # base_ratio = 0.3

    # example input with lora enabled
    # base_prompt = "Sketched style: A cute dinosaur playfully blowing tiny fire puffs over a cartoon city in a cheerful scene."
    # background_prompt = "white background"
    # regional_prompt_mask_pairs = {
    #     "0": {
    #         "description": "Sketched style: dinosaur with round eyes and a mischievous smile, puffing small flames over the city.",
    #         "mask": [0, 0, 640, 1280]
    #     },
    #     "1": {
    #         "description": "Sketched style: city with colorful buildings and tiny flames gently floating above, adding a playful touch.", 
    #         "mask": [640, 0, 1280, 1280]
    #     }
    # }
    # mask_inject_steps = 10
    # double_inject_blocks = 18
    # single_inject_blocks = 39
    # base_ratio = 0.2

    ## prepare regional prompts and masks
    # ensure image width and height are divisible by the vae scale factor
    image_width = (image_width // pipeline.vae_scale_factor) * pipeline.vae_scale_factor
    image_height = (image_height // pipeline.vae_scale_factor) * pipeline.vae_scale_factor

    regional_prompts = []
    regional_masks = []
    background_mask = torch.ones((image_height, image_width))
    for _, region in regional_prompt_mask_pairs.items():
        description = region['description']
        mask = region['mask']
        x1, y1, x2, y2 = mask
        mask = torch.zeros((image_height, image_width))
        mask[y1:y2, x1:x2] = 1.0

        background_mask -= mask

        regional_prompts.append(description)
        regional_masks.append(mask)
            
    # if regional masks don't cover the whole image, append background prompt and mask
    if background_mask.sum() > 0:
        regional_prompts.append(background_prompt)
        regional_masks.append(background_mask)
    
    # setup regional kwargs that pass to the pipeline
    joint_attention_kwargs = {
        'regional_prompts': regional_prompts,
        'regional_masks': regional_masks,
        'double_inject_blocks': double_inject_blocks,
        'single_inject_blocks': single_inject_blocks,
        'base_ratio': base_ratio,
    }
    # generate images
    if use_controlnet:
        images = pipeline(
            prompt=base_prompt,
            num_samples=num_samples,
            width=image_width, height=image_height,
            mask_inject_steps=mask_inject_steps,
            control_image=control_image,
            control_mode=control_mode,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cuda").manual_seed(seed),
            joint_attention_kwargs=joint_attention_kwargs,
        ).images
    else:
        images = pipeline(
            prompt=base_prompt,
            num_samples=num_samples,
            width=image_width, height=image_height,
            mask_inject_steps=mask_inject_steps,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cuda").manual_seed(seed),
            joint_attention_kwargs=joint_attention_kwargs,
        ).images

    for idx, image in enumerate(images):
        image.save(f"output_{idx}.jpg")