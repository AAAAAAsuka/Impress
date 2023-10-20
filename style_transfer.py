from datasets import load_dataset
import csv
import re
import os
import torch
from PIL import Image, ImageOps
import torchvision.transforms as T
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
import jsonlines
import numpy as np
import copy
import shutil
import argparse


def style_transfer(args, pipe_img2img,
                   aim_dir = f'../wikiart/preprocessed_data/claude-monet/clean/train/', aim_style_prompt = 'Cartoon'):


    save_dir = re.sub('clean', 'trans', copy.deepcopy(aim_dir)) + \
                f"/trans_{re.sub(' ', '_', copy.deepcopy(aim_style_prompt))}_seed{args.manual_seed}/"
    os.makedirs(save_dir, exist_ok=True)
    # copy meta data
    shutil.copy(f"{aim_dir}/metadata.jsonl", f"{save_dir}/metadata.jsonl")

    with open(f"{aim_dir}/metadata.jsonl", "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            file_name = item['file_name']
            # prompt = f"{item['text']} {aim_style_prompt} style"
            prompt = f"{aim_style_prompt} style"
            init_image = Image.open(f"{aim_dir}/{file_name}").convert("RGB")
            # with torch.autocast('cuda'):
            # init_image = init_image.to(device)
            torch.manual_seed(args.manual_seed)
            trans_image = pipe_img2img(prompt=prompt, image=init_image, strength=args.strength,
                                       guidance_scale=args.guidance, num_inference_steps=args.diff_steps).images[0]
            trans_image.save(f"{save_dir}/{file_name}")




def main(args):
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe_img2img.scheduler = DPMSolverMultistepScheduler.from_config(pipe_img2img.scheduler.config)
    pipe_img2img = pipe_img2img.to(device)
    pipe_img2img.enable_xformers_memory_efficient_attention()




    style_transfer(args, pipe_img2img,
                   aim_dir=f'../wikiart/preprocessed_data/{args.artist}/clean/train/', aim_style_prompt=args.aim_style)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='diffusion attack')

    # model_id_or_path = "runwayml/stable-diffusion-v1-5"
    # model_id_or_path = "CompVis/stable-diffusion-v1-4"
    # model_id_or_path = "CompVis/stable-diffusion-v1-3"
    # model_id_or_path = "CompVis/stable-diffusion-v1-2"
    # model_id_or_path = "CompVis/stable-diffusion-v1-1"
    parser.add_argument('--model', default='stabilityai/stable-diffusion-2-1-base', type=str,
                        help='stable diffusion weight')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--train', default= 0, type=bool,
                        help='training(True) or testing(False)')
    parser.add_argument('--artist', default='claude-monet', type=str,
                        help='stable diffusion weight')


    # parser.add_argument('--aim_style', type=str, default='Oil painting by Van Gogh')
    parser.add_argument('--aim_style', type=str, default='Cubism by Picasso')

    # stable diffusion Hyperparameters
    parser.add_argument('--strength', default=0.4, type=float, help='learning rate.')
    parser.add_argument('--guidance', default=7.5, type=float, help='learning rate.')
    parser.add_argument('--diff_steps', default=50, type=int, help='learning rate.')

    # Aim Model Hyperparameters
    parser.add_argument('--batch-size', default=128, type=int, help='batch size.')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate.')
    parser.add_argument('--epoch', default=50, type=int, help='training epoch.')
    # parser.add_argument('--norm', default=False, type=bool, help='normalize or not.')

    # Checkpoints

    # Miscs
    parser.add_argument('--manual_seed', default=0, type=int, help='manual seed')

    # Device options
    parser.add_argument('--device', default='cuda:3', type=str,
                        help='device used for training')

    args = parser.parse_args()
    np.random.seed(seed = args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.backends.cudnn.deterministic=True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)