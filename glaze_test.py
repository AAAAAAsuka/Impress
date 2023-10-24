import torch
import torch.nn as nn
import torchvision.models as models

import os
from PIL import Image, ImageOps
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import requests
from tqdm import tqdm
from io import BytesIO
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
import torchvision.transforms as T
import sys
from utils import preprocess, recover_image, plot
import argparse
import jsonlines
from glaze import glaze
import re
import copy

def main(args):
    # make sure you're logged in with `huggingface-cli login` - check https://github.com/huggingface/diffusers for more details
    to_pil = T.ToPILImage()
    if args.clean_model:
        args.checkpoint = "stabilityai/stable-diffusion-2-1-base"
    pipe_text2img = StableDiffusionPipeline.from_pretrained(args.checkpoint, torch_dtype=torch.float16)
    pipe_text2img.scheduler = DPMSolverMultistepScheduler.from_config(pipe_text2img.scheduler.config)
    pipe_text2img = pipe_text2img.to(device)
    # pipe_text2img.enable_xformers_memory_efficient_attention()

    os.makedirs(args.save_dir, exist_ok=True)

    with open(f"{args.test_data_dir}/metadata.jsonl", "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            file_name = item['file_name']
            prompt = item['text']
            torch.manual_seed(args.manual_seed)
            diff_purified_image = pipe_text2img(prompt=prompt, num_inference_steps=args.diff_steps).images[0]
            diff_purified_image.save(f"{args.save_dir}/{file_name}")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='diffusion attack')

    parser.add_argument('-c', '--checkpoint', default='../stable_diffusion_models/claude-monet', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')

    parser.add_argument('--clean_model', type=bool, default=False)

    # data
    parser.add_argument('--test_data_dir', type=str,
                        default='../wikiart/preprocessed_data/claude-monet/trans/test/trans_Cubism_by_Picasso_seed9222')
    parser.add_argument('--save_dir', type=str,
                        default='../wikiart/preprocessed_data/claude-monet/trans/test/trans_Cubism_by_Picasso_seed9222')


    parser.add_argument('--diff_steps', default=200, type=int, help='learning rate.')


    # Checkpoints

    # parser.add_argument('--model_name', default='/cnn_mnist.pth', type=str,
    #                     help='network structure choice')
    # Miscs
    parser.add_argument('--manual_seed', default=0, type=int, help='manual seed')

    # Device options
    parser.add_argument('--device', default='cuda', type=str,
                        help='device used for training')

    args = parser.parse_args()
    np.random.seed(seed = args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.backends.cudnn.deterministic=True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)


