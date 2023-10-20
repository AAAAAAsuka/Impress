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
from impress import impress
import shutil

def main(args):
    # make sure you're logged in with `huggingface-cli login` - check https://github.com/huggingface/diffusers for more details
    to_pil = T.ToPILImage()
    pipe_img2img = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe_img2img.scheduler = DPMSolverMultistepScheduler.from_config(pipe_img2img.scheduler.config)
    pipe_img2img = pipe_img2img.to(device)
    # pipe_img2img.enable_xformers_memory_efficient_attention()

    save_dir_adv = re.sub('/trans/', f'/{args.adv_para}/', copy.deepcopy(args.trans_data_dir))
    save_dir_pur = re.sub('/trans/', f'/{args.pur_para}/', copy.deepcopy(args.trans_data_dir))

    os.makedirs(save_dir_adv, exist_ok=True)
    os.makedirs(save_dir_pur, exist_ok=True)

    for name, param in pipe_img2img.vae.named_parameters():
        param.requires_grad = False
    shutil.copy(f"{args.clean_data_dir}/metadata.jsonl", f"{save_dir_pur}/metadata.jsonl")

    with open(f"{args.clean_data_dir}/metadata.jsonl", "r+", encoding="utf8") as f:

        for item in jsonlines.Reader(f):
            file_name = item['file_name']
            adv_image = Image.open(f"{save_dir_adv}/{file_name}").convert("RGB")

            x_adv = preprocess(adv_image).to(device).half()
            x_purified = impress(x_adv,
                                 model=pipe_img2img.vae,
                                 clamp_min=-1,
                                 clamp_max=1,
                                 eps=args.pur_eps,
                                 iters=args.pur_iters,
                                 lr=args.pur_lr,
                                 pur_alpha=args.pur_alpha,
                                 noise=args.pur_noise, )

            x_purified = (x_purified / 2 + 0.5).clamp(0, 1)
            purified_image = to_pil(x_purified[0]).convert("RGB")
            purified_image.save(f"{save_dir_pur}/{file_name}")




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='diffusion attack')


    # model_id = "stabilityai/stable-diffusion-2-1"
    parser.add_argument('--model', default='stabilityai/stable-diffusion-2-1-base', type=str,
                        help='stable diffusion weight')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--train', default= 0, type=bool,
                        help='training(True) or testing(False)')

    # data
    parser.add_argument('--clean_data_dir', type=str, default='../wikiart/preprocessed_data/claude-monet/clean/train/')
    parser.add_argument('--trans_data_dir', type=str, default='../wikiart/preprocessed_data/claude-monet/trans/train/trans_Cubism_by_Picasso_seed9222')

    parser.add_argument('--neg_feed', type=float, default=-1.)
    parser.add_argument('--adv_para', type=str, default=None)
    parser.add_argument('--pur_para', type=str, default=None)

    # ae Hyperparameters
    parser.add_argument('--pur_eps', default=0.1, type=float, help='ae Hyperparameters')
    parser.add_argument('--pur_iters', default=3000, type=int, help='ae Hyperparameters')
    parser.add_argument('--pur_lr', default=0.01, type=float, help='ae Hyperparameters')
    parser.add_argument('--pur_alpha', default=0.1, type=float, help='ae Hyperparameters')
    parser.add_argument('--pur_noise', default=0., type=float, help='ae Hyperparameters')



    # Miscs
    parser.add_argument('--manual_seed', default=0, type=int, help='manual seed')

    # Device options
    parser.add_argument('--device', default='cuda:2', type=str,
                        help='device used for training')

    args = parser.parse_args()
    np.random.seed(seed = args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.backends.cudnn.deterministic=True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)


