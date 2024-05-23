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
import shutil

def main(args):
    # make sure you're logged in with `huggingface-cli login` - check https://github.com/huggingface/diffusers for more details
    to_pil = T.ToPILImage()
    pipe_img2img = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32)
    pipe_img2img.scheduler = DPMSolverMultistepScheduler.from_config(pipe_img2img.scheduler.config)
    pipe_img2img = pipe_img2img.to(device)
    # pipe_img2img.enable_xformers_memory_efficient_attention()
    for name, param in pipe_img2img.vae.encoder.named_parameters():
        param.requires_grad = False

    save_dir_adv = re.sub('/trans/', f'/adv_p{args.p}_alpha{args.alpha}_iter{args.glaze_iters}_lr{args.lr}/', copy.deepcopy(args.trans_data_dir))
    os.makedirs(save_dir_adv, exist_ok=True)
    shutil.copy(f"{args.clean_data_dir}/metadata.jsonl", f"{save_dir_adv}/metadata.jsonl")


    with open(f"{args.clean_data_dir}/metadata.jsonl", "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            file_name = item['file_name']
            prompt = item['text']
            init_image = Image.open(f"{args.clean_data_dir}/{file_name}").convert("RGB")
            trans_image = Image.open(f"{args.trans_data_dir}/{file_name}").convert("RGB")

            x = preprocess(init_image).to(device)
            x_t = preprocess(trans_image).to(device)
            # x = torch.squeeze(x)
            x_adv = glaze(x, x_t, model=pipe_img2img.vae.encode,
                              p=args.p, alpha=args.alpha, iters=args.glaze_iters, lr=args.lr)


            x_adv = (x_adv / 2 + 0.5).clamp(0, 1)
            adv_image = to_pil(x_adv[0]).convert("RGB")
            adv_image.save(f"{save_dir_adv}/{file_name}")




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='diffusion attack')

    # model_id_or_path = "runwayml/stable-diffusion-v1-5"
    # model_id_or_path = "CompVis/stable-diffusion-v1-4"
    # model_id_or_path = "CompVis/stable-diffusion-v1-3"
    # model_id_or_path = "CompVis/stable-diffusion-v1-2"
    # model_id_or_path = "CompVis/stable-diffusion-v1-1"
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


    # pgd Hyperparameters
    parser.add_argument('--p', default=0.05, type=float, help='pgd Hyperparameters')
    parser.add_argument('--alpha', default=30, type=int, help='pgd Hyperparameters')
    parser.add_argument('--glaze_iters', default=500, type=int, help='pgd Hyperparameters')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate.')


    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='./ckpt', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    # parser.add_argument('--model_name', default='/cnn_mnist.pth', type=str,
    #                     help='network structure choice')
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


