import torch
import torch.nn as nn
import torchvision.models as models
from transformers import CLIPProcessor, CLIPModel
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
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline
import torchvision.transforms as T
import sys
from utils import preprocess, recover_image, plot
import argparse
import jsonlines
from glaze import glaze
import re
import copy
from impress import autoencoder
topil = T.ToPILImage()
# from ignite.metrics import FID
from sewar.full_ref import vifp, ssim, psnr
# from image_similarity_measures.quality_metrics import fsim

def image_quality_metrics(clean_image, diff_image, score_dict):
    # return FID, PR, SSIM, PSNR, VIFp, FSIM
    ssim_score = ssim(clean_image, diff_image)[0]
    psnr_score = psnr(clean_image, diff_image)
    vifp_score = vifp(clean_image, diff_image)
    # clean_image, diff_image = clean_image/255.0, diff_image/255.0
    # fsim_score = fsim(clean_image, diff_image)
    fsim_score = 0.
    score_dict["ssim"].append(ssim_score)
    score_dict["psnr"].append(psnr_score)
    score_dict["vifp"].append(vifp_score)
    score_dict["fsim"].append(fsim_score)

def load_image(file_dir, file_name):
    image = Image.open(f"{file_dir}/{file_name}").convert("RGB").resize((512, 512))
    image_np = np.asarray(image)
    return image_np


def main(args):
    adv_dir = f"../helen_face/adv_{args.attack_type}_eps{args.pg_eps}_step{args.pg_step_size}_iter{args.pg_iters}grad_reps{args.pg_grad_reps}_eta{args.pg_eta}_diff_steps{args.diff_steps}_guidance{args.guidance}_seed{args.manual_seed}"
    pur_dir = f"../helen_face/pur_eps{args.pur_eps}_pur_iters{args.pur_iters}_pur_lr{args.pur_lr}_pur_alpha{args.pur_alpha}_pur_noise{args.pur_noise}/"

    diff_dir_clean = re.sub("clean", "clean_diff", copy.deepcopy(args.clean_dir))
    diff_dir_adv = re.sub("adv", "adv_diff", copy.deepcopy(adv_dir))
    diff_dir_pur = re.sub("pur", "pur_diff", copy.deepcopy(pur_dir))

    file_dir_list = os.listdir(diff_dir_pur)
    image_file_names = [os.path.basename(path) for path in file_dir_list]

    adv_score_dict = {"ssim": [], "psnr": [], "vifp": [], "fsim": []}
    pur_score_dict = copy.deepcopy(adv_score_dict)
    clean_score_dict = copy.deepcopy(adv_score_dict)

    image_file_names = tqdm(image_file_names)
    for image_name in image_file_names:
        # true_image = load_image(args.clean_dir, image_name)
        clean_image = load_image(diff_dir_clean, image_name)
        adv_image = load_image(diff_dir_adv, image_name)
        pur_image = load_image(diff_dir_pur, image_name)


        # image_quality_metrics(true_image, clean_image, clean_score_dict)
        image_quality_metrics(clean_image, adv_image, adv_score_dict)
        image_quality_metrics(clean_image, pur_image, pur_score_dict)

    for key in adv_score_dict.keys():
        # clean_score_dict[key] = [sum(clean_score_dict[key]) / len(clean_score_dict[key]),
        #                          1.96 * torch.std(torch.tensor(clean_score_dict[key])).item() / len(clean_score_dict[key])]

        adv_score_dict[key] = [sum(adv_score_dict[key]) / len(adv_score_dict[key]),
                               torch.std(torch.tensor(adv_score_dict[key])).item() ]

        pur_score_dict[key] = [sum(pur_score_dict[key]) / len(pur_score_dict[key]),
                               torch.std(torch.tensor(pur_score_dict[key])).item() ]

    # print(f"test_num: {len(file_list)}")
    # print(f"clean: {clean_score_dict}")
    print(f"adv: {adv_score_dict}")
    print(f"pur: {pur_score_dict}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='diffusion attack')

    parser.add_argument('--model', default="runwayml/stable-diffusion-inpainting", type=str,
                        help='stable diffusion weight')

    # data
    parser.add_argument('--clean_dir', default="../helen_face/clean", type=str, help='l2 linf')
    parser.add_argument('--mask_dir', default="../helen_face/mask", type=str, help='l2 linf')
    # pgd Hyperparameters
    parser.add_argument('--attack_type', default="l2", type=str, help='l2 linf')
    parser.add_argument('--pg_eps', default=16, type=int, help='pgd Hyperparameters')
    parser.add_argument('--pg_step_size', default=1, type=int, help='pgd Hyperparameters')
    parser.add_argument('--pg_iters', default=200, type=int, help='pgd Hyperparameters')
    parser.add_argument('--pg_grad_reps', default=10, type=int, help='pgd Hyperparameters')
    parser.add_argument('--pg_eta', default=1, type=int, help='pgd Hyperparameters')
    parser.add_argument('--parallel_index', default=0, type=int, help='pgd Hyperparameters')

    parser.add_argument('--guidance', default=7.5, type=float, help='learning rate.')
    parser.add_argument('--diff_steps', default=4, type=int, help='learning rate.')

    # pur Hyperparameters
    parser.add_argument('--neg_feed', type=float, default=-1.)
    parser.add_argument('--pur_eps', default=0.1, type=float, help='ae Hyperparameters')
    parser.add_argument('--pur_iters', default=100, type=int, help='ae Hyperparameters')
    parser.add_argument('--pur_lr', default=0.01, type=float, help='ae Hyperparameters')
    parser.add_argument('--pur_alpha', default=0.1, type=float, help='ae Hyperparameters')
    parser.add_argument('--pur_noise', default=0.1, type=float, help='ae Hyperparameters')

    # stable diffusion Hyperparameters
    parser.add_argument('--prompt', default="A person on a plane", type=str, help='learning rate.')
    parser.add_argument('--test_guidance', default=7.5, type=float, help='learning rate.')
    parser.add_argument('--test_diff_steps', default=100, type=int, help='learning rate.')

    # Aim Model Hyperparameters
    parser.add_argument('--batch_size', default=16, type=int, help='batch size.')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate.')
    parser.add_argument('--epoch', default=50, type=int, help='training epoch.')
    # parser.add_argument('--norm', default=False, type=bool, help='normalize or not.')

    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='./ckpt', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    # parser.add_argument('--model_name', default='/cnn_mnist.pth', type=str,
    #                     help='network structure choice')
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
    # add "../"
    sys.path.append("..")
    main(args)


