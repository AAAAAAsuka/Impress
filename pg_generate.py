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
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline
import torchvision.transforms as T
import sys
from utils import preprocess, recover_image, plot, prepare_mask
import argparse
import re
import copy
topil = T.ToPILImage()

def get_image_after_vae(image, model, device):
    x = preprocess(image).to(device).half()
    vae_x = model(x).sample
    vae_x = (vae_x / 2 + 0.5).clamp(0, 1)
    vae_x_image = topil(vae_x[0]).convert("RGB")
    return vae_x_image

def test_image(image_dir, image_name, mask, save_dir, model, device, seed):
    image = Image.open(os.path.join(image_dir, image_name + '.png')).convert('RGB').resize((512,512))
    # cur_mask, cur_masked_image = prepare_mask_and_masked_image(image, mask)
    # image_pre = preprocess(image).to(device).half()

    np.random.seed(seed = seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    diff_image = model(prompt=args.prompt,
                     image=image,
                     mask_image=mask,
                     eta=1,
                     num_inference_steps=args.test_diff_steps,
                     guidance_scale=args.test_guidance,
                     # strength=strength
                     ).images[0]
    vae_image = get_image_after_vae(image, model.vae, device)
    # diff_image = topil(diff_image)
    diff_image.save(os.path.join(save_dir, image_name + '.png'))

    return image, diff_image, vae_image

def plot_test_image(clean_image, adv_image, pur_image, prompt, adv_dir, pur_dir, output_save_dir, image_name):
    plt.figure()
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    image_list = [clean_image, adv_image, pur_image]
    name_list = ['clean', 'adv', 'pur']
    for j, image in enumerate(image_list):
        ax[j].imshow(image)
        ax[j].set_title(f'{name_list[j]}', fontsize=16)
        ax[j].grid(False)
        ax[j].axis('off')
    fig.suptitle(f"generated data. Prompt: {prompt},\nadv: {adv_dir},\npur: {pur_dir}", fontsize=20)
    fig.savefig(fname=f'{output_save_dir}/{image_name}.png')

def plot_train_image(clean_image, adv_image, pur_image, clean_vae, adv_vae, pur_vae, prompt, adv_dir, pur_dir, output_save_dir, image_name):
    plt.figure()
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    image_list = [clean_image, adv_image, pur_image]
    vae_list = [clean_vae, adv_vae, pur_vae]
    all_list = [image_list, vae_list]
    
    name_list = ['clean', 'adv', 'pur']
    for i, img_list in enumerate(all_list):
        if i == 1: vae_flag = "vae_"
        else: vae_flag = ""
        
        for j, image in enumerate(img_list):
            ax[i, j].imshow(image)
            ax[i, j].set_title(f'{vae_flag}{name_list[j]}', fontsize=16)
            ax[i, j].grid(False)
            ax[i, j].axis('off')
    fig.suptitle(f"generated data. Prompt: {prompt},\nadv: {adv_dir},\npur: {pur_dir}", fontsize=20)
    fig.savefig(fname=f'{output_save_dir}/{image_name}.png')


def main(args):
    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        args.model,
        revision="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe_inpaint = pipe_inpaint.to(args.device)
    for name, param in pipe_inpaint.vae.named_parameters():
        param.requires_grad = False

    torch.manual_seed(args.manual_seed)

    # begin_index, end_index = args.parallel_index * 20, (args.parallel_index + 1) * 20


    adv_dir = f"../helen_face/adv_{args.attack_type}_eps{args.pg_eps}_step{args.pg_step_size}_iter{args.pg_iters}grad_reps{args.pg_grad_reps}_eta{args.pg_eta}_diff_steps{args.diff_steps}_guidance{args.guidance}_seed{args.manual_seed}"
    pur_dir = f"../helen_face/pur_eps{args.pur_eps}_pur_iters{args.pur_iters}_pur_lr{args.pur_lr}_pur_alpha{args.pur_alpha}_pur_noise{args.pur_noise}/"


    # begin_index, end_index = args.parallel_index * 20, (args.parallel_index + 1) * 20

    file_dir_list = sorted(os.listdir(pur_dir))
    if args.parallel_index >= 0:
        if len(file_dir_list) % 4 == 0:
            parallel_cut_step_size = len(file_dir_list) // 4
        else:
            parallel_cut_step_size = len(file_dir_list) // 4 + 1

        begin_index, end_index = args.parallel_index * parallel_cut_step_size, (args.parallel_index + 1) * parallel_cut_step_size
        file_dir_list = file_dir_list[begin_index: end_index]
    else:
        pass

    # file_dir_list = os.listdir(pur_dir)
    image_file_names = [os.path.basename(path)[:-4] for path in file_dir_list]

    save_dir_clean = re.sub("clean", "clean_diff", args.clean_dir)
    save_dir_adv = re.sub("adv", "adv_diff", adv_dir)
    save_dir_pur = re.sub("pur", "pur_diff", pur_dir)

    save_dir_test = f"../helen_face/result/{args.prompt}/test/pur_eps{args.pur_eps}_pur_iters{args.pur_iters}_pur_lr{args.pur_lr}_pur_alpha{args.pur_alpha}_pur_noise{args.pur_noise}_adv_{args.attack_type}_eps{args.pg_eps}_step{args.pg_step_size}_iter{args.pg_iters}grad_reps{args.pg_grad_reps}_eta{args.pg_eta}_diff_steps{args.diff_steps}_guidance{args.guidance}_seed{args.manual_seed}"
    save_dir_train = f"../helen_face/result/{args.prompt}/train/pur_eps{args.pur_eps}_pur_iters{args.pur_iters}_pur_lr{args.pur_lr}_pur_alpha{args.pur_alpha}_pur_noise{args.pur_noise}_adv_{args.attack_type}_eps{args.pg_eps}_step{args.pg_step_size}_iter{args.pg_iters}grad_reps{args.pg_grad_reps}_eta{args.pg_eta}_diff_steps{args.diff_steps}_guidance{args.guidance}_seed{args.manual_seed}"

    os.makedirs(save_dir_clean, exist_ok=True)
    os.makedirs(save_dir_adv, exist_ok=True)
    os.makedirs(save_dir_pur, exist_ok=True)
    os.makedirs(save_dir_test, exist_ok=True)
    os.makedirs(save_dir_train, exist_ok=True)

    for i, image_name in enumerate(image_file_names):
        # if os.path.exists(f"{save_dir_pur}/{image_name}.png"):
        #     print(f"{image_name} exists, skipping.")
        #     continue
        mask_image = Image.open(os.path.join(args.mask_dir, image_name + '.png')).convert('RGB').resize((512,512))
        mask_image = ImageOps.invert(mask_image)
        # mask_image = prepare_mask(mask_image)

        clean_image, diff_clean, vae_clean = test_image(args.clean_dir, image_name, mask_image, save_dir_clean, pipe_inpaint, args.device, seed=args.manual_seed)
        adv_image, diff_adv, vae_adv = test_image(adv_dir, image_name, mask_image, save_dir_adv, pipe_inpaint, args.device, seed=args.manual_seed)
        pur_image, diff_pur, vae_pur = test_image(pur_dir, image_name, mask_image, save_dir_pur, pipe_inpaint, args.device, seed=args.manual_seed)

        plot_test_image(diff_clean, diff_adv, diff_pur, args.prompt, adv_dir, pur_dir, save_dir_test, image_name)
        plot_train_image(clean_image, adv_image, pur_image, vae_clean, vae_adv, vae_pur, args.prompt, adv_dir, pur_dir, save_dir_train, image_name)



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
    parser.add_argument('--parallel_index', default=-1, type=int, help='pgd Hyperparameters')

    parser.add_argument('--guidance', default=7.5, type=float, help='learning rate.')
    parser.add_argument('--diff_steps', default=50, type=int, help='learning rate.')

    # pur Hyperparameters
    parser.add_argument('--neg_feed', type=float, default=-1.)
    parser.add_argument('--pur_eps', default=0.1, type=float, help='ae Hyperparameters')
    parser.add_argument('--pur_iters', default=100, type=int, help='ae Hyperparameters')
    parser.add_argument('--pur_lr', default=0.01, type=float, help='ae Hyperparameters')
    parser.add_argument('--pur_alpha', default=0.1, type=float, help='ae Hyperparameters')
    parser.add_argument('--pur_noise', default=0.1, type=float, help='ae Hyperparameters')

    # stable diffusion Hyperparameters
    parser.add_argument('--prompt', default="a person in an airplane", type=str, help='learning rate.')
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
    parser.add_argument('--device', default='cuda:0', type=str,
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


