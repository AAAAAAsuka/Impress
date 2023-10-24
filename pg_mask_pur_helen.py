import os
from PIL import Image, ImageOps
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import requests
from tqdm import tqdm
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
import torchvision.transforms as T
from typing import Union, List, Optional, Callable
import argparse
from utils import preprocess, prepare_mask_and_masked_image, recover_image, prepare_image
to_pil = T.ToPILImage()

from impress import impress

def main(args):
    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        args.model,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe_inpaint = pipe_inpaint.to(args.device)
    for name, param in pipe_inpaint.vae.named_parameters():
        param.requires_grad = False

    torch.manual_seed(args.manual_seed)


    adv_dir = f"../helen_face/adapt_adv_{args.attack_type}_eps{args.pg_eps}_step{args.pg_step_size}_iter{args.pg_iters}grad_reps{args.pg_grad_reps}_eta{args.pg_eta}_diff_steps{args.diff_steps}_guidance{args.guidance}_seed{args.manual_seed}"
    save_dir_pur = f"../helen_face/pur_eps{args.pur_eps}_pur_iters{args.pur_iters}_pur_lr{args.pur_lr}_pur_alpha{args.pur_alpha}_pur_noise{args.pur_noise}/"
    os.makedirs(save_dir_pur, exist_ok=True)

    file_dir_list = sorted(os.listdir(adv_dir))
    if args.parallel_index >= 0:
        if len(file_dir_list) % 4 == 0:
            parallel_cut_step_size = len(file_dir_list) // 4
        else:
            parallel_cut_step_size = len(file_dir_list) // 4 + 1

        begin_index, end_index = args.parallel_index * parallel_cut_step_size, (args.parallel_index + 1) * parallel_cut_step_size
        file_dir_list = file_dir_list[begin_index: end_index]
    else:
        pass

    image_file_names = [os.path.basename(path)[:-4] for path in file_dir_list]

    for i, image_name in enumerate(image_file_names):

        adv_image = Image.open(os.path.join(adv_dir, image_name + '.png')).convert('RGB').resize((512, 512))
        x_adv = preprocess(adv_image).to(args.device).half()

        x_purified = impress(x_adv,
                             model=pipe_inpaint.vae,
                             clamp_min=-1,
                             clamp_max=1,
                             eps=args.pur_eps,
                             iters=args.pur_iters,
                             lr=args.pur_lr,
                             pur_alpha=args.pur_alpha,
                             noise=args.pur_noise, )
            # convert pixels back to [0,1] range
        x_purified = (x_purified / 2 + 0.5).clamp(0, 1)
        purified_image = to_pil(x_purified[0]).convert("RGB")
        purified_image.save(f"{save_dir_pur}/{image_name}.png")




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='diffusion attack')

    parser.add_argument('--model', default="runwayml/stable-diffusion-inpainting", type=str,
                        help='stable diffusion weight')

    # data
    parser.add_argument('--clean_dir', default="../helen_face/clean", type=str, help='l2 linf')
    parser.add_argument('--adv_dir', default="../helen_face/adv", type=str, help='l2 linf')

    # pgd Hyperparameters
    parser.add_argument('--attack_type', default="l2", type=str, help='l2 linf')
    parser.add_argument('--pg_eps', default=16, type=int, help='pgd Hyperparameters')
    parser.add_argument('--pg_step_size', default=1, type=int, help='pgd Hyperparameters')
    parser.add_argument('--pg_iters', default=200, type=int, help='pgd Hyperparameters')
    parser.add_argument('--pg_grad_reps', default=10, type=int, help='pgd Hyperparameters')
    parser.add_argument('--pg_eta', default=1, type=int, help='pgd Hyperparameters')
    parser.add_argument('--parallel_index', default=-1, type=int, help='pgd Hyperparameters')

    # pur Hyperparameters
    parser.add_argument('--neg_feed', type=float, default=-1.)
    parser.add_argument('--pur_eps', default=0.1, type=float, help='ae Hyperparameters')
    parser.add_argument('--pur_iters', default=100, type=int, help='ae Hyperparameters')
    parser.add_argument('--pur_lr', default=0.01, type=float, help='ae Hyperparameters')
    parser.add_argument('--pur_alpha', default=0.1, type=float, help='ae Hyperparameters')
    parser.add_argument('--pur_noise', default=0.1, type=float, help='ae Hyperparameters')

    # stable diffusion Hyperparameters
    # parser.add_argument('--strength', default=0.5, type=float, help='learning rate.')
    parser.add_argument('--guidance', default=7.5, type=float, help='learning rate.')
    parser.add_argument('--diff_steps', default=4, type=int, help='learning rate.')

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
    sys.path.append("..")
    main(args)