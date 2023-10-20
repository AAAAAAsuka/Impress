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
from transformers import CLIPProcessor, CLIPModel
from utils import preprocess

def quantile_25_to_50_indices(tensor):
    q50 = torch.quantile(tensor, 0.5)
    q25 = torch.quantile(tensor, 0.25)
    indices = torch.nonzero((tensor >= q25) & (tensor <= q50)).squeeze()
    random_pick_index = np.random.choice(np.array(indices.cpu()), 1).item()
    return random_pick_index

def get_target_style(artist, model, processor, image_class, device):
    # clean_image_dir = f"../wikiart/preprocessed_data/{artist}/clean/train/"
    # probs_list = []
    # for curDir, dirs, files in os.walk(clean_image_dir):
    #     for file_name in files:
    #         if '.jsonl' in file_name: continue
    #         image = Image.open(f"{curDir}/{file_name}").convert("RGB")
    #         inputs = processor(text=image_class, images=image, return_tensors="pt", padding=True).to(device)
    #         outputs = model(**inputs)
    #         logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    #         probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    #         probs_list.append(probs)
    #         # class_index = probs.argmax(dim=1).item()
    #         # class_name = image_class[class_index]
    # prob_sum = torch.sum(torch.stack(probs_list), dim=(0,1))
    # # target_class_index = prob_sum.argmin(dim=0).item()
    # target_class_index = quantile_25_to_50_indices(prob_sum)
    # aim_style_prompt = image_class[target_class_index]
    aim_style_prompt = "Cubism by Picasso"
    # print(f"artist: {artist}, choose target style: ", aim_style_prompt, " with avg prob: ", prob_sum[target_class_index] / len(prob_sum))
    return aim_style_prompt


def style_transfer(args, pipe_img2img, artist, aim_style_prompt):
    clean_image_dir = f"../wikiart/preprocessed_data/{artist}/clean/train/"
    save_dir = f"../wikiart/preprocessed_data/{artist}/trans/train/transNum24_seed{args.manual_seed}/"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + f'/target_style_{aim_style_prompt}', exist_ok=True)
    # copy meta data
    shutil.copy(f"{clean_image_dir}/metadata.jsonl", f"{save_dir}/metadata.jsonl")

    with open(f"{clean_image_dir}/metadata.jsonl", "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            file_name = item['file_name']
            prompt = f"{aim_style_prompt} style"
            init_image = Image.open(f"{clean_image_dir}/{file_name}").convert("RGB")
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

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_class = ['Color_Field_Painting', 'Contemporary_Realism', 'Naive_Art_Primitivism', 'Post_Impressionism',
                   'Art_Nouveau_Modern', 'Analytical_Cubism', 'Impressionism', 'Expressionism', 'Action_painting',
                   'Early_Renaissance', 'Cubism', 'Mannerism_Late_Renaissance', 'Fauvism', 'Minimalism', 'New_Realism',
                   'Northern_Renaissance', 'Synthetic_Cubism', 'Rococo', 'Ukiyo_e', 'High_Renaissance', 'Symbolism',
                   'Pop_Art', 'Pointillism', 'Baroque', 'Romanticism', 'Realism', 'Abstract_Expressionism', 'Anime',
                   'Caricature', 'Cartoon', 'Picture Books', 'Comics', 'Commercial Art', 'Concept Art', 'Fantasy',
                   'Fashion', 'Fine art', 'Line art', 'Retro']
    for style in image_class:
        re.sub('_', ' ', style)

    artist_list = None
    for curDir, dirs, files in os.walk("../wikiart/preprocessed_data/"):
        if len(files) == 1:
            artist_list = dirs
            break

    for artist in artist_list:
        aim_style_prompt = get_target_style(artist, model, processor, image_class, device)
        style_transfer(args, pipe_img2img, artist, aim_style_prompt)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='diffusion attack')

    parser.add_argument('--model', default='stabilityai/stable-diffusion-2-1-base', type=str,
                        help='stable diffusion weight')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--train', default= 0, type=bool,
                        help='training(True) or testing(False)')


    # parser.add_argument('--aim_style', type=str, default='Oil painting by Van Gogh')
    parser.add_argument('--aim_style', type=str, default='Cubism by Picasso')

    # stable diffusion Hyperparameters
    parser.add_argument('--strength', default=0.5, type=float, help='learning rate.')
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
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='device used for training')

    args = parser.parse_args()
    np.random.seed(seed = args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.backends.cudnn.deterministic=True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)