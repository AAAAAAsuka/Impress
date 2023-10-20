import torch
import torch.nn as nn
import torchvision.models as models

import os
from PIL import Image, ImageOps
import requests
import matplotlib.pyplot as plt
import numpy as np
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

from transformers import CLIPProcessor, CLIPModel
import shutil

to_pil = T.ToPILImage()

def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

def get_clip_results(image_dir, file_name, image_class, model, processor, device):
    images = Image.open(f"{image_dir}/{file_name}").convert("RGB")
    inputs = processor(text=image_class, images=images, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

    _, top3_class_index = torch.sort(probs, descending=True)
    top3_class_index = top3_class_index[0][:3]
    top3_class_name = [image_class[i] for i in top3_class_index]
    # class_index = probs.argmax(dim=1).item()
    # class_name = image_class[class_index]
    return probs, top3_class_name





def main(args):
    image_class = ['Color_Field_Painting', 'Contemporary_Realism', 'Naive_Art_Primitivism', 'Post_Impressionism',
                   'Art_Nouveau_Modern', 'Analytical_Cubism', 'Impressionism', 'Expressionism', 'Action_painting',
                   'Early_Renaissance', 'Cubism', 'Mannerism_Late_Renaissance', 'Fauvism', 'Minimalism', 'New_Realism',
                   'Northern_Renaissance', 'Synthetic_Cubism', 'Rococo', 'Ukiyo_e', 'High_Renaissance', 'Symbolism',
                   'Pop_Art', 'Pointillism', 'Baroque', 'Romanticism', 'Realism', 'Abstract_Expressionism', 'Anime',
                   'Caricature', 'Cartoon', 'Picture Books', 'Comics', 'Commercial Art', 'Concept Art', 'Fantasy',
                   'Fashion', 'Fine art', 'Line art', 'Retro']
    for style in image_class:
        re.sub('_', ' ', style)

    all_artist = args.all_artists.split()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clean_acc_all, glaze_acc_all, pur_acc_all = [], [], []
    true_label_dir = f"../wikiart/preprocessed_data/style.jsonl"
    style_data = read_jsonl(true_label_dir)

    for artist in all_artist:
        args.artist = artist
        args.meta_data_dir = f"../wikiart/preprocessed_data/{args.artist}/clean/test/"
        args.clean_data_dir = f"../wikiart/preprocessed_data/{args.artist}/clean/test/{args.ft_step}/"
        args.adv_data_dir = f"../wikiart/preprocessed_data/{args.artist}/{args.adv_para}/test/{args.ft_step}_transNum{args.trans_num}_seed{args.manual_seed}/"
        args.pur_data_dir = f"../wikiart/preprocessed_data/{args.artist}/{args.pur_para}/test/{args.ft_step}_transNum{args.trans_num}_seed{args.manual_seed}/"
        clean_acc, adv_acc, pur_acc = 0, 0, 0
        for style in style_data:
            if style['artist'] == args.artist:
                true_style = style['style']
                break
        pur_dist_asr = 0
        with open(f"{args.meta_data_dir}/metadata.jsonl", "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                file_name = item['file_name']
                prompt = item['text']
                clean_prob, clean_class = get_clip_results(args.clean_data_dir, file_name, image_class, model, processor, device)
                adv_prob, adv_class = get_clip_results(args.adv_data_dir, file_name, image_class, model, processor, device)
                pur_prob, pur_class = get_clip_results(args.pur_data_dir, file_name, image_class, model, processor, device)
                # clean_prob_list.append(clean_prob)
                # adv_prob_list.append(adv_prob)
                # pur_prob_list.append(pur_prob)

                if true_style in pur_class:
                    pur_acc += 1
                if true_style  in adv_class:
                    adv_acc += 1
                if true_style in clean_class:
                    clean_acc += 1
                if torch.norm(clean_prob - pur_prob).item() < torch.norm(clean_prob - adv_prob).item():
                    pur_dist_asr += 1

        print("artist: ", args.artist)
        print(f"clean acc: {clean_acc/100}, adv_acc: {adv_acc/100}, pur_acc: {pur_acc/100}, pur_dist_asr: {pur_dist_asr/100}")
        clean_acc_all.append(clean_acc/100)
        glaze_acc_all.append(adv_acc/100)
        pur_acc_all.append(pur_acc/100)

    print(f"ALL: clean acc: {np.mean(clean_acc_all)}, adv_acc: {np.mean(glaze_acc_all)}, pur_acc: {np.mean(pur_acc_all)}")




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


    parser.add_argument('--meta_data_dir', type=str, default='../wikiart/preprocessed_data/kitagawa-utamaro/clean/test/')
    parser.add_argument('--clean_data_dir', type=str, default='../wikiart/preprocessed_data/kitagawa-utamaro/clean/test/')
    parser.add_argument('--adv_data_dir', type=str, default='../wikiart/preprocessed_data/kitagawa-utamaro/adv/test/')
    parser.add_argument('--pur_data_dir', type=str, default='../wikiart/preprocessed_data/kitagawa-utamaro/pur/test/')
    parser.add_argument('--all_artists', type=str, default='None')

    parser.add_argument('--adv_para', type=str, default='../wikiart/preprocessed_data/kitagawa-utamaro/adv/train/adv_')
    parser.add_argument('--pur_para', type=str, default='../wikiart/preprocessed_data/kitagawa-utamaro/pur/train/pur_')
    parser.add_argument('--ft_step', type=str, default='0')
    parser.add_argument('--trans_num', type=str, default='24')
    parser.add_argument('--manual_seed', default=0, type=int, help='manual seed')

    # Device options
    parser.add_argument('--device', default='cuda:1', type=str,
                        help='device used for training')

    args = parser.parse_args()
    np.random.seed(seed = args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.backends.cudnn.deterministic=True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)


