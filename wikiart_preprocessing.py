from datasets import load_dataset
import csv
import re
import os
from PIL import Image, ImageOps
import torchvision.transforms as T
import jsonlines
from random import sample
import random
from transformers import pipeline
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
# dataset = load_dataset("fusing/wikiart_captions")

random.seed(0)
np.random.seed(seed=0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='diffusion attack')

parser.add_argument('--wikiart_dir', default="../wikiart/wikiart/", type=str,)
parser.add_argument('--exp_data_dir', default="../wikiart/preprocessed_data", type=str)
args = parser.parse_args()


def detect_style_match(artist, true_style_name, model, processor, image_class, file_dir_list_all):
    correct_num = 0
    file_dir_list = sample(file_dir_list_all, 124)
    for j, file_dir in enumerate(file_dir_list):
        image =  Image.open(f"{args.wikiart_dir}/{file_dir}").convert("RGB")
        inputs = processor(text=image_class, images=image, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        _, top3_class_index = torch.sort(probs, descending=True)
        top3_class_index = top3_class_index[0][:3]
        top3_class_name = [image_class[k] for k in top3_class_index]
        if true_style_name in top3_class_name:
            correct_num += 1
    correct_prob = correct_num / len(file_dir_list)
    print(f"artist: {artist}, style: {true_style_name}, correct prob: {correct_prob}")
    return correct_prob

def data_preprocess(file_dir_list, artist_name_text, output_dir, image_to_text):
    os.makedirs(output_dir, exist_ok=True)
    meta_data = []
    new_art_name_list = []
    stop_words_list = ["a painting of ",]
    for j, file_dir in enumerate(file_dir_list):
        # change art`s file name
        # file_name  = re.sub('.*/','',file_dir)
        # art_name  = re.sub('-',' ',file_name[len(artist_name)+1:])
        # new_file_name = f"{artist_name} {art_name}"
        # new_file_name = re.sub('jpg','png',new_file_name)

        # load art and preprocess
        art = Image.open(f"{args.wikiart_dir}/{file_dir}").convert("RGB")
        new_art_name = image_to_text(art)[0]['generated_text']
        for stop_words in stop_words_list:
            new_art_name = re.sub(stop_words, '', new_art_name)

        if new_art_name in new_art_name_list:
            new_art_name = new_art_name + '1'

        new_art_name_list.append(new_art_name)
        new_file_name = f'{new_art_name} by {artist_name_text}.png'
        resize = T.transforms.Resize(512)
        center_crop = T.transforms.CenterCrop(512)
        preprocessed_art = center_crop(resize(art))
        preprocessed_art.save(f"{output_dir}/{new_file_name}")
        # get text description and generate meta data
        # text = re.sub('[^A-Za-z\s-]*', '', f"{artist_name_text} {art_name}"[:-4]).strip(' ') # del number, sign , ".jpg" and extra black
        meta_data.append({'file_name': new_file_name, 'text': new_file_name[:-4]})
    with jsonlines.open(f'{output_dir}/metadata.jsonl', 'w') as writer:
        writer.write_all(meta_data)
    print(f"{artist_name_text} sampling done. output dir: {output_dir}")


def sampling_wikiart_artist(train_num = 24, test_num = 96, artist_num = 10):


    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    os.makedirs(f'{args.exp_data_dir}/', exist_ok=True)
    # [{'generated_text': 'a soccer game with a player jumping to catch the ball '}]

    csv_reader = csv.reader(open(f"{args.wikiart_dir}_csv/artist_class.txt"))
    artist_name_list, artist_name_prompt_list = [], []
    for row in csv_reader:
        artist_name = re.sub('[0-9]*', '', row[0]).strip(' ').lower()
        artist_name = re.sub('_', '-', artist_name)
        artist_name_list.append(artist_name)
        artist_name_prompt_list.append(re.sub('-', ' ', artist_name))

    # select art
    # counter = 0
    file_dir_list_all_style = []
    for i in range(len(artist_name_list)):
        print(f'selecting {artist_name_list[i]}')
        file_dir_list_all_style.append({})
        for curDir, dirs, files in os.walk(f"{args.wikiart_dir}/"):
            if len(files) == 0: continue
            curDir = re.sub(f'{args.wikiart_dir}/', '', curDir)
            file_dir_list_all_style[-1][curDir] = []
            for file in files:
                if artist_name_list[i] in file:
                    file_dir_list_all_style[-1][curDir].append(f"{curDir}/{file}")

    file_dir_list = []
    # style_list = []
    style_data = []
    for i, file_dir in enumerate(file_dir_list_all_style):
        print(f'artist: {artist_name_list[i]}')
        max_art_num = 0
        max_art_style = ''
        for key in file_dir:
            if len(file_dir[key]) == 0: continue
            # print(f"style: {key}, num: {len(file_dir[key])}")
            if len(file_dir[key]) > max_art_num:
                max_art_num = len(file_dir[key])
                max_art_style = key
        print(f'selected style: {max_art_style}, num: {max_art_num}')
        file_dir_list.append(file_dir[max_art_style])
        # style_list.append(max_art_style)
        style_data.append({'artist': artist_name_list[i], 'style': max_art_style})

    art_num_list = []
    for file_dir in file_dir_list:
        art_num_list.append(len(file_dir))


    choose_artist_index = sorted(range(len(art_num_list)), key=lambda k: art_num_list[k], reverse=True)# [:artist_num]

    file_dir_list = [file_dir_list[i] for i in choose_artist_index]
    artist_name_list = [artist_name_list[i] for i in choose_artist_index]
    artist_name_prompt_list = [artist_name_prompt_list[i] for i in choose_artist_index]
    style_data = [style_data[i] for i in choose_artist_index]

    # detect style prob
    normal_prob_list = []

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image_class = ['Color_Field_Painting', 'Contemporary_Realism', 'Naive_Art_Primitivism', 'Post_Impressionism',
                   'Art_Nouveau_Modern', 'Analytical_Cubism', 'Impressionism', 'Expressionism', 'Action_painting',
                   'Early_Renaissance', 'Cubism', 'Mannerism_Late_Renaissance', 'Fauvism', 'Minimalism', 'New_Realism',
                   'Northern_Renaissance', 'Synthetic_Cubism', 'Rococo', 'Ukiyo_e', 'High_Renaissance', 'Symbolism',
                   'Pop_Art', 'Pointillism', 'Baroque', 'Romanticism', 'Realism', 'Abstract_Expressionism', 'Anime',
                   'Caricature', 'Cartoon', 'Picture Books', 'Comics', 'Commercial Art', 'Concept Art', 'Fantasy',
                   'Fashion', 'Fine art', 'Line art', 'Retro']

    for i in range(len(artist_name_list)):
        normal_prob_list.append(detect_style_match(artist_name_list[i], style_data[i]["style"], model, processor, image_class, file_dir_list[i]))

    normal_prob_list = np.array(normal_prob_list)

    top_prob_index_list = np.argsort(normal_prob_list)[-artist_num:]

    file_dir_list = [file_dir_list[i] for i in top_prob_index_list]
    artist_name_list = [artist_name_list[i] for i in top_prob_index_list]
    artist_name_prompt_list = [artist_name_prompt_list[i] for i in top_prob_index_list]
    style_data = [style_data[i] for i in top_prob_index_list]
    normal_prob_list = [normal_prob_list[i] for i in top_prob_index_list]
    print(f"chosen artist: {artist_name_list}, prob: {normal_prob_list}, style: {style_data}")




    with jsonlines.open(f'{args.exp_data_dir}//style.jsonl', 'w') as writer:
        writer.write_all(style_data)



    # sampling
    for i in range(len(artist_name_list)):
        print(f'sampling {artist_name_list[i]}')
        if len(file_dir_list[i]) < train_num+test_num:
            print(f'not enough art for {artist_name_list[i]}')
            continue
        sample_dist = sample(file_dir_list[i], train_num+test_num)
        train_list = sample_dist[:train_num]
        test_list = sample_dist[train_num:] # i hate float
        # train_list = sample(file_dir_list, int(sample_num*split))
        print(f'selected training arts list: \n {train_list}')
        # test_list = sample(file_dir_list, int(sample_num*(1-split)))
        print(f'selected testing arts list: \n {test_list}')
        # meta_data = []
        output_dir = f'{args.exp_data_dir}/{artist_name_list[i]}/clean/train/'
        data_preprocess(train_list, artist_name_prompt_list[i], output_dir, image_to_text)
        output_dir = f'{args.exp_data_dir}/{artist_name_list[i]}/clean/test/'
        data_preprocess(test_list, artist_name_prompt_list[i], output_dir, image_to_text)

    print(artist_name_list)


# sampling_wikiart_artist(artist_name='claude-monet', artist_name_text = "Claude Monet", sample_num=30, split=0.8)
# sampling_wikiart_artist(artist_name='vincent-van-gogh', artist_name_text = "Vincent Van Gogh", sample_num=30, split=0.8)
# sampling_wikiart_artist(artist_name='kitagawa-utamaro', artist_name_text = "Kitagawa Utamaro", sample_num=30, split=0.8)
# sampling_wikiart_artist(artist_name='marjorie-strider', artist_name_text = "Marjorie Strider", sample_num=30, split=0.8)
sampling_wikiart_artist(train_num = 24, test_num = 100, artist_num = 8)
# sampling_wikiart_artist(artist_name='louise-elisabeth-vigee-le-brun', artist_name_text = "louise elisabeth vigee le brun", sample_num=30, split=0.8)
# ['pierre-auguste-renoir', 'claude-monet', 'nicholas-roerich', 'vincent-van-gogh', 'albrecht-durer', 'camille-pissarro', 'rembrandt', 'gustave-dore', 'marc-chagall', 'edgar-degas', 'ivan-aivazovsky', 'childe-hassam']