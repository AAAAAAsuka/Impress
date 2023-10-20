from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
totensor = T.ToTensor()
topil = T.ToPILImage()


def recover_image(image, init_image, mask, background=False):
    image = totensor(image)
    mask = totensor(mask)
    init_image = totensor(init_image)
    if background:
        result = mask * init_image + (1 - mask) * image
    else:
        result = mask * image + (1 - mask) * init_image
    return topil(result)


def preprocess(image):
    w, h = image.size
    # w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    # image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image

def prepare_mask(mask):
    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    # mask = mask[None, None]
    # mask[mask < 0.5] = 0
    # mask[mask >= 0.5] = 1
    mask = topil(mask)
    mask = mask.convert("RGB")
    return mask

def prepare_image(image):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image[0]

def plot_all(init_image, adv_image, pur_image, image_nat, image_adv, image_pur, image_trans, prompt, seed, fname='pg_origin.jpg'):
    plt.figure()
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(50, 15))

    ax[0, 0].imshow(init_image)
    ax[1, 0].imshow(image_nat)
    ax[0, 1].imshow(adv_image)
    ax[1, 1].imshow(image_adv)
    ax[0, 2].imshow(pur_image)
    ax[1, 2].imshow(image_pur)
    ax[0, 3].imshow(image_trans)
    # ax[3].imshow(image_adv)

    ax[0, 0].set_title('Source Image', fontsize=16)
    ax[1, 0].set_title('Gen. Image Nat.', fontsize=16)
    ax[0, 1].set_title('Adv Image', fontsize=16)
    ax[1, 1].set_title('Gen. Image Adv.', fontsize=16)
    ax[0, 2].set_title('Pure Image', fontsize=16)
    ax[1, 2].set_title('Gen. Image Pure.', fontsize=16)
    ax[0, 3].set_title('trans Image', fontsize=16)
    ax[1, 3].set_title('', fontsize=16)

    for i in range(2):
        for j in range(4):
            ax[i,j].grid(False)
            ax[i,j].axis('off')

    fig.suptitle(f"Prompt: {prompt} | Seed:{seed}", fontsize=20)
    fig.savefig(fname=f'fig/{fname}')

def plot(init_image, adv_image, image_nat, image_adv, prompt, seed, fname='pg_origin.jpg'):
    plt.figure()
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 6))

    ax[0].imshow(init_image)
    ax[1].imshow(adv_image)
    ax[2].imshow(image_nat)
    ax[3].imshow(image_adv)
    # ax[4].imshow(image_trans)

    ax[0].set_title('Source Image', fontsize=16)
    ax[1].set_title('Adv Image', fontsize=16)
    ax[2].set_title('Gen. Image Nat.', fontsize=16)
    ax[3].set_title('Gen. Image Adv.', fontsize=16)
    # ax[4].set_title('trans image.', fontsize=16)

    for i in range(4):
        ax[i].grid(False)
        ax[i].axis('off')

    fig.suptitle(f"Prompt: {prompt} | Seed:{seed}", fontsize=20)
    fig.savefig(fname=f'fig/{fname}')