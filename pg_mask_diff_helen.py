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


def attack_forward(
        self,
        prompt: Union[str, List[str]],
        masked_image: Union[torch.FloatTensor, Image.Image],
        mask: Union[torch.FloatTensor, Image.Image],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,):
    text_inputs = self.tokenizer(
        prompt,
        padding="max_length",
        max_length=self.tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]

    uncond_tokens = [""]
    max_length = text_input_ids.shape[-1]
    uncond_input = self.tokenizer(
        uncond_tokens,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
    seq_len = uncond_embeddings.shape[1]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    text_embeddings = text_embeddings.detach()

    num_channels_latents = self.vae.config.latent_channels

    latents_shape = (1, num_channels_latents, height // 8, width // 8)
    latents = torch.randn(latents_shape, device=self.device, dtype=text_embeddings.dtype)

    mask = torch.nn.functional.interpolate(mask, size=(height // 8, width // 8))
    mask = torch.cat([mask] * 2)

    masked_image_latents = self.vae.encode(masked_image).latent_dist.sample()
    masked_image_latents = 0.18215 * masked_image_latents
    masked_image_latents = torch.cat([masked_image_latents] * 2)

    latents = latents * self.scheduler.init_noise_sigma

    self.scheduler.set_timesteps(num_inference_steps)
    timesteps_tensor = self.scheduler.timesteps.to(self.device)

    for i, t in enumerate(timesteps_tensor):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = self.scheduler.step(noise_pred, t, latents, eta=eta).prev_sample

    latents = 1 / 0.18215 * latents
    image = self.vae.decode(latents).sample
    return image


def compute_grad(diff_model, cur_mask, cur_masked_image, prompt, target_image, **kwargs):
    torch.set_grad_enabled(True)
    cur_mask = cur_mask.clone()
    cur_masked_image = cur_masked_image.clone()
    cur_mask.requires_grad = False
    cur_masked_image.requires_grad_()
    image_nat = attack_forward(diff_model, mask=cur_mask,
                               masked_image=cur_masked_image,
                               prompt=prompt,
                               **kwargs)

    loss = (image_nat - target_image).norm(p=2) + 10 * (cur_masked_image - image_nat).norm(p=2)
    grad = torch.autograd.grad(loss, [cur_masked_image])[0] * (1 - cur_mask)

    return grad, loss.item(), image_nat.data.cpu()


def super_l2(diff_model, cur_mask, X, prompt, step_size, iters, eps, clamp_min, clamp_max, grad_reps=5, target_image=0, **kwargs):
    X_adv = X.clone()
    iterator = tqdm(range(iters))
    for i in iterator:

        all_grads = []
        losses = []
        for i in range(grad_reps):
            c_grad, loss, last_image = compute_grad(diff_model, cur_mask, X_adv, prompt, target_image=target_image, **kwargs)
            all_grads.append(c_grad)
            losses.append(loss)
        grad = torch.stack(all_grads).mean(0)

        iterator.set_description_str(f'AVG Loss: {np.mean(losses):.3f}')

        l = len(X.shape) - 1
        grad_norm = torch.norm(grad.detach().reshape(grad.shape[0], -1), dim=1).view(-1, *([1] * l))
        grad_normalized = grad.detach() / (grad_norm + 1e-10)

        # actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        actual_step_size = step_size
        X_adv = X_adv - grad_normalized * actual_step_size

        d_x = X_adv - X.detach()
        d_x_norm = torch.renorm(d_x, p=2, dim=0, maxnorm=eps)
        X_adv.data = torch.clamp(X + d_x_norm, clamp_min, clamp_max)

    torch.cuda.empty_cache()

    return X_adv, last_image


def super_linf(diff_model, cur_mask, X, prompt, step_size, iters, eps, clamp_min, clamp_max, grad_reps=5, target_image=0, **kwargs):
    X_adv = X.clone()
    iterator = tqdm(range(iters))
    for i in iterator:

        all_grads = []
        losses = []
        for i in range(grad_reps):
            c_grad, loss, last_image = compute_grad(diff_model, cur_mask, X_adv, prompt, target_image=target_image, **kwargs)
            all_grads.append(c_grad)
            losses.append(loss)
        grad = torch.stack(all_grads).mean(0)

        iterator.set_description_str(f'AVG Loss: {np.mean(losses):.3f}')

        # actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        actual_step_size = step_size
        X_adv = X_adv - grad.detach().sign() * actual_step_size

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)

    torch.cuda.empty_cache()

    return X_adv, last_image

def main(args):
    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe_inpaint = pipe_inpaint.to(args.device)
    # for name, param in pipe_inpaint.vae.named_parameters():
    #     param.requires_grad = False
    # for name, param in pipe_inpaint.unet.named_parameters():
    #     param.requires_grad = False

    # init_image = Image.open(f'photoguard/assets/trevor_5.jpg').convert('RGB').resize((512,512))
    # mask_image = Image.open(f'photoguard/assets/trevor_5.tif').convert('RGB')
    # mask_image = ImageOps.invert(mask_image).resize((512,512))

    prompt = ""

    torch.manual_seed(args.manual_seed)
    target_url = "https://i.pinimg.com/originals/18/37/aa/1837aa6f2c357badf0f588916f3980bd.png"
    response = requests.get(target_url)
    target_image = Image.open(BytesIO(response.content)).convert("RGB")
    target_image = target_image.resize((512, 512))

    file_dir_list = sorted(os.listdir(args.clean_dir))
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

    adv_dir = f"../helen_face/adv_{args.attack_type}_eps{args.pg_eps}_step{args.pg_step_size}_iter{args.pg_iters}grad_reps{args.pg_grad_reps}_eta{args.pg_eta}_diff_steps{args.diff_steps}_guidance{args.guidance}_seed{args.manual_seed}"
    os.makedirs(adv_dir, exist_ok=True)

    for i, image_name in enumerate(image_file_names):
        if os.path.exists(f"{adv_dir}/{image_name}.png"):
            print(f"{image_name} exists, skipping.")
            continue
        init_image = Image.open(os.path.join(args.clean_dir, image_name + '.png')).convert('RGB').resize((512,512))
        mask_image = Image.open(os.path.join(args.mask_dir, image_name + '.png')).convert('RGB').resize((512,512))
        mask_image = ImageOps.invert(mask_image)
        cur_mask, cur_masked_image = prepare_mask_and_masked_image(init_image, mask_image)

        cur_mask = cur_mask.half().to(args.device)
        cur_masked_image = cur_masked_image.half().to(args.device)
        target_image_tensor = prepare_image(target_image)
        target_image_tensor = 0*target_image_tensor.to(args.device) # we can either attack towards a target image or simply the zero tensor

        if args.attack_type == "l2":
            result, last_image= super_l2(pipe_inpaint, cur_mask, cur_masked_image,
                              prompt=prompt,
                              target_image=target_image_tensor,
                              eps=args.pg_eps,
                              step_size=args.pg_step_size,
                              iters=args.pg_iters,
                              clamp_min = -1,
                              clamp_max = 1,
                              eta=args.pg_eta,
                              num_inference_steps=args.diff_steps,
                              guidance_scale=args.guidance,
                              grad_reps=args.pg_grad_reps
                             )
        elif args.attack_type == "linf":
            result, last_image= super_linf(pipe_inpaint, cur_mask, cur_masked_image,
                              prompt=prompt,
                              target_image=target_image_tensor,
                              eps=args.pg_eps,
                              step_size=args.pg_step_size,
                              iters=args.pg_iters,
                              clamp_min = -1,
                              clamp_max = 1,
                              eta=args.pg_eta,
                              num_inference_steps=args.diff_steps,
                              guidance_scale=args.guidance,
                              grad_reps=args.pg_grad_reps
                             )
        else:
            raise NameError

        adv_X = (result / 2 + 0.5).clamp(0, 1)
        adv_image = to_pil(adv_X[0]).convert("RGB")
        adv_image = recover_image(adv_image, init_image, mask_image, background=True)
        adv_image.save(f"{adv_dir}/{image_name}.png")



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