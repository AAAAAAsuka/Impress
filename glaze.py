import torch
from tqdm import tqdm
import lpips
import torch.nn as nn

def glaze(x, x_trans, model, p=0.1, alpha=0.1, iters=500, lr=0.002):
    # x_adv = x.clone().detach()  + (torch.rand(*x.shape) * 2 * p - p).to(x.device)
    delta = (torch.rand(*x.shape) * 2 * p - p).to(x.device)
    # input_var = nn.Parameter(torch.rand(*x.shape) * 2 * p - p, requires_grad=True).to(x.device)
    pbar = tqdm(range(iters))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([delta], lr=lr)
    loss_fn_alex = lpips.LPIPS(net='vgg').to(x.device)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iters, eta_min=0.001)
    for i in pbar:
        # x_adv_image = x.clone().detach()

        delta.requires_grad_(True)
        x_adv = x + delta
        x_adv.data = torch.clamp(x_adv, min=-1.0, max=1.0)
        x_emb = model(x_adv).latent_dist.sample()
        x_trans_emb = model(x_trans).latent_dist.sample()
        # x_trans_emb = model(x_trans).latent_dist.sample()
        optimizer.zero_grad()
        d = loss_fn_alex(x, x_adv)
        sim_loss = alpha * max(d-p, 0)
        loss = criterion(x_emb, x_trans_emb) + sim_loss
        # Getting gradients w.r.t. parameters
        loss.backward()
        # Updating parameters
        optimizer.step()
        # scheduler.step()
        pbar.set_description(f"[Running glaze]: Loss {loss.item():.5f} | sim loss {alpha * max(d.item()-p, 0):.5f} | dist {d.item():.5f}")
    x_adv = x + delta
    x_adv.data = torch.clamp(x_adv, min=-1.0, max=1.0)
    return x_adv


def glaze_adapt(x, x_trans, encoder, vae, p=0.1, alpha=0.1, iters=500, lr=0.002):
    # x_adv = x.clone().detach()  + (torch.rand(*x.shape) * 2 * p - p).to(x.device)
    delta = (torch.rand(*x.shape) * 2 * p - p).to(x.device)
    # input_var = nn.Parameter(torch.rand(*x.shape) * 2 * p - p, requires_grad=True).to(x.device)
    pbar = tqdm(range(iters))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([delta], lr=lr)
    loss_fn_alex = lpips.LPIPS(net='vgg').to(x.device)
    beta = 30
    for i in pbar:
        # x_adv_image = x.clone().detach()

        delta.requires_grad_(True)
        x_adv = x + delta
        x_adv.data = torch.clamp(x_adv, min=-1.0, max=1.0)
        x_emb = encoder(x_adv).latent_dist.sample()
        x_trans_emb = encoder(x_trans).latent_dist.sample()
        # x_trans_emb = model(x_trans).latent_dist.sample()
        optimizer.zero_grad()
        d = loss_fn_alex(x, x_adv)
        sim_loss = alpha * max(d-p, 0)

        _x_adv = vae(x_adv).sample
        consist_loss = criterion(_x_adv, x_adv)
        loss = criterion(x_emb, x_trans_emb) + sim_loss + beta * consist_loss

        # Getting gradients w.r.t. parameters
        loss.backward()
        # Updating parameters
        optimizer.step()
        pbar.set_description(f"[Running glaze]: Loss {loss.item():.5f} | sim loss {alpha * max(d.item()-p, 0):.5f} | dist {d.item():.5f}| consist loss {consist_loss.item():.5f}")
    x_adv = x + delta
    x_adv.data = torch.clamp(x_adv, min=-1.0, max=1.0)
    return x_adv






