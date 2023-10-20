import torch.nn as nn
import torch
from tqdm import tqdm
import lpips


def impress(X_adv, model, eps=0.1, iters=40, clamp_min=0, clamp_max=1, lr=0.001, pur_alpha=0.5, noise=0.1):
    # init purified X
    X_p = X_adv.clone().detach()  + (torch.randn(*X_adv.shape) * noise).to(X_adv.device).half()
    pbar = tqdm(range(iters))
    criterion = nn.MSELoss()
    loss_fn_alex = lpips.LPIPS(net='vgg').to(X_adv.device)
    optimizer = torch.optim.Adam([X_p], lr=lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iters, eta_min=1e-5)
    for i in pbar:
        X_p.requires_grad_(True)
        _X_p = model(X_p).sample
        optimizer.zero_grad()
        lnorm_loss = criterion(_X_p, X_p)
        d = loss_fn_alex(X_p, X_adv)
        lpips_loss = max(d - eps, 0)
        loss = lnorm_loss + pur_alpha * lpips_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        X_p.data = torch.clamp(X_p, min=clamp_min, max=clamp_max)
        pbar.set_description(f"[Running purify]: Loss: {loss.item():.5f} | l2 dist: {lnorm_loss.item():.4} | lpips loss: {d.item():.4}")
    X_p.requires_grad_(False)
    return X_p
