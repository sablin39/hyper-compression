import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np
from dice_loss import dice_coeff
from utils import batch


def eval_net(net, dataset, lendata, device="cuda", batch_size=8, is_loss=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    criterion = nn.BCELoss()
    with torch.no_grad():
        for i, b in enumerate(batch(dataset, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs).to(device)
            true_masks = torch.from_numpy(true_masks).to(device)



            masks_pred = net(imgs)[0]

            if is_loss:
                loss = criterion(masks_pred, true_masks)
                tot += loss.item()
                # progress_bar.update(batch_size)
                # progress_bar.set_postfix(BCE=loss.item())
            else:
                masks_pred = (masks_pred > 0.5).float()
                dice = dice_coeff(masks_pred, true_masks).item()
                tot += dice
                # progress_bar.update(batch_size)
                # progress_bar.set_postfix(DICE=dice)

    value = tot / i
    return value
