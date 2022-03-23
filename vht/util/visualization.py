import torch
import torch.nn.functional as F

import matplotlib


@torch.no_grad()
def plot_2Dlandmarks(img_tensor, lmks):
    assert lmks.shape[0] == img_tensor.shape[0]

    B, C, H, W = img_tensor.shape
    NUM_LMKS = lmks.shape[1]
    kernel = torch.tensor([[0.25, 0.5, 0.25]]).to(img_tensor.device)
    circles = torch.zeros((B, NUM_LMKS, H, W)).to(img_tensor.device)
    masks = (
        (lmks[:, :, 0] < W)
        & (lmks[:, :, 1] < H)
        & (lmks[:, :, 0] >= 0)
        & (lmks[:, :, 1] >= 0)
    )
    lmks = lmks.long()

    # create scattered dots
    for i, (img_i, lmks_i, mask_i) in enumerate(zip(img_tensor, lmks, masks)):
        circles[i, mask_i, lmks_i[:, 1][mask_i], lmks_i[:, 0][mask_i]] = 1

    # increase dot size
    circles = F.conv2d(
        circles,
        kernel[None, None, ...].expand(NUM_LMKS, 1, 1, 3),
        padding=(0, 1),
        groups=NUM_LMKS,
    )
    circles = F.conv2d(
        circles,
        kernel.T[None, None, ...].expand(NUM_LMKS, 1, 3, 1),
        padding=(1, 0),
        groups=NUM_LMKS,
    )

    # make resulting circles solid and paste them onto image tensor
    cmap = matplotlib.cm.get_cmap("Spectral")
    for lmk_i in range(NUM_LMKS):
        # create a color
        color = cmap(lmk_i / NUM_LMKS)[:3]
        color = torch.tensor(color).to(img_tensor.device).view(1, 3, 1, 1)
        color = color.expand(img_tensor.shape).float()

        lmk_circle_mask = circles[:, [lmk_i], :, :].expand(B, C, H, W) > 0
        img_tensor[lmk_circle_mask] = color[lmk_circle_mask]

    return img_tensor
