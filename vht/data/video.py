from torch.utils.data import Dataset
from pathlib import Path

import json
import os
import numpy as np
import torch
import torchvision.transforms.functional as ttf
import PIL.Image as Image


def frame2id(frame_name):
    return int(frame_name.split("_")[-1])


def id2frame(frame_id):
    return f"frame_{frame_id:04d}"


def view2id(view_name):
    return int(view_name.split("_")[-1].split(".")[0])


def id2view(view_id):
    return f"image_{view_id:04d}.png"


class VideoDataset(Dataset):
    def __init__(self, path, scale_factor=1.0):
        """
        :param path: Path to dataset with the following directory layout
            root/
            |---frame_1/
            |   |---image_0000.png
            |   |---keypoints_static.npz
        :param scale_factor: frames input resolution will be scaled by that factor. landmarks
                    are scaled accordingly
        """

        super().__init__()
        self._path = Path(path)
        self._scale = scale_factor

        self._views = []
        frames = [f for f in os.listdir(self._path) if f.startswith("frame_")]
        for f in frames:
            self._views.append(self._path / f / "image_0000.png")

        self._views = sorted(self._views, key=lambda x: frame2id(x.parent.name))
        self.max_frame_id = max([frame2id(x.parent.name) for x in self._views])

    def __len__(self):
        return len(self._views)

    def __getitem__(self, i):
        """
        Get i-th sample from the dataset.
        """

        view = self._views[i]
        frame_path = view.parent
        sample = {}

        # subject and frame info
        sample["frame"] = frame2id(frame_path.name)

        rgba = ttf.to_tensor(Image.open(view).convert("RGBA"))
        H, W = rgba.shape[1:]
        target_H, target_W = int(H / self._scale), int(W / self._scale)
        rgba = ttf.resize(rgba, (target_H, target_W), antialias=True)
        sample["rgb"] = (rgba[:3] - 0.5) / 0.5

        # landmarks
        lmk_file = view.name.replace("image", "keypoints_static")
        lmk_file = lmk_file.replace(".png", ".json")
        path = frame_path / lmk_file

        with open(path, "r") as f:
            lmks_info = json.load(f)
            lmks_view = lmks_info["people"][0]["face_keypoints_2d"]
            lmks_iris = lmks_info["people"][0].get("iris_keypoints_2d", None)

        sample["lmk2d"] = (
            torch.from_numpy(np.array(lmks_view)).float()[:204].view(-1, 3)
        )
        # scale coordinates
        sample["lmk2d"][:, 0] *= target_W / W
        sample["lmk2d"][:, 1] *= target_H / H
        sample["lmk2d"][:, 2:] = 1.0

        if lmks_iris is not None:
            sample["lmk2d_iris"] = torch.from_numpy(np.array(lmks_iris)).float()[:204]
            sample["lmk2d_iris"] = sample["lmk2d_iris"].view(-1, 3)[[1, 0]]
            sample["lmk2d_iris"][:, 0] *= target_W / W
            sample["lmk2d_iris"][:, 1] *= target_H / H

        if lmks_iris is not None:
            if torch.sum(sample["lmk2d_iris"][:, :2] == -1) > 0:
                sample["lmk2d_iris"][:, 2:] = 0.0
            else:
                sample["lmk2d_iris"][:, 2:] = 1.0

        return sample

    @property
    def frame_list(self):
        frames = []
        for view in self._views:
            frames.append(frame2id(view.parent.name))
        return frames

    @property
    def view_list(self):
        views = []
        for view in self._views:
            views.append(view.name)
        return views
