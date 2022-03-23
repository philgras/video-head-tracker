from vht.model.flame import FlameHead, FlameTex
from vht.util.general import blur_tensors, DecayScheduler
from vht.util.graphics import (
    calibrate_camera,
    calibrate_extrinsics,
    batch_rodrigues,
    normalize_image_points,
)
from vht.util.render import SHRenderer, create_camera_objects
from vht.util.log import get_logger
from vht.util.visualization import plot_2Dlandmarks

from torch.utils.tensorboard import SummaryWriter
from pytorch3d.structures.meshes import Meshes
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import io
import os
from matplotlib.animation import FuncAnimation

logger = get_logger(__name__)


FLAME_LOWER_NECK_FACES_PATH = "assets/flame/lower_neck_face_idcs.npy"


def _to_batch(x, indices):
    return torch.stack([x[i] for i in indices])


class FlameTracker:
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        argparse_fields = FlameTracker._get_argparse_fields()
        for f in argparse_fields:
            parser.add_argument(f.pop("name_or_flags"), **f)

        return parser

    @staticmethod
    def _get_argparse_fields():
        """
        Returns the command line argparse fields as list of dicts. Seperated from get_argparse_args()
        so that can be manipulated before adding to argparser in combined models that may have same arguments
        :return:
        """
        fields = [
            # flags
            dict(name_or_flags="--calibrated", action="store_true"),
            dict(name_or_flags="--load_tracked_flame_params", type=Path, default=None),
            # learning rates
            dict(name_or_flags="--lr", default=0.005, type=float),
            dict(name_or_flags="--pos_lr", default=0.001, type=float),
            dict(name_or_flags="--cam_lr", default=0.005, type=float),
            dict(name_or_flags="--light_lr", default=0.005, type=float),
            # energy weights
            dict(name_or_flags="--w_lmk", default=5, type=float),
            dict(name_or_flags="--w_photo", default=30, type=float),
            dict(name_or_flags="--w_shape_reg", default=1e-3, type=float),
            dict(name_or_flags="--w_expr_reg", default=1e-3, type=float),
            dict(name_or_flags="--w_tex_reg", default=1e-3, type=float),
            dict(name_or_flags="--w_pos_trans", default=1e1, type=float),
            dict(name_or_flags="--w_pos_glob", default=1e-2, type=float),
            dict(name_or_flags="--w_pos_neck", default=1e-2, type=float),
            dict(name_or_flags="--w_pos_jaw", default=1e-4, type=float),
            dict(name_or_flags="--w_pos_eyes", default=1e-4, type=float),
            dict(name_or_flags="--w_eyes_sym", default=10, type=float),
            dict(name_or_flags="--w_eyes_lmk", default=2, type=float),
            # decays
            dict(name_or_flags="--blur_sigma", default=[10, 1e-4], type=float, nargs=2),
            # rendering
            dict(name_or_flags="--sampling_scale", default=4, type=float),
            # flame params
            dict(name_or_flags="--n_shape", default=300, type=int),
            dict(name_or_flags="--n_expr", default=100, type=int),
            dict(name_or_flags="--n_tex", default=100, type=int),
            dict(name_or_flags="--tex_res", default=512, type=int),
            dict(name_or_flags="--ignore_lower_neck", action="store_true"),
            # steps
            dict(name_or_flags="--steps_per_frame", type=int, default=10),
            dict(name_or_flags="--sub_steps", type=int, default=5),
            dict(name_or_flags="--init_steps", type=int, default=400),
            # logging
            dict(name_or_flags="--img_log_freq", type=int, default=50),
            dict(name_or_flags="--energy_log_freq", type=int, default=10),
            # misc
            dict(name_or_flags="--output_path", type=str, required=True),
            dict(name_or_flags="--save_period", type=int, required=False, default=1),
            dict(name_or_flags="--device", type=str, required=True),
            dict(name_or_flags="--keyframes", type=int, nargs="*", default=()),
            dict(name_or_flags="--cutframes", type=int, nargs="*", default=()),
            dict(name_or_flags="--frame_rate", type=int, default=30),
        ]

        return fields

    def __init__(self, dataset, **kwargs):

        self._config = kwargs

        n_frames = len(dataset)

        n_shape = self._config["n_shape"]
        n_expr = self._config["n_expr"]
        n_tex = self._config["n_tex"]

        device = self._config["device"]

        self._n_frames = n_frames
        self._calibrated = self._config["calibrated"]
        self._dataset = dataset
        self._device = device

        train_tensors = []

        self._flame = FlameHead(n_shape, n_expr)
        self._flame.to(device)

        self._flame_tex = FlameTex(n_tex)
        self._flame_tex.to(device)

        if self._config["ignore_lower_neck"]:
            self._face_mask = torch.ones(len(self._flame.faces), device=device).bool()
            lower_neck_ids = np.load(FLAME_LOWER_NECK_FACES_PATH)
            self._face_mask[lower_neck_ids] = 0

        # flame model params
        self._shape = torch.zeros(n_shape).to(device)
        self._expr = [torch.zeros(n_expr).to(device) for _ in range(n_frames)]

        # joint poses
        neutral = self._flame.get_neutral_joint_rotations()
        self._neck_pose = [neutral["neck"].detach().clone() for _ in range(n_frames)]
        self._jaw_pose = [neutral["jaw"].detach().clone() for _ in range(n_frames)]
        neutral_eyes = torch.cat([neutral["eyes"], neutral["eyes"]])
        self._eyes_pose = [neutral_eyes.detach().clone() for _ in range(n_frames)]

        # rigid pose
        self._translation = [torch.zeros(3).to(device) for _ in range(n_frames)]
        self._rotation = [torch.zeros(3).to(device) for _ in range(n_frames)]

        # texture and lighting params
        self._texture = torch.zeros(n_tex).to(device)
        self._lights = torch.zeros(9, 3).to(device)
        # initialize with uniform lighting
        self._lights[0] = (
            torch.tensor([np.sqrt(4 * np.pi)]).expand(3).float().to(device)
        )

        train_tensors += (
            [self._shape, self._texture, self._lights]
            + self._expr
            + self._neck_pose
            + self._jaw_pose
            + self._translation
            + self._rotation
            + self._eyes_pose
        )

        # camera definition
        if not self._calibrated:
            # K contains focal length and principle point
            self._K = torch.zeros(3).to(device)
            self._RT = torch.eye(3, 4).to(device)
            train_tensors += [self._K]

        for t in train_tensors:
            t.requires_grad = True

        # renderer for visualization, dense photometric energy
        self._render = SHRenderer()
        self._image_size = dataset[0]["rgb"].shape[-2:]

        # decays for different quantities
        def make_decay(key, is_init, geometric):
            steps = (
                self._config["init_steps"]
                if is_init
                else self._config["steps_per_frame"]
            )
            return DecayScheduler(*self._config[key], 0, steps, geometric)

        self._decays = {
            "blur_sigma_init": make_decay("blur_sigma", True, True),
            "blur_sigma": make_decay("blur_sigma", False, True),
        }

        # tensorboard logger
        out_dir = Path(self._config["output_path"])
        if not out_dir.exists():
            os.makedirs(out_dir)

        version = 0
        for fname in out_dir.iterdir():
            if fname.is_dir() and fname.name.startswith("tracking"):
                num = int(fname.name.split("_")[-1])
                if num >= version:
                    version = num + 1
        out_dir = out_dir / f"tracking_{version}"
        out_dir.mkdir(parents=True)

        self._frame_idx = 0
        self._out_dir = out_dir
        self._logger = SummaryWriter(self._out_dir)

        if self._config["load_tracked_flame_params"] is not None:
            self.load_from_tracked_flame_params(
                self._config["load_tracked_flame_params"]
            )

    def load_from_tracked_flame_params(self, fp):
        """
        loads checkpoint from tracked_flame_params file. Counterpart to export_result()
        :param fp:
        :return:
        """
        report = np.load(fp)

        # LOADING PARAMETERS
        def load_param(param, ckpt_array):
            param.data[:] = torch.from_numpy(ckpt_array).to(param.device)

        def load_param_list(param_list, ckpt_array):
            for i in range(min(len(param_list), len(ckpt_array))):
                load_param(param_list[i], ckpt_array[i])

        load_param_list(self._rotation, report["rotation"])
        load_param_list(self._translation, report["translation"])
        load_param_list(self._neck_pose, report["neck_pose"])
        load_param_list(self._jaw_pose, report["jaw_pose"])
        load_param_list(self._eyes_pose, report["eyes_pose"])
        load_param(self._shape, report["shape"])
        load_param_list(self._expr, report["expr"])
        load_param(self._texture, report["texture"])
        load_param(self._lights, report["light"][0])
        self._frame_idx = report["n_processed"]
        if not self._calibrated:
            load_param(self._K, report["K"])
            load_param(self._RT, report["RT"])

    def _trimmed_decays(self, is_init):
        decays = {}
        for k, v in self._decays.items():
            if is_init and "init" in k or not is_init and "init" not in k:
                decays[k.replace("_init", "")] = v
        return decays

    def _clear_cache(self):
        self._render.clear_cache()

    def optimize(self):
        """
        Optimizes flame parameters on all frames of the dataset passed at initialization
        :return:
        """
        logger.info("Saving hyperparameters...")
        self._save_hyperparameters()

        logger.info(f"Start tracking FLAME in {self._n_frames} frames")
        for frame_idx in range(self._frame_idx, self._n_frames):
            # first initialize frame either from calibration or previous frame
            with torch.no_grad():
                self._initialize_frame(frame_idx)

            # get all parameters for this frame optimization and create optimizer
            # if 'is_init_frame' flag is set, the parameters will include not only the parameters
            # of the frame but also of all keyframes in self._config['keyframes']
            is_init_frame = frame_idx == 0
            train_params = self._get_train_parameters(frame_idx, is_init_frame)
            optimizer = self._configure_optimizer(train_params, is_init_frame)
            sample = self._get_current_frame(frame_idx, is_init_frame)

            if is_init_frame or frame_idx in self._config["cutframes"]:
                num_steps = self._config["init_steps"]
            else:
                num_steps = self._config["steps_per_frame"]

            logger.info(f"Start optimization of frame {frame_idx}")
            for step_i in range(num_steps):

                # compute loss and update parameters
                self._clear_cache()

                for _ in range(self._config["sub_steps"]):
                    self._fill_cam_params_into_sample(sample)
                    E_total, log_dict, verts, lmks, albedos = self._compute_energy(
                        sample, frame_idx, is_init_frame, step_i
                    )
                    optimizer.zero_grad()
                    E_total.backward()
                    optimizer.step()

                self._clear_cache()

                # log energy terms and visualize
                is_last = step_i == num_steps - 1
                has_to_log_energy = (
                    step_i % self._config["energy_log_freq"] == 0 or is_last
                )
                has_to_log_img = step_i % self._config["img_log_freq"] == 0 or is_last

                if has_to_log_energy or has_to_log_img:
                    with torch.no_grad():
                        res = self._compute_energy(
                            sample, frame_idx, is_init_frame, step_i
                        )
                        E_total, log_dict, verts, lmks, albedos = res

                    if has_to_log_energy:
                        self._log_scalars(log_dict, frame_idx, step_i)

                    if has_to_log_img:
                        self._log_tracking(
                            verts, lmks, albedos, sample, frame_idx, step_i
                        )

            if is_init_frame and not self._calibrated:
                logger.info(
                    f"Camera intrinsics optimized after initialization frame: {self._K}"
                )

            self._frame_idx = frame_idx + 1
            if frame_idx > 0 and frame_idx % self._config["save_period"] == 0:
                self._export_result()

        logger.info("Finished optimization. Saving results ...")
        self._export_result(make_visualization=True)

    def _get_current_frame(self, frame_idx, include_keyframes=False):
        """
        Creates a single item batch from the frame data at index frame_idx in the dataset.
        If include_keyframes option is set, keyframe data will be appended to the batch. However,
        it is guaranteed that the frame data belonging to frame_idx is at position 0
        :param frame_idx:
        :return:
        """
        indices = [frame_idx]
        samples = []
        if include_keyframes:
            indices += self._config["keyframes"]

        for idx in indices:
            sample = self._dataset[idx]
            sample["frame_index"] = idx

            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    sample[k] = v[None, ...].to(self._device)

            samples.append(sample)

        # if also keyframes have been loaded, stack all data
        sample = {}
        for k, v in samples[0].items():
            values = [s[k] for s in samples]
            if isinstance(v, torch.Tensor):
                values = torch.cat(values, dim=0)
            sample[k] = values

        if "lmk2d_iris" in sample:
            sample["lmk2d"] = torch.cat([sample["lmk2d"], sample["lmk2d_iris"]], dim=1)

        return sample

    def _fill_cam_params_into_sample(self, sample):
        """
        Adds intrinsics and extrinics to sample, if data is not calibrated
        """
        if self._calibrated:
            assert "cam_intrinsic" in sample and "cam_extrinsic" in sample
        else:
            b, _, w, h = sample["rgb"].shape
            K = torch.eye(3, 3).to(self._device)

            # denormalize cam params
            f = self._K[[0]] * max(h, w)
            cx, cy = self._K[[1]] * w, self._K[[2]] * h
            K[[0, 0, 1, 1], [0, 2, 1, 2]] = torch.cat([f, cx, f, cy])

            sample["cam_intrinsic"] = K[None, ...].expand(b, -1, -1)
            sample["cam_extrinsic"] = self._RT[None, ...].expand(b, -1, -1)

    def _get_train_parameters(self, frame_idx, include_keyframes=False):
        """
        Collects the parameters to be optimized for the current frame
        :param frame_idx: frame number
        :param first_frame: if true shape params and camera intrinsics will be optimized as well
        :return: dict of parameters
        """

        indices = [frame_idx]
        params = {
            "expr": [],
            "neck": [],
            "jaw": [],
            "translation": [],
            "rotation": [],
            "eyes": [],
        }

        # if init_frame, add all key frame params and shape, texture, lights and camera intrinsics
        if include_keyframes:
            indices += self._config["keyframes"]
            params["shape"] = [self._shape]
            params["lights"] = [self._lights]
            params["texture"] = [self._texture]
            if not self._calibrated:
                params["cam"] = [self._K]

        for idx in indices:
            params["expr"].append(self._expr[idx])
            params["eyes"].append(self._eyes_pose[idx])
            params["neck"].append(self._neck_pose[idx])
            params["jaw"].append(self._jaw_pose[idx])
            params["translation"].append(self._translation[idx])
            params["rotation"].append(self._rotation[idx])

        return params

    def _configure_optimizer(self, params, is_init_frame=True):
        """
        Creates optimizer for the given set of parameters
        :param params:
        :return:
        """
        # copy dict because we will call 'pop'
        params = params.copy()
        param_groups = []
        default_lr = self._config["lr"]

        # dict map group name to param dict keys
        group_def = {"pos": ["translation"]}
        if is_init_frame:
            if not self._calibrated:
                group_def["cam"] = ["cam"]
            group_def["light"] = ["lights"]

        # dict map group name to lr
        group_lr = {"pos": self._config["pos_lr"]}
        if is_init_frame:
            if not self._calibrated:
                group_lr["cam"] = self._config["cam_lr"]
            group_lr["light"] = self._config["light_lr"]

        for group_name, param_keys in group_def.items():
            selected = []
            for p in param_keys:
                selected += params.pop(p)
            param_groups.append({"params": selected, "lr": group_lr[group_name]})

        # create default group with remaining params
        selected = []
        for _, v in params.items():
            selected += v
        param_groups.append({"params": selected})

        optim = torch.optim.Adam(param_groups, lr=default_lr)
        return optim

    def _initialize_frame(self, frame_idx):
        """
        Initializes parameters of frame frame_idx
        :param frame_idx:
        :return:
        """
        if frame_idx > 0:
            self._initialize_from_previous(frame_idx)
        else:
            self._initialize_from_calibration(frame_idx)

    def _initialize_from_previous(self, frame_idx):
        """
        Initializes the flame parameters with the optimized ones from the previous frame
        :param frame_idx:
        :return:
        """
        if frame_idx == 0:
            return

        for param in [
            self._expr,
            self._neck_pose,
            self._jaw_pose,
            self._translation,
            self._rotation,
            self._eyes_pose,
        ]:
            param[frame_idx].data = param[frame_idx - 1].detach().clone().data

    def _initialize_from_calibration(self, frame_idx):
        """
        Initializes frame frame_idx from camera calibration. Right now this is only supported
        for frame 0
        :param frame_idx:
        :return:
        """

        device = self._device
        sample = self._get_current_frame(frame_idx, include_keyframes=True)
        assert frame_idx == sample["frame_index"][0] == 0

        # get image resolution
        img_h, img_w = sample["rgb"].shape[2:]

        # use landmarks for calibration
        lmks2d = sample["lmk2d"].clone()
        lmks2d, confidence = lmks2d[:, :, :2], lmks2d[:, :, 2]
        _, lmks3d, _ = self._forward_flame(frame_idx, include_keyframes=True)

        # calibrate camera using lmk correspondences as pattern
        world_pts, img_pts = [], []
        for lmks2d_i, lmks3d_i, conf_i in zip(lmks2d, lmks3d, confidence):
            # mask 2d landmarks with low confidence (possible false prediction)
            mask = conf_i > 0.2
            assert mask.sum() >= 6

            world_pts.append(lmks3d_i[mask].cpu().numpy())
            img_pts.append(lmks2d_i[mask].cpu().numpy())

        if self._calibrated:
            # calibrate flame pose (extrinics with only the intrinsics used) and then
            # transform them using the inverse of the true extrinsics
            assert len(world_pts) == len(img_pts)

            rs, ts = [], []
            for w_pts, i_pts, K in zip(world_pts, img_pts, sample["cam_intrinsic"]):
                K = K.detach().cpu().numpy()
                success, rot, tra = calibrate_extrinsics(w_pts, i_pts, K, None)
                assert success
                logger.debug(
                    f"Calibrated pose of frame {frame_idx} with success: {success}"
                )
                rs.append(rot)
                ts.append(tra)

        else:
            f = max(img_w, img_h)
            K = np.array([f, 0, img_w / 2, 0, f, img_h / 2, 0, 0, 1.0]).reshape(3, 3)

            error, new_K, dist, rs, ts = calibrate_camera(
                world_pts, img_pts, image_size=(img_h, img_w), K=K, ignore_dist=True
            )

            # normalize cam inputs by image size
            f = new_K[0, 0] / max(img_w, img_h)
            cx, cy = new_K[0, 2] / img_w, new_K[1, 2] / img_h
            self._K.data = torch.tensor([f, cx, cy], device=device).float()

            logger.debug(f"Calibration matrix is {self._K}")
            logger.debug(f"Calibrated camera of frame {frame_idx} with error {error}")

        flip_xy = R.from_rotvec([0, 0, np.pi])
        for i, (r, t, frame_i) in enumerate(zip(rs, ts, sample["frame_index"])):
            r, t = r.flatten(), t.flatten()

            # cam coord from opencv ( x right, y down)
            # cam coord from pytorch3d (x left, y up)
            # hence, flip_xy
            r = flip_xy * R.from_rotvec(r)
            r = r.as_rotvec()
            t[[0, 1]] *= -1

            if self._calibrated:
                # apply inverse extrinsics matrix
                def rt2RT(r, t):
                    RT = np.eye(4)
                    RT[:3, :3] = R.from_rotvec(r).as_matrix()
                    RT[:3, 3] = t
                    return RT

                RT_o = rt2RT(r, t)
                neg = np.eye(4)
                neg[2, 2] = -1

                RT_c = sample["cam_extrinsic"][i].detach().cpu().numpy()
                RT_c = np.concatenate([RT_c, [[0, 0, 0, 1]]], axis=0)
                inv_RT_c = np.linalg.inv(RT_c)
                new_RT_o = neg @ inv_RT_c @ neg @ RT_o

                new_r = R.from_matrix(new_RT_o[:3, :3]).as_rotvec()
                new_t = new_RT_o[:3, 3]

                r = new_r
                t = new_t

            # init weights
            self._translation[frame_i].data = torch.from_numpy(t).float().to(device)
            self._rotation[frame_i].data = torch.from_numpy(r).float().to(device)

    def _select_frame_indices(self, frame_idx, include_keyframes):
        indices = [frame_idx]
        if include_keyframes:
            indices += self._config["keyframes"]
        return indices

    def _forward_flame(self, frame_idx, include_keyframes):
        """
        Evaluates the flame model using the given parameters
        :param flame_params:
        :return:
        """
        indices = self._select_frame_indices(frame_idx, include_keyframes)

        ret = self._flame(
            self._shape[None, ...].expand(len(indices), -1),
            _to_batch(self._expr, indices),
            _to_batch(self._rotation, indices),
            _to_batch(self._neck_pose, indices),
            _to_batch(self._jaw_pose, indices),
            _to_batch(self._eyes_pose, indices),
            _to_batch(self._translation, indices),
        )

        verts, lmks = ret[0], ret[1]
        albedos = self._flame_tex(self._texture[None, ...].expand(len(indices), -1))
        return verts, lmks, albedos

    def _rasterize_flame(self, sample, vertices, scale=1, use_cache=True):

        """
        Rasterizes the flame head mesh
        :param vertices:
        :param albedos:
        :param K:
        :param RT:
        :param resolution:
        :param scale:
        :param only_face:
        :param use_cache:
        :return:
        """
        faces = self._flame.faces
        if self._config["ignore_lower_neck"]:
            faces = faces[self._face_mask]

        # create cameras
        K = sample["cam_intrinsic"]
        RT = sample["cam_extrinsic"]
        H, W = self._image_size
        H, W = int(H * scale), int(W * scale)
        K = K * scale
        cameras = create_camera_objects(K, RT, (H, W), self._device)

        # create mesh from selected faces and vertices
        flame_meshes = Meshes(verts=vertices, faces=faces.expand(len(vertices), -1, -1))

        # rasterize fragments
        render_result = self._render.rasterize(flame_meshes, cameras, (H, W), use_cache)

        return {
            "fragments": render_result[0],
            "screen_coords": render_result[1],
            "meshes": flame_meshes,
        }

    def _render_rgba(self, rasterization_result, albedos):
        """
        Renders the rgba image from the rasterization result and
        the optimized texture + lights
        """
        fragments = rasterization_result["fragments"]
        meshes = rasterization_result["meshes"]

        B = len(meshes)
        lights = self._lights[None].expand(B, -1, -1)
        uv_coords = self._flame.face_uvcoords
        if self._config["ignore_lower_neck"]:
            uv_coords = uv_coords[self._face_mask]
        uv_coords = uv_coords.repeat(B, 1, 1)
        return self._render.render_rgba(meshes, fragments, uv_coords, albedos, lights)

    def _compute_lmk_energy(self, frame_idx, sample, pred_lmks):
        """
        Computes the landmark energy loss term between groundtruth landmarks and flame landmarks
        :param sample:
        :param pred_lmks:
        :return: the lmk loss for all 68 facial landmarks, a separate 2 pupil landmark loss and
                 a relative eye close term
        """
        img_size = sample["rgb"].shape[-2:]
        K = sample["cam_intrinsic"]
        RT = sample["cam_extrinsic"]

        lmks = sample["lmk2d"].clone()
        lmks, confidence = lmks[:, :, :2], lmks[:, :, 2]
        lmks[:, :, 0], lmks[:, :, 1] = normalize_image_points(
            lmks[:, :, 0], lmks[:, :, 1], img_size
        )

        # create cameras
        K = sample["cam_intrinsic"]
        RT = sample["cam_extrinsic"]
        cameras = create_camera_objects(K, RT, self._image_size, self._device)

        proj_pred_lmks = cameras.transform_points_screen(pred_lmks)
        proj_pred_lmks[:, :, 0], proj_pred_lmks[:, :, 1] = normalize_image_points(
            proj_pred_lmks[:, :, 0], proj_pred_lmks[:, :, 1], img_size
        )

        proj_pred_lmks = proj_pred_lmks[:, :, :2]
        diff = lmks - proj_pred_lmks

        # compute general landmark term
        lmk_loss = torch.norm(diff[:, :68], dim=2, p=1) * confidence[:, :68]

        # compute pupil landmark term
        eye_lmk_loss = torch.norm(diff[:, 68:], dim=2, p=1) * confidence[:, 68:]

        return lmk_loss.mean(), eye_lmk_loss.mean()

    def _compute_photometric_energy(
        self, sample, albedos, rasterization_result, step_i
    ):
        """
        Computes the dense photometric energy
        :param sample:
        :param vertices:
        :param albedos:
        :return:
        """
        gt_rgb = sample["rgb"] * 0.5 + 0.5
        gt_rgb = gt_rgb  # * gt_seg

        screen_coords = rasterization_result["screen_coords"] * -1
        render_result = self._render_rgba(rasterization_result, albedos)

        predicted_images = render_result[:, :3]
        mask = render_result[:, [3]].detach() > 0
        mask = mask.expand(-1, 3, -1, -1)

        # coarse to fine
        decays = self._trimmed_decays(sample["frame_index"] == 0)
        gt_rgb = blur_tensors(gt_rgb, sigma=decays["blur_sigma"].get(step_i))[0]

        screen_colors = F.grid_sample(gt_rgb, screen_coords)

        photo_loss = (predicted_images - screen_colors)[mask].abs()
        return photo_loss.sum() / mask.sum()

    def _compute_regularization_energy(self, frame_idx, include_keyframes):
        """
        Computes the energy term that penalizes strong deviations from the flame base model
        """
        std_tex = 1
        std_expr = 1
        std_shape = 1

        indices = self._select_frame_indices(frame_idx, include_keyframes)

        reg_shape = (self._shape / std_shape) ** 2
        reg_tex = (self._texture / std_tex) ** 2
        expr = _to_batch(self._expr, indices)
        reg_expr = (expr / std_expr) ** 2

        return reg_shape.sum(), reg_expr.sum() / len(indices), reg_tex.sum()

    def _compute_pose_energy(self, frame_idx, include_keyframes):
        """
        Regularizes the pose of the flame head model towards neutral joint locations
        """
        indices = self._select_frame_indices(frame_idx, include_keyframes)

        neutral_poses = self._flame.get_neutral_joint_rotations()
        neutral_poses["global"] = torch.zeros(3, device=self._device)

        # get current poses
        # current_poses = [_to_batch(self._rotation, indices)]
        current_poses = [_to_batch(self._neck_pose, indices)]
        current_poses += [_to_batch(self._jaw_pose, indices)]
        eye_pose = _to_batch(self._eyes_pose, indices)
        current_poses += [eye_pose[:, :3], eye_pose[:, 3:]]

        E_pos = 0
        for key, current, weight in zip(
            ["neck", "jaw", "eyes", "eyes"],  # ['global']
            current_poses,
            ["neck", "jaw", "eyes", "eyes"],  # ['glob']
        ):
            neutral = neutral_poses[key][None, ...]
            rotmats = batch_rodrigues(torch.cat([neutral, current], dim=0))

            diff = (rotmats[[0]] - rotmats[1:]) ** 2
            E_pos += diff.sum() / len(indices) * self._config[f"w_pos_{weight}"]

        if frame_idx == 0:
            trans = _to_batch(self._translation, indices)
            ref_trans = torch.mean(trans, dim=0, keepdim=True)
        else:
            trans, ref_trans = (
                self._translation[frame_idx],
                self._translation[frame_idx - 1],
            )

        diff_trans = (trans - ref_trans) ** 2
        E_pos += diff_trans.sum() * self._config["w_pos_trans"]

        return E_pos

    def _compute_eye_symmetry_energy(self, frame_idx, include_keyframes):
        """
        Computes the difference between the left and right eyeball rotation
        :param frame_idx: current frame
        :param include_keyframes: if key frames shall be included
        :return: Mean-squared difference of eye ball rotations
        """

        indices = self._select_frame_indices(frame_idx, include_keyframes)
        eyes = _to_batch(self._eyes_pose, indices)
        right_eye, left_eye = eyes[:, :3], eyes[:, 3:]
        diff = (right_eye - left_eye) ** 2
        return diff.sum() / len(indices)

    def _compute_energy(self, sample, frame_idx, include_keyframes, step_i):
        """
        Compute total energy for frame frame_idx
        :param sample:
        :param frame_idx:
        :param include_keyframes: if key frames shall be included when predicting the per
        frame energy
        :return: loss, log dict, predicted vertices and landmarks
        """
        verts, lmks, albedos = self._forward_flame(frame_idx, include_keyframes)
        rasterization_results = self._rasterize_flame(
            sample, verts, scale=self._config["sampling_scale"], use_cache=True
        )

        E_lmk, E_eyes_lmk = self._compute_lmk_energy(frame_idx, sample, lmks)

        E_photo = self._compute_photometric_energy(
            sample, albedos, rasterization_results, step_i
        )

        E_shape, E_expr, E_tex = self._compute_regularization_energy(
            frame_idx, include_keyframes
        )

        E_pos = self._compute_pose_energy(frame_idx, include_keyframes)
        E_eyes_sym = self._compute_eye_symmetry_energy(frame_idx, include_keyframes)

        E_total = (
            self._config["w_lmk"] * E_lmk
            + self._config["w_eyes_lmk"] * E_eyes_lmk
            + self._config["w_photo"] * E_photo
            + self._config["w_expr_reg"] * E_expr
            + self._config["w_shape_reg"] * E_shape
            + self._config["w_tex_reg"] * E_tex
            + self._config["w_eyes_sym"] * E_eyes_sym
            + E_pos
        )

        log_dict = {
            "E_total": E_total,
            "E_lmk": E_lmk,
            "E_eyes_lmk": E_eyes_lmk,
            "E_photo": E_photo,
            "E_expr_reg": E_expr,
            "E_shape_reg": E_shape,
            "E_tex_reg": E_tex,
            "E_pos": E_pos,
            "E_eyes_sym": E_eyes_sym,
        }

        return E_total, log_dict, verts, lmks, albedos

    def _global_step(self, frame_idx, step_idx):
        """
        Returns unique global step number
        :param frame_idx:
        :param step_idx:
        :return:
        """
        if frame_idx == 0:
            step_id = step_idx
        else:
            step_id = self._config["init_steps"]
            step_id += (frame_idx - 1) * self._config["steps_per_frame"] + step_idx
        return step_id

    def _log_scalars(self, log_dict, frame_idx, step_i):
        """
        Logs scalars in log_dict to tensorboard and logger
        :param log_dict:
        :param frame_idx:
        :param step_i:
        :return:
        """

        log_msg = ""
        global_step = self._global_step(frame_idx, step_i)

        for k, v in self._decays.items():
            decay = v.get(step_i)
            log_dict[f"decay_{k}"] = decay

        for k, v in log_dict.items():
            if not k.startswith("decay"):
                log_msg += " {}: {:.4f}; ".format(k, v)
            self._logger.add_scalar(k, v, global_step)

        logger.info(f"Training progress frame {frame_idx} step {step_i}: {log_msg}")

    def _log_tracking(self, vertices, lmks, albedos, sample, frame_idx, step_i):
        """
        Logs current tracking visualization to tensorboard
        :param vertices:
        :param lmks:
        :param sample:
        :param frame_idx:
        :param step_i:
        :param show_lmks:
        :param show_overlay:
        :return:
        """
        step_id = self._global_step(frame_idx, step_i)
        log_figure = self._visualize_tracking(vertices, lmks, albedos, sample)
        self._logger.add_image("tracking_result", log_figure, step_id)

        log_figure = self._visualize_flame_multiview(vertices, albedos, sample)
        self._logger.add_image("flame_multiview", log_figure, step_id)

        # log_figure = self._visualize_trajectory(sample)
        # self._logger.add_image("translation_trajectory", log_figure, step_id)

    @torch.no_grad()
    def _visualize_flame_multiview(self, vertices, albedos, sample):

        # prepare sample to be used with three instances
        new_sample = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                new_sample[k] = v[[0, 0, 0]]
            else:
                new_sample[k] = v
        sample = new_sample

        # rotate vertices to view them from center, left and right
        frame_idx = sample["frame_index"][0]
        vertices = vertices[0]

        # subtract translation before rotating
        t = self._translation[frame_idx][None]
        vertices_center = vertices - t

        # rotate
        turn_90 = batch_rodrigues(
            torch.tensor([[0, np.pi / 2, 0]], device=self._device).float()
        )[0]
        vertices_right = (turn_90 @ vertices_center.T).T
        vertices_left = (turn_90.T @ vertices_center.T).T

        # add translation back to the vertices
        vertices = (
            torch.stack([vertices_center, vertices_right, vertices_left]) + t[None]
        )

        # render
        rasterization_results = self._rasterize_flame(
            sample, vertices, scale=1, use_cache=False
        )
        render_result = self._render_rgba(rasterization_results, albedos[[0, 0, 0]])
        predicted_images, alpha = render_result[:, :3], render_result[:, 3:]
        predicted_images = torch.clip(predicted_images, min=0, max=1)
        predicted_images = predicted_images * alpha

        return torchvision.utils.make_grid(predicted_images, nrow=3)

    @torch.no_grad()
    def _visualize_tracking(
        self,
        vertices,
        lmks,
        albedos,
        sample,
        max_instances=5,
        return_imgs_seperately=False,
    ):
        """
        Visualizes the tracking result
        """

        image = sample["rgb"] * 0.5 + 0.5
        K = sample["cam_intrinsic"]
        RT = sample["cam_extrinsic"]
        resolution = image.shape[-2:]

        rasterization_results = self._rasterize_flame(
            sample, vertices, scale=1, use_cache=False
        )
        render_result = self._render_rgba(rasterization_results, albedos)
        predicted_images, alpha = render_result[:, :3], render_result[:, 3:]
        predicted_images = torch.clip(predicted_images, min=0, max=1)
        predicted_images = predicted_images * alpha

        # visualize the information of the batch in 'sample' but at most 'max_instances'
        images = []
        n = min(sample["rgb"].shape[0], max_instances)
        for i in range(n):
            image_i = image[[i]]
            gt_lmks = sample["lmk2d"][[i]]
            images.append(image_i)
            images.append(predicted_images[[i]])

            # add overlay of silhouette and landmarks
            alpha_i = alpha[[i]]
            overlay_image = alpha_i * alpha_i * 0.5 + image_i * (1 - alpha_i * 0.5)

            cameras = create_camera_objects(K[[i]], RT[[i]], resolution, self._device)
            proj_lmks = cameras.transform_points_screen(lmks[[i]])

            overlay_image = plot_2Dlandmarks(overlay_image.clone(), proj_lmks)
            images.append(overlay_image)

            # add overlay of gt landmarks
            lmk_image = plot_2Dlandmarks(image_i.clone(), gt_lmks)
            images.append(lmk_image)

            # trajectory_image = self._visualize_trajectory(sample)
            # images.append(trajectory_image)

        if return_imgs_seperately:
            return images
        else:
            images = torch.cat(images, dim=0)
            return torchvision.utils.make_grid(images, nrow=4)

    @torch.no_grad()
    def _visualize_trajectory(self, sample):
        """
        Visualizes the trajectory of the tracked model in camera space. That is the trajectory
        of the translation vectors
        :param sample:
        :return:
        """
        indices = list(range(max(sample["frame_index"])))
        translations = _to_batch(self._translation, indices).detach().cpu().numpy()

        fig = plt.figure(figsize=(2, 2), dpi=100, tight_layout=True)
        ax = fig.add_subplot(projection="3d")

        cmap = matplotlib.cm.get_cmap("Spectral")
        for i in range(len(translations) - 1):
            neighbors = translations[[i, i + 1]]
            ax.plot(
                neighbors[:, 0],
                -neighbors[:, 2],
                neighbors[:, 1],
                color=cmap(i / len(translations)),
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        fig.canvas.draw()

        # convert figure to image
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)).transpose(
            2, 0, 1
        )
        return torch.tensor(data, device=self._device, dtype=float) * 2 - 1

    @torch.no_grad()
    def _save_tracking_animation(self, outpath):
        """
        Creates tracking animation based on the optimized parameters and saves it to 'outpath'
        :return: figure, artists list (see matplotlib)
        """

        # init animation
        fig, ax = plt.subplots(figsize=(20, 4))
        sample = self._get_current_frame(0, include_keyframes=False)
        self._fill_cam_params_into_sample(sample)
        verts, lmks, albedos = self._forward_flame(0, include_keyframes=False)
        frame_img = self._visualize_tracking(verts, lmks, albedos, sample).cpu().numpy()
        actors = [ax.imshow(frame_img.transpose(1, 2, 0))]
        ax.axis("off")

        def update(n):
            sample = self._get_current_frame(n, include_keyframes=False)
            self._fill_cam_params_into_sample(sample)
            verts, lmks, albedos = self._forward_flame(n, include_keyframes=False)
            frame_img = (
                self._visualize_tracking(verts, lmks, albedos, sample).cpu().numpy()
            )
            actors[0].set_array(frame_img.transpose(1, 2, 0))
            return actors

        logger.info("Started Animation")
        anim = FuncAnimation(
            fig,
            func=update,
            frames=self._n_frames,
            interval=1000.0 / self._config["frame_rate"],
        )
        callback = lambda i, n: print(f"Saving frame {i} of {n}")
        os.makedirs(outpath.parent, exist_ok=True)
        anim.save(outpath, progress_callback=callback)
        logger.info("Finished Animation")
        return anim

    def _save_hyperparameters(self):
        """
        Saves hyperparameters
        :return:
        """
        with open(self._out_dir / "config.ini", "w") as f:
            for k, v in self._config.items():
                f.write(f"{k} = {v}\n")

    def _export_result(self, fname=None, make_visualization=False):
        """
        Saves tracked/optimized flame parameters. In addition, the tracking visualization is
        stored as well.
        :return:
        """
        # save parameters
        keys = [
            "rotation",
            "translation",
            "neck_pose",
            "jaw_pose",
            "eyes_pose",
            "shape",
            "expr",
            "texture",
            "frame",
            "view",
            "light",
            "n_processed",
        ]
        values = [
            self._rotation,
            self._translation,
            self._neck_pose,
            self._jaw_pose,
            self._eyes_pose,
            self._shape,
            self._expr,
            self._texture,
            np.array(self._dataset.frame_list[: len(self._expr)]),
            np.array(self._dataset.view_list[: self._n_frames]),
            self._lights[None].expand(self._n_frames, -1, -1),
            self._frame_idx,
        ]

        if not self._calibrated:
            keys += ["K", "RT"]
            values += [self._K, self._RT]

        export_dict = {}
        for k, v in zip(keys, values):
            if not isinstance(v, np.ndarray):
                if isinstance(v, list):
                    v = torch.stack(v)
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
            export_dict[k] = v

        export_dict["image_size"] = np.array(self._image_size)
        fname = fname if fname is not None else "tracked_flame_params.npz"
        bytesbuffer = io.BytesIO()
        np.savez(bytesbuffer, **export_dict)
        # with fsspec.open(self._out_dir / fname, "wb") as f:
        #    f.write(bytesbuffer.getvalue())

        # save tracking animation
        if make_visualization:
            self._save_tracking_animation(self._out_dir / "tracking_visual.mp4")
