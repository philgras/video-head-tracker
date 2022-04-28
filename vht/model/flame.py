# Code heavily inspired by https://github.com/HavenFeng/photometric_optimization/blob/master/models/FLAME.py.
# Please consider citing their work if you find this code useful. The code is subject to the license available via
# https://github.com/vchoutas/smplx/edit/master/LICENSE

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de


from vht.model.lbs import lbs, vertices2landmarks
from vht.util.graphics import face_vertices
from vht.util.log import get_logger
from pytorch3d.io import load_obj

import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F

logger = get_logger(__name__)

FLAME_MODEL_PATH = "assets/flame/generic_model.pkl"
FLAME_MESH_PATH = "assets/flame/head_template_mesh.obj"
FLAME_PARTS_PATH = "assets/flame/FLAME_masks.pkl"
FLAME_LMK_PATH = "assets/flame/landmark_embedding_with_eyes.npy"
FLAME_TEX_PATH = "assets/flame/FLAME_texture.npz"


def to_tensor(array, dtype=torch.float32):
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class FlameHead(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(
        self,
        shape_params,
        expr_params,
        flame_model_path=FLAME_MODEL_PATH,
        flame_lmk_embedding_path=FLAME_LMK_PATH,
        flame_template_mesh_path=FLAME_MESH_PATH,
        eye_limits=((-50, 50), (-50, 50), (-0.1, 0.1)),
        neck_limits=((-90, 90), (-60, 60), (-80, 80)),
        jaw_limits=((-5, 60), (-0.1, 0.1), (-0.1, 0.1)),
    ):
        super().__init__()

        self.n_shape_params = shape_params
        self.n_expr_params = expr_params

        with open(flame_model_path, "rb") as f:
            ss = pickle.load(f, encoding="latin1")
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        # The vertices of the template model
        self.register_buffer(
            "v_template", to_tensor(to_np(flame_model.v_template), dtype=self.dtype)
        )

        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat(
            [shapedirs[:, :, :shape_params], shapedirs[:, :, 300 : 300 + expr_params]],
            2,
        )
        self.register_buffer("shapedirs", shapedirs)

        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype))
        #
        self.register_buffer(
            "J_regressor", to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype)
        )
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer(
            "lbs_weights", to_tensor(to_np(flame_model.weights), dtype=self.dtype)
        )

        # rotation limits for the respective joints
        neck_limits = torch.tensor(neck_limits).float() / 180 * np.pi
        self.register_buffer("neck_limits", neck_limits)
        jaw_limits = torch.tensor(jaw_limits).float() / 180 * np.pi
        self.register_buffer("jaw_limits", jaw_limits)
        eye_limits = torch.tensor(eye_limits).float() / 180 * np.pi
        self.register_buffer("eye_limits", eye_limits)

        # Landmark embeddings for FLAME
        lmk_embeddings = np.load(
            flame_lmk_embedding_path, allow_pickle=True, encoding="latin1"
        )
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer(
            "full_lmk_faces_idx",
            torch.tensor(lmk_embeddings["full_lmk_faces_idx"], dtype=torch.long),
        )
        self.register_buffer(
            "full_lmk_bary_coords",
            torch.tensor(lmk_embeddings["full_lmk_bary_coords"], dtype=self.dtype),
        )

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer("neck_kin_chain", torch.stack(neck_kin_chain))

        # add faces and uvs
        _, faces, aux = load_obj(flame_template_mesh_path, load_textures=False)
        vertex_uvs = aux.verts_uvs
        face_uvs_idx = faces.textures_idx  # index into verts_uvs

        # create uvcoords per face --> this is what you can use for uv map rendering
        # range from -1 to 1 (-1, -1) = left top; (+1, +1) = right bottom
        # pad 1 to the end
        pad = torch.ones(vertex_uvs.shape[0], 1)
        vertex_uvs = torch.cat([vertex_uvs, pad], dim=-1)
        vertex_uvs = vertex_uvs * 2 - 1
        vertex_uvs[..., 1] = -vertex_uvs[..., 1]

        face_uv_coords = face_vertices(vertex_uvs[None], face_uvs_idx[None])[0]
        self.register_buffer("face_uvcoords", face_uv_coords, persistent=False)
        self.register_buffer("faces", faces.verts_idx, persistent=False)

    def _apply_rotation_limit(self, rotation, limit):
        r_min, r_max = limit[:, 0].view(1, 3), limit[:, 1].view(1, 3)
        diff = r_max - r_min
        return r_min + (torch.tanh(rotation) + 1) / 2 * diff

    def apply_rotation_limits(self, neck, jaw, eyes):
        """
        method to call for applying rotation limits. Don't use _apply_rotation_limit() in other methods as this
        might cause some bugs if we change which poses are affected by rotation limits. For this reason, in this method,
        all affected poses are limited within one function so that if we add more restricted poses, they can just be
        updated here
        :param neck:
        :param jaw:
        :return:
        """
        neck = self._apply_rotation_limit(neck, self.neck_limits)
        jaw = self._apply_rotation_limit(jaw, self.jaw_limits)
        eye_r = self._apply_rotation_limit(eyes[:, :3], self.eye_limits)
        eye_l = self._apply_rotation_limit(eyes[:, 3:], self.eye_limits)
        eyes = torch.cat([eye_r, eye_l], dim=1)
        return neck, jaw, eyes

    def _revert_rotation_limit(self, rotation, limit):
        """
        inverse function of _apply_rotation_limit()
        from rotation angle vector (rodriguez) -> scalars from -inf ... inf
        :param rotation: tensor of shape N x 3
        :param limit: tensor of shape 3 x 2 (min, max)
        :return:
        """
        r_min, r_max = limit[:, 0].view(1, 3), limit[:, 1].view(1, 3)
        diff = r_max - r_min
        rotation = rotation.clone()
        for i in range(3):
            rotation[:, i] = torch.clip(
                rotation[:, i],
                min=r_min[0, i] + diff[0, i] * 0.01,
                max=r_max[0, i] - diff[0, i] * 0.01,
            )
        return torch.atanh((rotation - r_min) / diff * 2 - 1)

    def revert_rotation_limits(self, neck, jaw, eyes):
        """
        inverse function of apply_rotation_limits()
        from rotation angle vector (rodriguez) -> scalars from -inf ... inf
        :param rotation:
        :param limit:
        :return:
        """
        neck = self._revert_rotation_limit(neck, self.neck_limits)
        jaw = self._revert_rotation_limit(jaw, self.jaw_limits)
        eye_r = self._revert_rotation_limit(eyes[:, :3], self.eye_limits)
        eye_l = self._revert_rotation_limit(eyes[:, 3:], self.eye_limits)
        eyes = torch.cat([eye_r, eye_l], dim=1)
        return neck, jaw

    def get_neutral_joint_rotations(self):
        res = {}
        for name, limit in zip(
            ["neck", "jaw", "eyes"],
            [self.neck_limits, self.jaw_limits, self.eye_limits],
        ):
            r_min, r_max = limit[:, 0], limit[:, 1]
            diff = r_max - r_min
            res[name] = torch.atanh(-2 * r_min / diff - 1)
            # assert (r_min + (torch.tanh(res[name]) + 1) / 2 * diff) < 1e-7
        return res

    def forward(
        self,
        shape,
        expr,
        rotation,
        neck,
        jaw,
        eyes,
        translation,
        zero_centered=True,
        use_rotation_limits=True,
        return_landmarks=True,
    ):
        """
        Input:
            shape_params: N X number of shape parameters
            expression_params: N X number of expression parameters
            pose_params: N X number of pose parameters (6)
        return:d
            vertices: N X V X 3
            landmarks: N X number of landmarks X 3
        """
        batch_size = shape.shape[0]

        # apply limits to joint rotations
        if use_rotation_limits:
            neck, jaw, eyes = self.apply_rotation_limits(neck, jaw, eyes)

        betas = torch.cat([shape, expr], dim=1)
        full_pose = torch.cat([rotation, neck, jaw, eyes], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        faces = self.faces.unsqueeze(0).expand(batch_size, -1, -1)
        vertices, J, mat_rot = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            dtype=self.dtype,
            faces=faces,
        )

        if zero_centered:
            vertices = vertices - J[:, [0]]
            J = J - J[:, [0]]

        vertices = vertices + translation[:, None, :]
        J = J + translation[:, None, :]
        ret_vals = [vertices]

        # compute landmarks if desired
        if return_landmarks:
            bz = vertices.shape[0]
            landmarks = vertices2landmarks(
                vertices,
                self.faces,
                self.full_lmk_faces_idx.repeat(bz, 1),
                self.full_lmk_bary_coords.repeat(bz, 1, 1),
            )
            ret_vals.append(landmarks)

        if len(ret_vals) > 1:
            return ret_vals
        else:
            return ret_vals[0]


class FlameTex(nn.Module):
    def __init__(self, tex_params, tex_size=256, tex_space_path=FLAME_TEX_PATH):
        super().__init__()
        self._tex_size = tex_size
        tex_params = tex_params
        tex_space = np.load(tex_space_path)
        texture_mean = tex_space["mean"].reshape(1, -1)
        texture_basis = tex_space["tex_dir"].reshape(-1, 200)
        texture_mean = torch.from_numpy(texture_mean).float()[None, ...]
        texture_basis = torch.from_numpy(texture_basis[:, :tex_params]).float()[
            None, ...
        ]
        self.register_buffer("texture_mean", texture_mean)
        self.register_buffer("texture_basis", texture_basis)

    def forward(self, texcode):
        texture = self.texture_mean + (self.texture_basis * texcode[:, None, :]).sum(-1)
        texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
        texture = F.interpolate(texture, [self._tex_size, self._tex_size])
        texture = texture[:, [2, 1, 0], :, :]
        texture = texture / 255.0
        return texture.clamp(0, 1)
