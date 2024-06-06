import copy
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

# from embedder import get_embedder
from utils.rigging import RigModel


# import torch
# from torch import nn


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim



class DeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, t_multires=6, multires=10,
                 is_blender=False, local_frame=False, pred_opacity=False, pred_color=False, 
                 resnet_color=True, hash_color=False, color_wrt_dir=False, 
                 progressive_brand_time=False, max_d_scale=-1, **kwargs):  # t_multires 6 for D-NeRF; 10 for HyperNeRF
        super(DeformNetwork, self).__init__()
        self.name = 'mlp'
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]

        self.progressive_brand_time = progressive_brand_time
        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        # self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.input_ch = time_input_ch

        self.pred_opacity = pred_opacity
        self.pred_color = pred_color
        self.resnet_color = resnet_color
        self.hash_color = not resnet_color and hash_color
        self.color_wrt_dir = color_wrt_dir
        self.max_d_scale = max_d_scale

        self.reg_loss = 0.

        # if is_blender:
        #     # Better for D-NeRF Dataset
        #     self.time_out = 30

        #     self.timenet = nn.Sequential(
        #         nn.Linear(time_input_ch, 256), nn.ReLU(inplace=True),
        #         nn.Linear(256, self.time_out))

        #     self.linear = nn.ModuleList(
        #         [nn.Linear(self.time_out, W)] + [
        #             nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.time_out, W)
        #             for i in range(D - 1)]
        #     )

        # else:
        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)]
        )

        self.is_blender = False
        
        # TODO: change path?
        joint_load = np.load('jnb.npy', allow_pickle=True).item()
        self.joints = torch.from_numpy(joint_load['joints'])
        self.bones_list = joint_load['bones']
        self.root_idx = joint_load['root_idx']
        
        self.rig_model = RigModel(self.joints, self.bones_list, self.root_idx)

        self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_scaling = nn.Linear(W, len(self.joints), 3)
        self.bone_rotation = nn.Linear(W, len(self.joints), 4)
        self.joint_rotation = nn.Linear(W, len(self.joints), 4)

        # self.local_frame = local_frame
        # if self.local_frame:
        #     self.local_rotation = nn.Linear(W, 4)
        #     nn.init.normal_(self.local_rotation.weight, mean=0, std=1e-4)
        #     nn.init.zeros_(self.local_rotation.bias)
        
        for layer in self.linear:
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)

        nn.init.normal_(self.gaussian_warp.weight, mean=0, std=1e-5)
        nn.init.normal_(self.gaussian_scaling.weight, mean=0, std=1e-8)
        nn.init.normal_(self.bone_rotation.weight, mean=0, std=1e-5)
        nn.init.normal_(self.joint_rotation.weight, mean=0, std=1e-5)
        nn.init.zeros_(self.gaussian_warp.bias)
        nn.init.zeros_(self.gaussian_scaling.bias)
        nn.init.zeros_(self.bone_rotation.bias)
        nn.init.zeros_(self.joint_rotation.bias)

    def trainable_parameters(self):
        return [{'params': list(self.parameters()), 'name': 'mlp'}]
    
    def get_nodes(self):
        return self.joints

    def forward(self, t, **kwargs):
        # t: 1
        assert len(t.shape) == 1 or len(t.shape) == 0
        t_emb = self.embed_time_fn(t)
        t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
        # x_emb = self.embed_fn(x)
        h = t_emb
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([t_emb, h], -1)

        d_root = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h).reshape(-1, len(self.joints), 3)  # Joint, 3
        rot = self.bone_rotation(h).reshape(-1, len(self.joints), 4)  # Joint, 4
        rotation = self.joint_rotation(h).reshape(-1, len(self.joints), 4)  # joint,4 
        
        # use rigging model to estimate RESIDUAL for node positions, rotations
        d_xyz = self.rig_model.animate(rot)  # Bone, 4 -> Joint, 3 & Joint, 4

        if self.max_d_scale > 0:
            scaling = torch.tanh(scaling) * np.log(self.max_d_scale)

        return_dict = {'d_xyz': d_xyz + d_root, 'd_rotation': rotation, 'd_scaling': scaling, 'hidden': h}
        return_dict['d_opacity'] = None
        return_dict['d_color'] = None
        # if self.local_frame:
        #     return_dict['local_rotation'] = self.local_rotation(h)
        return return_dict
    
    def update(self, iteration, *args, **kwargs):
        if self.progressive_brand_time:
            self.embed_time_fn.update_step(iteration)
        return
