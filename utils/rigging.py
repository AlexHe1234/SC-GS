import torch
import copy


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def find_new_q(b, i):
    ret_p = []
    ret_b = []
    for p in b:
        if p[0] == i: ret_p.append(p[-1])
        else:         ret_b.append(p)
    return ret_p, ret_b


def find_to_joint(b, i):
    for bi in b:
        if bi[-1] == i: return bi[0]
    raise ValueError('This joint is disconnected from the graph')
    

class RigModel:
    def __init__(self, joints: torch.Tensor, bones: list, root: int):
        self.j = joints
        self.b = bones
        self.r = root
        
        # bones in the same order as their "to" joints
        self.b_vec = torch.empty_like(self.j)
        self.b_vec[self.r] = 0.
        for i in range(len(self.j)):
            if i == self.r: continue
            fj = find_to_joint(self.b, i)
            self.b_vec[i] = self.j[i] - self.j[fj]
            
        self.quat_bias = torch.tensor([1.,0.,0.,0.])
        
    def animate(self, quat: torch.Tensor): # Bone, 4 -> Joint, 3 & Joint, 4
        # ! return RESIDUAL
        quat = quat + self.quat_bias
        rot_mats = quaternion_to_matrix(quat)
        # breakpoint()
        # rot_mats = torch.eye(3)[None].repeat(len(rot_mats),1,1)
        # breakpoint()
        # bfs, but i think there's quicker ways
        bones = copy.deepcopy(self.b)
        q = [self.r]
        
        joint_res = torch.empty_like(self.j)
        joint_res[self.r] = self.j[self.r]
        # bone_res = torch.empty_like(self.b_vec)
        # bone_res[self.r] = self.b_vec[self.r]
        rot_res = torch.empty(len(self.j), 4).to(quat)
        rot_res[self.r] = 0.
        
        while len(q) > 0:
            f = q.pop(0)  
            todo, bones = find_new_q(bones, f)
            for it in todo:  # from f to it
                q.append(it)
                joint_res[it] = joint_res[f] + rot_mats[it] @ self.b_vec[it]
                # joint_res[it] = joint_res[f] + self.b_vec[it]
            
        assert len(bones) == 0
        
        return joint_res - self.j
    
    
def random_rotation_matrix():
    # Generate a random 3x3 orthogonal matrix
    orthogonal_matrix = np.random.randn(3, 3)
    q, _ = np.linalg.qr(orthogonal_matrix)

    # Ensure determinant is 1 to preserve orientation
    if np.linalg.det(q) < 0:
        q[:, 2] *= -1

    return q
    
    
if __name__ == '__main__':
    import numpy as np
    a = np.load('jnb.npy',allow_pickle=True).item()
    b = a['bones']
    j = a['joints']
    r = a['root_idx']
    rigmodel = RigModel(torch.from_numpy(j),
                        b,r)    
    # rot = torch.eye(3)[None].repeat(len(j),1,1)
    # rot[3] = torch.from_numpy(random_rotation_matrix())
    d_xyz = rigmodel.animate(torch.zeros((len(j),4)))
    # breakpoint()
    j = j + d_xyz.cpu().numpy()
    from cvtb import vis
    vis.pcd_static(j)
    # breakpoint()
