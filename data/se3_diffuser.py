"""SE(3) diffusion methods."""
import numpy as np
from data import so3_diffuser
from data import r3_diffuser
from data import embedding_diffuser
from scipy.spatial.transform import Rotation
from openfold.utils import rigid_utils as ru
from data import utils as du
import torch
import logging

def _extract_trans_rots(rigid: ru.Rigid):
    rot = rigid.get_rots().get_rot_mats().cpu().numpy()
    rot_shape = rot.shape
    num_rots = np.cumprod(rot_shape[:-2])[-1]
    rot = rot.reshape((num_rots, 3, 3))
    rot = Rotation.from_matrix(rot).as_rotvec().reshape(rot_shape[:-2] +(3,))
    tran = rigid.get_trans().cpu().numpy()
    return tran, rot

def _assemble_rigid(rotvec, trans):
    rotvec_shape = rotvec.shape
    num_rotvecs = np.cumprod(rotvec_shape[:-1])[-1]
    rotvec = rotvec.reshape((num_rotvecs, 3))
    rotmat = Rotation.from_rotvec(rotvec).as_matrix().reshape(
        rotvec_shape[:-1] + (3, 3))
    return ru.Rigid(
            rots=ru.Rotation(
                rot_mats=torch.Tensor(rotmat)),
            trans=torch.tensor(trans))

class SE3Diffuser:

    def __init__(self, se3_conf):
        self._log = logging.getLogger(__name__)
        self._se3_conf = se3_conf

        self._diffuse_rot = se3_conf.diffuse_rot
        self._so3_diffuser = so3_diffuser.SO3Diffuser(self._se3_conf.so3)

        self._diffuse_trans = se3_conf.diffuse_trans
        self._r3_diffuser = r3_diffuser.R3Diffuser(self._se3_conf.r3)
        
        self._diffuse_embed = se3_conf.diffuse_embed
        self._embed_diffuser = embedding_diffuser.EmbeDiffuser(self._se3_conf.embed)
        

    def forward_marginal(
            self,
            rigids_0: ru.Rigid,
            representation: torch.tensor,
            t: float,
            diffuse_mask: np.ndarray = None,
            motif_seq_mask: np.ndarray = None,
            as_tensor_7: bool=True,
        ):
        """
        Args:
            rigids_0: [..., N] openfold Rigid objects
            t: continuous time in [0, 1].

        Returns:
            rigids_t: [..., N] noised rigid. [..., N, 7] if as_tensor_7 is true. 
            trans_score: [..., N, 3] translation score
            rot_score: [..., N, 3] rotation score
            trans_score_norm: [...] translation score norm
            rot_score_norm: [...] rotation score norm
        """
        trans_0, rot_0 = _extract_trans_rots(rigids_0)

        if not self._diffuse_rot:
            rot_t, rot_score, rot_score_scaling = (
                rot_0,
                np.zeros_like(rot_0),
                np.ones_like(t)
            )
        else:
            rot_t, rot_score, rot_cause_trans = self._so3_diffuser.forward_marginal(
                rot_0, t, motif_seq_mask, trans_0=trans_0)
            rot_score_scaling = self._so3_diffuser.score_scaling(t)

        if not self._diffuse_trans:
            trans_t, trans_score, trans_score_scaling = (
                trans_0,
                np.zeros_like(trans_0),
                np.ones_like(t)
            )
        else:
            trans_t, trans_score = self._r3_diffuser.forward_marginal(
                trans_0, t, motif_seq_mask, rot_cause_trans)
            trans_score_scaling = self._r3_diffuser.score_scaling(t)
            trans_t = trans_t + rot_cause_trans
        one_hot = self._embed_diffuser.forward_marginal(representation, t, motif_seq_mask)

        if diffuse_mask is not None:
            # diffuse_mask = torch.tensor(diffuse_mask).to(rot_t.device)
            rot_t = self._apply_mask(
                rot_t, rot_0, diffuse_mask[..., None])
            trans_t = self._apply_mask(
                trans_t, trans_0, diffuse_mask[..., None])

            trans_score = self._apply_mask(
                trans_score,
                np.zeros_like(trans_score),
                diffuse_mask[..., None])
            rot_score = self._apply_mask(
                rot_score,
                np.zeros_like(rot_score),
                diffuse_mask[..., None])
        rigids_t = _assemble_rigid(rot_t, trans_t)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        return {
            'rigids_t': rigids_t,
            'one_hot': one_hot,
            'trans_score': trans_score,
            'rot_score': rot_score,
            'trans_score_scaling': trans_score_scaling,
            'rot_score_scaling': rot_score_scaling,
        }

    def calc_trans_0(self, trans_score, trans_t, t):
        return self._r3_diffuser.calc_trans_0(trans_score, trans_t, t)

    def calc_trans_score(self, trans_t, trans_0, t, use_torch=False, scale=True):
        return self._r3_diffuser.score(
            trans_t, trans_0, t, use_torch=use_torch, scale=scale)

    def calc_rot_score(self, rots_t, rots_0, t, t_trans=None, motif_mask=None):
        rots_0_inv = rots_0.invert()
        quats_0_inv = rots_0_inv.get_quats()
        quats_t = rots_t.get_quats()
        quats_0t = ru.quat_multiply(quats_0_inv, quats_t)
        rotvec_0t = du.quat_to_rotvec(quats_0t)
        if t_trans is not None:
            rotation_to_trans = ru.Rotation(quats=quats_0t, normalize_quats=False)
            bs, num_motif, seq_len = motif_mask.shape
            seq_mask = torch.ones_like(motif_mask[:,0,:])
            rot_cause_trans = torch.zeros_like(t_trans)
            for i in range(num_motif):
                seq_mask *= (~motif_mask[:,i,:])
                center_m = torch.mean(t_trans * motif_mask[:,i].unsqueeze(-1), -2)
                trans_from_center = (t_trans - center_m.unsqueeze(1)) * motif_mask[:,i].unsqueeze(-1)
                #trans_from_center = t_trans
                trans_for_update = rotation_to_trans.invert_apply(trans_from_center)
                rot_cause_trans += ((trans_for_update + center_m.unsqueeze(1)) * motif_mask[:,i].unsqueeze(-1))
                #rot_cause_trans += ((trans_for_update) * motif_mask[:,i].unsqueeze(-1))
            #new_translation = self._trans * seq_mask.unsqueeze(-1) + trans_update *(~seq_mask.unsqueeze(-1))
            rot_increase_trans = (rot_cause_trans - t_trans) * motif_mask[:,i].unsqueeze(-1)
            return self._so3_diffuser.torch_score(rotvec_0t, t), rot_increase_trans
        return self._so3_diffuser.torch_score(rotvec_0t, t)

    def _apply_mask(self, x_diff, x_fixed, diff_mask):
        return diff_mask * x_diff + (1 - diff_mask) * x_fixed

    def trans_parameters(self, trans_t, score_t, t, dt, mask):
        return self._r3_diffuser.distribution(
            trans_t, score_t, t, dt, mask)

    def score(
            self,
            rigid_0: ru.Rigid,
            rigid_t: ru.Rigid,
            t: float):
        tran_0, rot_0 = _extract_trans_rots(rigid_0)
        tran_t, rot_t = _extract_trans_rots(rigid_t)

        if not self._diffuse_rot:
            rot_score = np.zeros_like(rot_0)
        else:
            rot_score = self._so3_diffuser.score(
                rot_t, t)

        if not self._diffuse_trans:
            trans_score = np.zeros_like(tran_0)
        else:
            trans_score = self._r3_diffuser.score(tran_t, tran_0, t)

        return trans_score, rot_score

    def score_scaling(self, t):
        rot_score_scaling = self._so3_diffuser.score_scaling(t)
        trans_score_scaling = self._r3_diffuser.score_scaling(t)
        return rot_score_scaling, trans_score_scaling

    def reverse(
            self,
            rigid_t: ru.Rigid,
            rot_score: np.ndarray,
            trans_score: np.ndarray,
            t: float,
            dt: float,
            diffuse_mask: np.ndarray = None,
            center: bool=True,
            noise_scale: float=1.0,
            motif_mask: np.array=None,
            rot_increase_trans: np.array=None,
        ):
        """Reverse sampling function from (t) to (t-1).

        Args:
            rigid_t: [..., N] protein rigid objects at time t.
            rot_score: [..., N, 3] rotation score.
            trans_score: [..., N, 3] translation score.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: [..., N] which residues to update.
            center: true to set center of mass to zero after step

        Returns:
            rigid_t_1: [..., N] protein rigid objects at time t-1.
        """
        trans_t, rot_t = _extract_trans_rots(rigid_t)
        if not self._diffuse_rot:
            rot_t_1 = rot_t
        else:
            rot_t_1, trans_from_rot, seq_mask = self._so3_diffuser.reverse(
                rot_t=rot_t,
                score_t=rot_score,
                t=t,
                dt=dt,
                #noise_scale=noise_scale,
                noise_scale=1.0,
                motif_mask=motif_mask,
                trans_t = trans_t,
                )
        if not self._diffuse_trans:
            trans_t_1 = trans_t
        else:
            trans_t_1 = self._r3_diffuser.reverse(
                x_t=trans_t,
                score_t=trans_score,
                t=t,
                dt=dt,
                center=center,
                noise_scale=noise_scale,
                motif_mask=motif_mask,
                rot_increase_trans=rot_increase_trans,
                )

        if diffuse_mask is not None:
            trans_t_1 = self._apply_mask(
                trans_t_1, trans_t, diffuse_mask[..., None])
            rot_t_1 = self._apply_mask(
                rot_t_1, rot_t, diffuse_mask[..., None])
        return _assemble_rigid(rot_t_1, trans_t_1+trans_from_rot)

    def apply_motif_mask(self, motif_seq_mask, rigid_t, motif_rigids_0):
        # (2, 108) torch.Size([108, 7]) (2, 80, 7)
        if motif_seq_mask.shape[-1] != rigid_t.shape[0]:
            start_index = 0
            for i in range(motif_seq_mask.shape[0]):
                length = motif_seq_mask[i,:].sum()
                rigid_t[start_index:start_index+length,:] = motif_rigids_0[i,start_index:start_index+length,:]
                start_index = start_index + length
            return rigid_t

        rigid_t = rigid_t.numpy()
        motif_seq_mask = motif_seq_mask
        seq_mask = np.ones(motif_seq_mask.shape[-1], dtype=np.bool_)
        return_rigid = np.zeros_like(rigid_t) #(60,7)
        for i in range(motif_seq_mask.shape[0]):
            seq_mask = seq_mask*(~motif_seq_mask[i,:])
            length = motif_seq_mask[i,:].sum()
            return_rigid[motif_seq_mask[i,:],:] = motif_rigids_0[i,:length,:]
        return_rigid[seq_mask,:] = rigid_t[seq_mask,:]
        return return_rigid

    def apply_embed_mask(self, motif_seq_mask, rigid_t, motif_rigids_0):
        if motif_seq_mask.shape[-1] != rigid_t.shape[0]:
            start_index = 0
            for i in range(motif_seq_mask.shape[0]):
                length = motif_seq_mask[i,:].sum()
                rigid_t[start_index:start_index+length,:] = motif_rigids_0[i,start_index:start_index+length,:]
                start_index = start_index + length
            return rigid_t

        seq_mask = np.ones(motif_seq_mask.shape[-1], dtype=np.bool_)
        return_rigid = np.zeros_like(rigid_t) #(60,7)
        for i in range(motif_seq_mask.shape[0]):
            seq_mask = seq_mask*(~motif_seq_mask[i,:])
            length = motif_seq_mask[i,:].sum()
            return_rigid[motif_seq_mask[i,:],:] = motif_rigids_0[i,:length,:]
        return_rigid[seq_mask,:] = rigid_t[seq_mask,:]
        return return_rigid


    def sample_ref(
            self,
            n_samples: int,
            impute: ru.Rigid=None,
            diffuse_mask: np.ndarray=None,
            as_tensor_7: bool=False,
            motif_dict: dict=None,
            motif_mask: np.array=None,
            scale: float=None,
        ):
        """Samples rigids from reference distribution.

        Args:
            n_samples: Number of samples.
            impute: Rigid objects to use as imputation values if either
                translations or rotations are not diffused.
        """
        if impute is not None:
            assert impute.shape[0] == n_samples
            trans_impute, rot_impute = _extract_trans_rots(impute)
            trans_impute = trans_impute.reshape((n_samples, 3))
            rot_impute = rot_impute.reshape((n_samples, 3))
            trans_impute = self._r3_diffuser._scale(trans_impute)

        if diffuse_mask is not None and impute is None:
            raise ValueError('Must provide imputation values.')

        if (not self._diffuse_rot) and impute is None:
            raise ValueError('Must provide imputation values.')

        if (not self._diffuse_trans) and impute is None:
            raise ValueError('Must provide imputation values.')

        if self._diffuse_rot:
            rot_ref = self._so3_diffuser.sample_ref(
                n_samples=n_samples)#, motif_mask=motif_mask)
        else:
            rot_ref = rot_impute

        if self._diffuse_trans:
            trans_ref = self._r3_diffuser.sample_ref(
                n_samples=n_samples, motif_mask=motif_mask, scale=scale
            )
        else:
            trans_ref = trans_impute
        
        one_hot = self._embed_diffuser.sample_ref(n_samples=n_samples)
        # (2, 60) torch.Size([60, 22]) (60, 22)
        one_hot = self.apply_embed_mask(motif_dict['motif_seq_masks'], one_hot, np.eye(22)[motif_dict['motif_aatype']])

        if diffuse_mask is not None:
            rot_ref = self._apply_mask(
                rot_ref, rot_impute, diffuse_mask[..., None])
            trans_ref = self._apply_mask(
                trans_ref, trans_impute, diffuse_mask[..., None])
        trans_ref = self._r3_diffuser._unscale(trans_ref)
        rigids_t = _assemble_rigid(rot_ref, trans_ref)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
            rigids_t = self.apply_motif_mask(motif_dict['motif_seq_masks'], rigids_t, motif_dict['motif_rigids_0'])
        return {'rigids_t': rigids_t,
                'one_hot': one_hot}
