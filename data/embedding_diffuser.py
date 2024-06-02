"""One-hot embedding diffusion methods."""
import numpy as np
from scipy.special import gamma
import torch


class EmbeDiffuser:
    """VP-SDE diffuser class for translations."""
    def __init__(self, emb_conf):
        """
        Args:
            min_b: starting value in variance schedule.
            max_b: ending value in variance schedule.
        """
        self._emb_conf = emb_conf
        self.min_b = emb_conf.min_b
        self.max_b = emb_conf.max_b


    def marginal_b_t(self, t):
        return t*self.min_b + (1/2)*(t**2)*(self.max_b-self.min_b)

    def apply_motif_mask(self, motif_seq_mask, sampled_rots):
        sequence_mask = np.ones((motif_seq_mask.shape[-1]))
        result_noise = np.zeros_like(sampled_rots)
        for i in range(motif_seq_mask.shape[0]):
            sequence_mask = sequence_mask*(~motif_seq_mask[i,:])
            result_noise += (motif_seq_mask[i,:][:, np.newaxis])*(((np.sum(sampled_rots*motif_seq_mask[i,:][:, np.newaxis], 0))/np.sum(motif_seq_mask[i,:]))[np.newaxis, :])
        result_noise = sequence_mask[:, np.newaxis]*sampled_rots + result_noise
        return result_noise

    def forward_marginal(self, x_0: np.ndarray, t: float, motif_seq_mask: np.ndarray):
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        alpha = -1/2*self.marginal_b_t(t)
        beta = 1 - np.exp(-self.marginal_b_t(t))
        
        noise = np.random.normal(
            #loc=np.exp(alpha) * x_0,
            loc = 0,
            scale=np.sqrt(beta),
            size=x_0.shape
        )
        sequence_mask = np.ones((motif_seq_mask.shape[-1]))
        for i in range(motif_seq_mask.shape[0]):
            sequence_mask = sequence_mask*(~motif_seq_mask[i,:])
      
        x_t = x_0 + sequence_mask[:, np.newaxis]*noise
        return x_t

    def score_scaling(self, t: float):
        return 1 / np.sqrt(self.conditional_var(t))

    def reverse(
            self,
            *,
            x_t: np.ndarray,
            score_t: np.ndarray,
            t: float,
            dt: float,
            mask: np.ndarray=None,
            center: bool=True,
            noise_scale: float=1.0,
        ):
        """Simulates the reverse SDE for 1 step

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] positions at next step t-1.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        z = noise_scale * np.random.normal(size=score_t.shape)
        perturb = (f_t - g_t**2 * score_t) * dt + g_t * np.sqrt(dt) * z

        if mask is not None:
            perturb *= mask[..., None]
        else:
            mask = np.ones(x_t.shape[:-1])
        x_t_1 = x_t - perturb
        if center:
            com = np.sum(x_t_1, axis=-2) / np.sum(mask, axis=-1)[..., None]
            x_t_1 -= com[..., None, :]
        x_t_1 = self._unscale(x_t_1)
        return x_t_1

    def conditional_var(self, t, use_torch=False):
        """Conditional variance of p(xt|x0).

        Var[x_t|x_0] = conditional_var(t)*I

        """
        if use_torch:
            return 1 - torch.exp(-self.marginal_b_t(t))
        return 1 - np.exp(-self.marginal_b_t(t))

    def score(self, x_t, x_0, t, use_torch=False, scale=False):
        if use_torch:
            exp_fn = torch.exp
        else:
            exp_fn = np.exp
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)
        return -(x_t - exp_fn(-1/2*self.marginal_b_t(t)) * x_0) / self.conditional_var(t, use_torch=use_torch)

    def sample_ref(self, n_samples: float=1):
        return np.random.normal(size=(n_samples, 22))