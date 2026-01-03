"""Latent space diffuser for VP-SDE in learned representations.

This module extends the R3Diffuser to operate in latent space, enabling
smoother diffusion dynamics and more stable Langevin sampling.
"""

import numpy as np
import torch
from protein_ebm.model.r3_diffuser import R3Diffuser


class LatentDiffuser:
    """VP-SDE diffuser for latent space representations.

    This wraps the standard R3Diffuser but operates on latent vectors
    instead of 3D coordinates. The diffusion equations are the same,
    but the interpretation is different.

    Args:
        config: Diffuser configuration with min_b, max_b, etc.
        latent_dim: Dimension of latent space
    """

    def __init__(self, config, latent_dim: int = 128):
        self.config = config
        self.latent_dim = latent_dim
        self.min_b = config.min_b
        self.max_b = config.max_b

        # Note: We don't use coordinate_scaling for latent space
        # since latent vectors are already normalized

    def b_t(self, t):
        """Variance schedule."""
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        return self.min_b + t * (self.max_b - self.min_b)

    def marginal_b_t(self, t):
        """Marginal variance at time t."""
        return t * self.min_b + (1/2) * (t**2) * (self.max_b - self.min_b)

    def diffusion_coef(self, t):
        """Time-dependent diffusion coefficient."""
        return np.sqrt(self.b_t(t))

    def drift_coef(self, x, t):
        """Time-dependent drift coefficient."""
        return -1/2 * self.b_t(t) * x

    def conditional_var(self, t, use_torch=False):
        """Conditional variance of p(zt|z0)."""
        if use_torch:
            return 1 - torch.exp(-self.marginal_b_t(t))
        return 1 - np.exp(-self.marginal_b_t(t))

    def forward_marginal(self, z_0, t):
        """Sample from marginal p(z(t) | z(0)).

        Args:
            z_0: [..., n, latent_dim] initial latent codes
            t: continuous time in [0, 1]

        Returns:
            z_t: [..., n, latent_dim] latent codes at time t
            score_t: [..., n, latent_dim] score at time t
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')

        # Sample noisy latent
        z_t = np.random.normal(
            loc=np.exp(-1/2 * self.marginal_b_t(t)) * z_0,
            scale=np.sqrt(1 - np.exp(-self.marginal_b_t(t)))
        )

        # Compute score
        score_t = self.score(z_t, z_0, t)

        return z_t, score_t

    def forward_marginal_torch(self, z_0, t):
        """PyTorch version of forward_marginal for use in training.

        Args:
            z_0: [..., n, latent_dim] initial latent codes (torch tensor)
            t: continuous time in [0, 1] (torch tensor or scalar)

        Returns:
            z_t: [..., n, latent_dim] latent codes at time t
            score_t: [..., n, latent_dim] score at time t
        """
        # Handle both scalar and tensor t
        if torch.is_tensor(t):
            if t.ndim == 0:
                t_val = t.item()
            else:
                # For batched t, we need to handle it differently
                # Expand marginal_b_t to match batch dimensions
                beta_t = self.marginal_b_t(t.cpu().numpy())
                beta_t = torch.tensor(beta_t, device=z_0.device, dtype=z_0.dtype)
                # Reshape for broadcasting
                while beta_t.ndim < z_0.ndim:
                    beta_t = beta_t.unsqueeze(-1)
        else:
            beta_t = self.marginal_b_t(t)
            beta_t = torch.tensor(beta_t, device=z_0.device, dtype=z_0.dtype)

        # Sample noise
        noise = torch.randn_like(z_0)

        # Forward diffusion
        mean_coef = torch.exp(-1/2 * beta_t)
        std = torch.sqrt(1 - torch.exp(-beta_t))

        z_t = mean_coef * z_0 + std * noise

        # Compute score
        score_t = -(z_t - mean_coef * z_0) / (1 - torch.exp(-beta_t))

        return z_t, score_t

    def score(self, z_t, z_0, t, use_torch=False):
        """Compute the score function.

        Args:
            z_t: Noisy latent at time t
            z_0: Clean latent
            t: Time value
            use_torch: Whether to use PyTorch operations

        Returns:
            score: Score function ∇_z log p(z_t | z_0)
        """
        if use_torch:
            exp_fn = torch.exp
        else:
            exp_fn = np.exp

        beta_t = self.marginal_b_t(t)
        return -(z_t - exp_fn(-1/2 * beta_t) * z_0) / self.conditional_var(t, use_torch=use_torch)

    def calc_trans_0(self, score_t, z_t, t, use_torch=True):
        """Predict z_0 from z_t and score.

        Args:
            score_t: Score at time t
            z_t: Latent at time t
            t: Time value
            use_torch: Whether to use PyTorch

        Returns:
            z_0: Predicted clean latent
        """
        beta_t = self.marginal_b_t(t)

        # Handle tensor t
        if use_torch and torch.is_tensor(t):
            beta_t = torch.tensor(beta_t, device=z_t.device, dtype=z_t.dtype)
            # Reshape for broadcasting
            beta_t = beta_t.view((beta_t.shape[0],) + (1,) * (len(score_t.shape) - 1))
            exp_fn = torch.exp
        elif use_torch:
            beta_t = torch.tensor(beta_t, device=z_t.device, dtype=z_t.dtype)
            exp_fn = torch.exp
        else:
            exp_fn = np.exp

        cond_var = 1 - exp_fn(-beta_t)
        z_0 = (score_t * cond_var + z_t) / exp_fn(-1/2 * beta_t)

        return z_0

    def reverse(
        self,
        *,
        z_t,
        score_t,
        t: float,
        dt: float,
        mask=None,
        noise_scale: float = 1.0,
    ):
        """Simulate reverse SDE for 1 step in latent space.

        Args:
            z_t: [..., latent_dim] current latent at time t
            score_t: [..., latent_dim] score at time t
            t: continuous time in [0, 1]
            dt: continuous step size in [0, 1]
            mask: Optional mask for selective diffusion
            noise_scale: Scale for noise term

        Returns:
            z_t_1: [..., latent_dim] latent at time t-dt
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')

        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(z_t, t)

        # Random noise
        z = noise_scale * np.random.normal(size=score_t.shape)

        # Reverse step
        perturb = (f_t - g_t**2 * score_t) * dt + g_t * np.sqrt(dt) * z

        if mask is not None:
            perturb *= mask[..., None]

        z_t_1 = z_t - perturb

        return z_t_1

    def distribution(self, z_t, score_t, t, mask, dt):
        """Compute distribution parameters for reverse step.

        Args:
            z_t: Latent at time t
            score_t: Score at time t
            t: Time value
            mask: Mask tensor
            dt: Time step

        Returns:
            mu: Mean of reverse distribution
            std: Standard deviation of reverse distribution
        """
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(z_t, t)
        std = g_t * np.sqrt(dt)
        mu = z_t - (f_t - g_t**2 * score_t) * dt

        if mask is not None:
            mu *= mask[..., None]

        return mu, std
