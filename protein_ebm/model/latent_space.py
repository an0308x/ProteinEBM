"""Latent space encoder/decoder for smoother EBM dynamics.

This module provides an autoencoder to map protein coordinates to a latent representation,
enabling more stable Langevin dynamics and efficient exploration.
"""

import torch
from torch import nn
from torch.nn import Module
from protein_ebm.model.boltz_utils import LinearNoBias


class ResidualBlock(Module):
    """Residual block with LayerNorm for better gradient flow."""

    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 2

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            LinearNoBias(dim, hidden_dim),
            nn.SiLU(),
            LinearNoBias(hidden_dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class LatentEncoder(Module):
    """Encodes protein coordinates into a smooth latent representation.

    Uses residual MLP blocks with LayerNorm for stable training and
    better gradient flow.

    Args:
        coord_dim: Dimension of input coordinates (3 for backbone, 111 for all atoms)
        latent_dim: Dimension of latent space (default: 128)
        num_blocks: Number of residual blocks (default: 3)
        hidden_multiplier: Hidden dimension multiplier (default: 2)
    """

    def __init__(
        self,
        coord_dim: int = 3,
        latent_dim: int = 128,
        num_blocks: int = 3,
        hidden_multiplier: int = 2,
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim

        # Initial projection
        self.proj_in = LinearNoBias(coord_dim, latent_dim)

        # Residual blocks for non-linear encoding
        self.blocks = nn.ModuleList([
            ResidualBlock(latent_dim, latent_dim * hidden_multiplier)
            for _ in range(num_blocks)
        ])

        # Final normalization
        self.norm_out = nn.LayerNorm(latent_dim)

    def forward(self, coords):
        """Encode coordinates to latent space.

        Args:
            coords: Tensor of shape [batch, res, coord_dim]

        Returns:
            latent: Tensor of shape [batch, res, latent_dim]
        """
        # Project to latent dimension
        x = self.proj_in(coords)

        # Apply residual blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm_out(x)

        return x


class LatentDecoder(Module):
    """Decodes latent representations back to protein coordinates.

    Mirrors the encoder architecture with residual blocks for stable
    reconstruction.

    Args:
        latent_dim: Dimension of latent space (default: 128)
        coord_dim: Dimension of output coordinates (3 for backbone, 111 for all atoms)
        num_blocks: Number of residual blocks (default: 3)
        hidden_multiplier: Hidden dimension multiplier (default: 2)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        coord_dim: int = 3,
        num_blocks: int = 3,
        hidden_multiplier: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.coord_dim = coord_dim

        # Residual blocks for non-linear decoding
        self.blocks = nn.ModuleList([
            ResidualBlock(latent_dim, latent_dim * hidden_multiplier)
            for _ in range(num_blocks)
        ])

        # Final projection to coordinate space
        self.norm_out = nn.LayerNorm(latent_dim)
        self.proj_out = LinearNoBias(latent_dim, coord_dim)

    def forward(self, latent):
        """Decode latent representation to coordinates.

        Args:
            latent: Tensor of shape [batch, res, latent_dim]

        Returns:
            coords: Tensor of shape [batch, res, coord_dim]
        """
        x = latent

        # Apply residual blocks
        for block in self.blocks:
            x = block(x)

        # Project back to coordinate space
        x = self.norm_out(x)
        coords = self.proj_out(x)

        return coords


class LatentAutoencoder(Module):
    """Combined encoder-decoder for latent space operations.

    This module handles the full encoding and decoding pipeline, making
    it easy to integrate with the EBM.

    Args:
        coord_dim: Dimension of coordinates (3 for backbone, 111 for all atoms)
        latent_dim: Dimension of latent space (default: 128)
        num_blocks: Number of residual blocks in each encoder/decoder (default: 3)
        hidden_multiplier: Hidden dimension multiplier (default: 2)
    """

    def __init__(
        self,
        coord_dim: int = 3,
        latent_dim: int = 128,
        num_blocks: int = 3,
        hidden_multiplier: int = 2,
    ):
        super().__init__()

        self.encoder = LatentEncoder(
            coord_dim=coord_dim,
            latent_dim=latent_dim,
            num_blocks=num_blocks,
            hidden_multiplier=hidden_multiplier,
        )

        self.decoder = LatentDecoder(
            latent_dim=latent_dim,
            coord_dim=coord_dim,
            num_blocks=num_blocks,
            hidden_multiplier=hidden_multiplier,
        )

        self.coord_dim = coord_dim
        self.latent_dim = latent_dim

    def encode(self, coords):
        """Encode coordinates to latent space."""
        return self.encoder(coords)

    def decode(self, latent):
        """Decode latent to coordinates."""
        return self.decoder(latent)

    def forward(self, coords):
        """Full autoencoder pass (for reconstruction loss).

        Args:
            coords: Tensor of shape [batch, res, coord_dim]

        Returns:
            reconstructed: Tensor of shape [batch, res, coord_dim]
            latent: Tensor of shape [batch, res, latent_dim]
        """
        latent = self.encode(coords)
        reconstructed = self.decode(latent)
        return reconstructed, latent

    def reconstruction_loss(self, coords, reconstructed, mask=None):
        """Compute MSE reconstruction loss.

        Args:
            coords: Original coordinates [batch, res, coord_dim]
            reconstructed: Reconstructed coordinates [batch, res, coord_dim]
            mask: Optional mask [batch, res] to ignore padding

        Returns:
            loss: Scalar reconstruction loss
        """
        mse = (coords - reconstructed) ** 2
        mse = mse.sum(dim=-1)  # Sum over coordinate dimension

        if mask is not None:
            mse = mse * mask
            loss = mse.sum() / mask.sum()
        else:
            loss = mse.mean()

        return loss
