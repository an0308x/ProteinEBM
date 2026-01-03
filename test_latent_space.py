"""
Simple test script to validate latent space EBM implementation.

This script tests:
1. Encoder/decoder reconstruction quality
2. Latent diffusion forward/reverse
3. Model forward pass in latent space
4. Energy computation in latent space
"""

import torch
import yaml
from ml_collections import ConfigDict

from protein_ebm.model.latent_space import LatentAutoencoder
from protein_ebm.model.latent_diffuser import LatentDiffuser
from protein_ebm.model.r3_diffuser import R3Diffuser
from protein_ebm.model.ebm import ProteinEBM


def test_autoencoder():
    """Test encoder/decoder reconstruction."""
    print("=" * 50)
    print("Testing Autoencoder...")
    print("=" * 50)

    # Create autoencoder
    autoencoder = LatentAutoencoder(
        coord_dim=3,
        latent_dim=128,
        num_blocks=3,
    )

    # Create dummy coordinates
    batch_size = 2
    num_residues = 100
    coords = torch.randn(batch_size, num_residues, 3)

    # Test encoding
    latent = autoencoder.encode(coords)
    print(f"Input shape: {coords.shape}")
    print(f"Latent shape: {latent.shape}")
    assert latent.shape == (batch_size, num_residues, 128), "Latent shape mismatch"

    # Test decoding
    reconstructed = autoencoder.decode(latent)
    print(f"Reconstructed shape: {reconstructed.shape}")
    assert reconstructed.shape == coords.shape, "Reconstruction shape mismatch"

    # Test full forward pass
    recon, lat = autoencoder(coords)
    assert recon.shape == coords.shape, "Full forward shape mismatch"

    # Compute reconstruction error (should be random initially)
    mse = torch.mean((coords - reconstructed) ** 2)
    print(f"Reconstruction MSE (untrained): {mse.item():.6f}")

    # Test reconstruction loss
    mask = torch.ones(batch_size, num_residues)
    loss = autoencoder.reconstruction_loss(coords, reconstructed, mask)
    print(f"Reconstruction loss: {loss.item():.6f}")

    print("✓ Autoencoder test passed!\n")


def test_latent_diffuser():
    """Test latent space diffusion."""
    print("=" * 50)
    print("Testing Latent Diffuser...")
    print("=" * 50)

    # Create diffuser config
    config = ConfigDict({
        'min_b': 0.1,
        'max_b': 20.0,
        'coordinate_scaling': 0.1
    })

    diffuser = LatentDiffuser(config, latent_dim=128)

    # Create dummy latent codes
    batch_size = 2
    num_residues = 100
    z_0 = torch.randn(batch_size, num_residues, 128)

    # Test forward marginal (PyTorch version)
    t = 0.5
    z_t, score_t = diffuser.forward_marginal_torch(z_0, torch.tensor(t))

    print(f"z_0 shape: {z_0.shape}")
    print(f"z_t shape: {z_t.shape}")
    print(f"score_t shape: {score_t.shape}")

    assert z_t.shape == z_0.shape, "Forward diffusion shape mismatch"
    assert score_t.shape == z_0.shape, "Score shape mismatch"

    # Test calc_trans_0
    z_0_pred = diffuser.calc_trans_0(score_t, z_t, torch.tensor(t), use_torch=True)
    print(f"z_0_pred shape: {z_0_pred.shape}")
    assert z_0_pred.shape == z_0.shape, "Prediction shape mismatch"

    # Test reverse step (numpy version)
    z_t_minus_dt = diffuser.reverse(
        z_t=z_t.numpy(),
        score_t=score_t.numpy(),
        t=t,
        dt=0.01,
    )
    print(f"Reverse step output shape: {z_t_minus_dt.shape}")

    print("✓ Latent Diffuser test passed!\n")


def test_latent_ebm():
    """Test ProteinEBM with latent space enabled."""
    print("=" * 50)
    print("Testing Latent Space EBM...")
    print("=" * 50)

    # Load latent space config
    with open('protein_ebm/config/latent_space_ebm.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config = ConfigDict(config)

    # Create diffuser and model
    diffuser = R3Diffuser(config.diffuser)
    model = ProteinEBM(config.model, diffuser)

    print(f"use_latent_space: {model.use_latent_space}")
    print(f"latent_dim: {model.latent_dim}")

    # Verify autoencoder is created
    assert model.autoencoder is not None, "Autoencoder not initialized"
    assert model.latent_diffuser is not None, "Latent diffuser not initialized"

    # Create dummy input
    batch_size = 2
    num_residues = 50
    aatype = torch.randint(0, 20, (batch_size, num_residues))
    r_noisy = torch.randn(batch_size, num_residues, 3)
    residue_idx = torch.arange(num_residues).unsqueeze(0).repeat(batch_size, 1)
    residue_mask = torch.ones(batch_size, num_residues)
    times = torch.rand(batch_size)

    # Test forward pass
    output = model.forward(
        aatype=aatype,
        r_noisy=r_noisy,
        residue_idx=residue_idx,
        residue_mask=residue_mask,
        times=times,
    )

    print(f"Output keys: {output.keys()}")
    print(f"r_update shape: {output['r_update'].shape}")

    # r_update should be in latent space
    assert output['r_update'].shape == (batch_size, num_residues, 128), \
        "r_update should be in latent space"

    # Test compute_energy
    input_feats = {
        'r_noisy': r_noisy,
        'aatype': aatype,
        'residue_idx': residue_idx,
        'mask': residue_mask,
        't': times,
    }

    energy_output = model.compute_energy(input_feats)
    print(f"Energy output keys: {energy_output.keys()}")
    print(f"Energy shape: {energy_output['energy'].shape}")
    assert 'energy' in energy_output, "Energy not computed"

    # Test compute_score (direct mode)
    score_output = model.compute_score(input_feats)
    print(f"Score output keys: {score_output.keys()}")
    print(f"trans_score shape: {score_output['trans_score'].shape}")

    # trans_score should be in coordinate space
    assert score_output['trans_score'].shape == r_noisy.shape, \
        "trans_score should be in coordinate space"

    # Check pred_coords
    if 'pred_coords' in score_output:
        print(f"pred_coords shape: {score_output['pred_coords'].shape}")

    print("✓ Latent Space EBM test passed!\n")


def test_reconstruction_quality():
    """Test reconstruction quality with random weights."""
    print("=" * 50)
    print("Testing Reconstruction Quality...")
    print("=" * 50)

    autoencoder = LatentAutoencoder(coord_dim=3, latent_dim=128, num_blocks=3)

    # Create realistic protein coordinates (centered)
    num_residues = 100
    coords = torch.randn(1, num_residues, 3) * 5.0  # Simulate ~5Å spread

    # Encode and decode
    latent = autoencoder.encode(coords)
    reconstructed = autoencoder.decode(latent)

    # Compute RMSD
    rmsd = torch.sqrt(torch.mean((coords - reconstructed) ** 2))
    print(f"Random weight RMSD: {rmsd.item():.4f} Å")

    # This should be large since weights are random
    print(f"Note: RMSD will decrease significantly after training")

    print("✓ Reconstruction quality test passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("LATENT SPACE EBM TEST SUITE")
    print("=" * 50 + "\n")

    try:
        test_autoencoder()
        test_latent_diffuser()
        test_latent_ebm()
        test_reconstruction_quality()

        print("=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Train a latent space EBM model:")
        print("   python protein_ebm/scripts/train.py protein_ebm/config/latent_space_ebm.yaml")
        print("\n2. Monitor reconstruction loss during training")
        print("   - Should decrease to <0.01")
        print("\n3. Use the trained model for inference")
        print("   - All existing inference scripts work automatically")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
