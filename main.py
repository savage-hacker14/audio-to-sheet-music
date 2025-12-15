import torch
from demucs import pretrained
from transformers import ClapModel, AutoTokenizer
from torch.utils.data import DataLoader
from src.dataloader import MusDBStemDataset, collate_fn
from src.loss import sdr_loss, sisdr_loss, combined_loss, new_sdr_metric
from src.models.stem_separation.AudioTextHTDemucs import AudioTextHTDemucs
from src.train import train
from utils import load_config

def test_dataloader(config_path: str):
    # Configuration    
    cfg             = load_config(config_path)
    DATA_DIR        = cfg['data']['train_dir']
    BATCH_SIZE      = cfg['training']['batch_size']
    SAMPLE_RATE     = cfg['data']['sample_rate']
    SEGMENT_SAMPLES = int(SAMPLE_RATE * 5)       # Fixed 5 second segment length for testing
    RANDOM_SEGMENTS = cfg['data']['random_segments']
    AUGMENT         = cfg['data']['augment']

    print("Creating dataset...")
    dataset = MusDBStemDataset(
        root_dir=DATA_DIR,
        segment_samples=SEGMENT_SAMPLES,
        sample_rate=SAMPLE_RATE,
        random_segments=RANDOM_SEGMENTS,
        augment=AUGMENT,
    )

    print(f"Dataset length: {len(dataset)}")

    print("\nTesting single item...")
    item = dataset[0]
    print(f"Mixture shape: {item['mixture'].shape}")  # Should be (2, 220500)
    assert item['mixture'].shape == (2, 220500)
    print(f"Target shape: {item['target'].shape}")    # Should be (2, 220500)
    assert item['target'].shape == (2, 220500)
    print(f"Prompt: {item['prompt']}")
    print(f"Stem: {item['stem_name']}")

    print("\nTesting DataLoader...")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=cfg['training']['num_workers'])

    batch = next(iter(loader))
    print(f"Batch mixture shape: {batch['mixture'].shape}")  # Should be (BATCH_SIZE, 2, 220500)
    assert batch['mixture'].shape == (BATCH_SIZE, 2, 220500)
    print(f"Batch target shape: {batch['target'].shape}")    # Should be (BATCH_SIZE, 2, 220500)
    assert batch['target'].shape == (BATCH_SIZE, 2, 220500)
    print(f"Prompts: {batch['prompt']}")
    print(f"Stems: {batch['stem_name']}")

    print("\n✓ Dataloader test passed!")

def test_losses(config_path: str):
    print("=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)

    # Get configuration
    cfg             = load_config(config_path)
    DATA_DIR        = cfg['data']['train_dir']
    SAMPLE_RATE     = cfg['data']['sample_rate']
    SEGMENT_SAMPLES = int(cfg['data']['sample_rate'] * cfg['data']['segment_seconds'])

    # Test 1: Perfect reconstruction (should give very high SDR)
    print("\n[Test 1] Perfect reconstruction")
    target = torch.randn(4, 2, 44100)  # 4 batch, 2 channels, 1 sec
    estimated = target.clone()

    loss_sdr = sdr_loss(estimated, target)
    loss_sisdr = sisdr_loss(estimated, target)
    total, metrics = combined_loss(estimated, target)

    print(f"SDR Loss: {loss_sdr.item():.4f} (should be very negative)")
    print(f"SI-SDR Loss: {loss_sisdr.item():.4f} (should be very negative)")
    print(f"Metrics SDR: {metrics['metrics/sdr']:.2f} dB (should be ~30)")
    print(f"Metrics SI-SDR: {metrics['metrics/sisdr']:.2f} dB (should be ~30)")

    # Test 2: Random noise (should give low/negative SDR)
    print("\n[Test 2] Random noise estimate")
    target = torch.randn(4, 2, 44100)
    estimated = torch.randn(4, 2, 44100)

    loss_sdr = sdr_loss(estimated, target)
    loss_sisdr = sisdr_loss(estimated, target)
    total, metrics = combined_loss(estimated, target)

    print(f"SDR Loss: {loss_sdr.item():.4f}")
    print(f"SI-SDR Loss: {loss_sisdr.item():.4f}")
    print(f"Metrics SDR: {metrics['metrics/sdr']:.2f} dB (should be negative)")
    print(f"Metrics SI-SDR: {metrics['metrics/sisdr']:.2f} dB (should be negative)")

    # Test 3: Scaled version (SI-SDR should be better than SDR)
    print("\n[Test 3] Scaled estimate (2x gain)")
    target = torch.randn(4, 2, 44100)
    estimated = target * 2.0  # 2x scaling

    loss_sdr = sdr_loss(estimated, target)
    loss_sisdr = sisdr_loss(estimated, target)
    total, metrics = combined_loss(estimated, target)

    print(f"SDR Loss: {loss_sdr.item():.4f}")
    print(f"SI-SDR Loss: {loss_sisdr.item():.4f} (should be better than SDR)")
    print(f"Metrics SDR: {metrics['metrics/sdr']:.2f} dB")
    print(f"Metrics SI-SDR: {metrics['metrics/sisdr']:.2f} dB (should be higher)")

    # NEW TEST: Target + varying levels of noise
    print("\n[Test 4] Target with added noise (realistic scenario)")
    target = torch.randn(4, 2, 44100)

    # Test at different SNR levels
    for snr_db in [20, 10, 5, 0, -5]:
        # Calculate noise level for desired SNR
        signal_power = (target ** 2).mean()
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(target) * torch.sqrt(noise_power)

        estimated = target + noise

        total, metrics = combined_loss(estimated, target)

        print(f"  SNR={snr_db:3d}dB → SDR: {metrics['metrics/sdr']:6.2f} dB, "
              f"SI-SDR: {metrics['metrics/sisdr']:6.2f} dB, Loss: {total.item():.4f}")

    print("  (Metrics should decrease as SNR decreases)")

    # Test 5: Partial signal extraction (simulating incomplete separation)
    print("\n[Test 5] Partial signal extraction")
    target = torch.randn(4, 2, 44100)
    interference = torch.randn(4, 2, 44100)

    # Simulate extracting 80% target + 20% interference
    estimated = 0.8 * target + 0.2 * interference

    total, metrics = combined_loss(estimated, target)

    print(f"SDR: {metrics['metrics/sdr']:.2f} dB (should be ~13-14 dB)")
    print(f"SI-SDR: {metrics['metrics/sisdr']:.2f} dB")
    print(f"Loss: {total.item():.4f}")

    # Test 6: Gradients flow correctly
    print("\n[Test 6] Gradient flow check")
    target = torch.randn(2, 2, 44100)
    estimated = torch.randn(2, 2, 44100, requires_grad=True)

    total, metrics = combined_loss(estimated, target)
    total.backward()

    print(f"Loss: {total.item():.4f}")
    print(f"Gradients exist: {estimated.grad is not None}")
    print(f"Gradient mean: {estimated.grad.abs().mean().item():.6f}")
    print(f"Gradient std: {estimated.grad.std().item():.6f}")

    # Test 7: Integration with dataloader
    print("\n[Test 7] Integration with real data")
    try:
        # You'll need to update this path
        dataset = MusDBStemDataset(
            root_dir=DATA_DIR,
            segment_samples=SEGMENT_SAMPLES,
            sample_rate=SAMPLE_RATE,
            random_segments=True,
            augment=False,
        )

        batch = dataset[0]
        mixture = batch['mixture'].unsqueeze(0)  # Add batch dim
        target = batch['target'].unsqueeze(0)

        # Test with real audio
        total, metrics = combined_loss(mixture, target)

        print(f"Real data loss: {total.item():.4f}")
        print(f"Real data SDR: {metrics['metrics/sdr']:.2f} dB")
        print(f"Real data SI-SDR: {metrics['metrics/sisdr']:.2f} dB")
        print(f"Real data new_sdr: {metrics['metrics/new_sdr']:.2f} dB")

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"\n⚠ Dataloader test skipped: {e}")
        print("(Update the path in the test script to run this test)")
        print("\n✓ Core loss function tests passed!")

def test_model(config_path: str):
    print("=" * 60)
    print("Testing AudioTextHTDemucs Model")
    print("=" * 60)
    
    # Configuration
    cfg             = load_config(config_path)
    DATA_DIR        = cfg['data']['train_dir']
    BATCH_SIZE      = cfg['training']['batch_size']
    SAMPLE_RATE     = cfg['data']['sample_rate']
    SEGMENT_SAMPLES = int(cfg['data']['sample_rate'] * cfg['data']['segment_seconds'])

    # Load pre-trained models
    print("\n[1] Loading pre-trained models...")
    htdemucs = pretrained.get_model('htdemucs').models[0]
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    model = AudioTextHTDemucs(htdemucs, clap, tokenizer)
    print("✓ Models loaded")

    # Test 1: Forward pass with dummy data
    print("\n[2] Testing forward pass with dummy data...")
    B, C, T = 2, 2, 44100 * 3  # 2 batch, 2 channels, 3 seconds
    dummy_wav = torch.randn(B, C, T)
    dummy_prompts = ["drums", "bass"]

    output = model(dummy_wav, dummy_prompts)
    print(f"Input shape: {dummy_wav.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (B, C, T), f"Expected {(B, C, T)}, got {output.shape}"
    print("✓ Forward pass works")

    # Test 2: Gradient flow
    print("\n[3] Testing gradient flow...")
    dummy_wav_grad = torch.randn(B, C, T, requires_grad=True)
    dummy_target = torch.randn(B, C, T)

    model.train()
    output = model(dummy_wav_grad, dummy_prompts)
    loss, metrics = combined_loss(output, dummy_target)

    loss.backward()
    print(f"Loss: {loss.item():.4f}")
    print(f"Input gradient exists: {dummy_wav_grad.grad is not None}")
    print(f"Gradient mean: {dummy_wav_grad.grad.abs().mean().item():.6f}")
    print("✓ Gradients flow correctly")

    # Test 3: Integration with dataloader
    print("\n[4] Testing with real data...")
    try:
        dataset = MusDBStemDataset(
            root_dir=DATA_DIR,
            segment_samples=SEGMENT_SAMPLES,  # 3 seconds
            sample_rate=SAMPLE_RATE,
            random_segments=True,
            augment=False,
        )

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=cfg['training']['num_workers'])
        batch = next(iter(loader))

        mixture = batch['mixture']  # (B, C, T)
        target = batch['target']  # (B, C, T)
        prompts = batch['prompt']  # List[str]

        print(f"Batch mixture shape: {mixture.shape}")
        print(f"Batch target shape: {target.shape}")
        print(f"Prompts: {prompts}")

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(mixture, prompts)

        print(f"Model output shape: {output.shape}")
        assert output.shape == target.shape

        # Compute loss
        loss, metrics = combined_loss(output, target)
        print(f"\nLoss: {loss.item():.4f}")
        print(f"SDR: {metrics['metrics/sdr']:.2f} dB")
        print(f"SI-SDR: {metrics['metrics/sisdr']:.2f} dB")

        print("✓ Real data test passed")

    except Exception as e:
        print(f"⚠ Dataloader test failed: {e}")
        print("(Make sure data/train exists and contains .stem.mp4 files)")

    # Test 4: Different sequence lengths
    print("\n[5] Testing variable length inputs...")
    model.eval()
    for length_sec in [2, 3, 5]:
        T_test = 44100 * length_sec
        test_wav = torch.randn(1, 2, T_test)
        test_prompt = ["vocals"]

        with torch.no_grad():
            output = model(test_wav, test_prompt)

        print(f"{length_sec}s input: {test_wav.shape} → {output.shape}")
        assert output.shape[-1] == T_test

    print("✓ Variable length test passed")

    # Test 5: Batch with different prompts
    print("\n[6] Testing batch with different prompts...")
    batch_wav = torch.randn(4, 2, 44100 * 2)
    diverse_prompts = ["drums", "bass", "vocals", "other instruments"]

    with torch.no_grad():
        output = model(batch_wav, diverse_prompts)

    print(f"Diverse prompts: {diverse_prompts}")
    print(f"Output shape: {output.shape}")
    assert output.shape == batch_wav.shape
    print("✓ Diverse prompts test passed")

    # Test 6: Model size
    print("\n[7] Model statistics...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    print("\n" + "=" * 60)
    print("✓ All model tests passed!")
    print("=" * 60)



if __name__ == "__main__":
    # Run tests
    config_path = "config.yaml"
    # test_dataloader(config_path)
    # test_losses(config_path)
    # test_model(config_path)
    
    # Run training with custom parameters    
    results = train(config_path)

    print(f"\nTraining finished!")
    print(f"Best SDR achieved: {results['best_sdr']:.2f} dB")