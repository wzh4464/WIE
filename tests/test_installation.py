#!/usr/bin/env python
# ruff: noqa: F401
"""
Test script to verify the installation and basic functionality
"""

import sys
import os


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from wie.training.config import TrainingConfig
        from wie.training.trainer import Trainer

        print("✓ Training modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import training modules: {e}")
        return False

    try:
        from wie.data.imdb import IMDBDataset, IMDBDataModule

        print("✓ Data modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import data modules: {e}")
        return False

    try:
        from wie.models.bert_sentiment import BertForSentiment, create_model_and_tokenizer

        print("✓ Model modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import model modules: {e}")
        return False

    return True


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")

    try:
        from wie.training.config import TrainingConfig

        # Test default config
        config = TrainingConfig()
        print(
            f"✓ Default config created: {config.num_epochs} epochs, lr={config.learning_rate}"
        )

        # Test YAML loading
        if os.path.exists("configs/imdb_bert_base.yaml"):
            config = TrainingConfig.from_yaml("configs/imdb_bert_base.yaml")
            print("✓ YAML config loaded successfully")

        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_dependencies():
    """Test that required dependencies are installed."""
    print("\nTesting dependencies...")

    required = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "datasets": "Datasets",
        "sklearn": "Scikit-learn",
        "tqdm": "TQDM",
        "yaml": "PyYAML",
    }

    all_installed = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name} is installed")
        except ImportError:
            print(f"✗ {name} is not installed (pip install {module})")
            all_installed = False

    # Check optional dependencies
    print("\nOptional dependencies:")
    optional = {"tensorboard": "TensorBoard", "wandb": "Weights & Biases"}

    for module, name in optional.items():
        try:
            __import__(module)
            print(f"✓ {name} is installed")
        except ImportError:
            print(f"○ {name} is not installed (optional)")

    return all_installed


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
            cuda_version = getattr(getattr(torch, "version", object()), "cuda", None)
            print(f"  CUDA version: {cuda_version if cuda_version else 'unknown'}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
        else:
            print("○ CUDA is not available (CPU training will be used)")
        return True
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("BERT Sentiment Pipeline - Installation Test")
    print("=" * 60)

    tests = [
        ("Dependencies", test_dependencies),
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("CUDA", test_cuda),
    ]

    results = []
    for name, test_func in tests:
        success = test_func()
        results.append((name, success))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name:20} {status}")
        if not success:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed! The pipeline is ready to use.")
        print("\nQuick start command:")
        print("  python -m wie.cli.train_imdb --config configs/imdb_bert_base.yaml")
    else:
        print("\n✗ Some tests failed. Please install missing dependencies.")
        print("\nMinimal installation:")
        print("  pip install torch transformers datasets scikit-learn tqdm pyyaml")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
