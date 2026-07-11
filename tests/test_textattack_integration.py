#!/usr/bin/env python3
"""
Test script to verify TextAttack-style training integration
"""

import os
import sys
import subprocess
import tempfile


def test_textattack_training():
    """Test the TextAttack-style training integration."""
    print("Testing TextAttack-style training integration...")

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # Test command (with minimal parameters)
        test_cmd = [
            "python",
            "-m",
            "wie.training.train_textattack_style",
            "--target",
            "sentiment",
            "--model",
            "bert",
            "--save_dir",
            temp_dir,
            "--relabel",
            "0",
            "--seed",
            "42",
            "--gpu",
            "0",
            "--log_level",
            "INFO",
            "--num_epoch",
            "1",  # Just 1 epoch for testing
            "--batch_size",
            "4",  # Small batch size
            "--lr",
            "2e-5",
        ]

        print(f"Running command: {' '.join(test_cmd)}")

        try:
            # Run the training command
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                print("✓ Training completed successfully!")
                print("Output:", result.stdout[-500:])  # Last 500 chars

                # Check if output files were created
                expected_files = [
                    "relabel_000_pct_global_info_042.json",
                    "relabel_000_pct_042.dat",
                    "logs/train.log",
                ]

                for file_path in expected_files:
                    full_path = os.path.join(temp_dir, file_path)
                    if os.path.exists(full_path):
                        print(f"✓ Created file: {file_path}")
                    else:
                        print(f"✗ Missing file: {file_path}")

                return True
            else:
                print("✗ Training failed!")
                print("Return code:", result.returncode)
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False

        except subprocess.TimeoutExpired:
            print("✗ Training timed out!")
            return False
        except Exception as e:
            print(f"✗ Training failed with exception: {e}")
            return False


def test_epoch_wise_script():
    """Test the epoch-wise keep ratio script with TextAttack integration."""
    print("\nTesting epoch-wise keep ratio script...")

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # Test command (with minimal parameters)
        test_cmd = [
            "python",
            "-m",
            "scripts.epoch_wise_keep_ratio",
            "--target",
            "sentiment",
            "--model",
            "bert",
            "--save_dir",
            temp_dir,
            "--relabel",
            "0",
            "--seed",
            "42",
            "--gpu",
            "0",
            "--log_level",
            "INFO",
            "--num_epoch",
            "1",  # Just 1 epoch for testing
            "--batch_size",
            "4",  # Small batch size
            "--lr",
            "2e-5",
            "--keep_ratio",
            "90",
        ]

        print(f"Running command: {' '.join(test_cmd)}")

        try:
            # Run the script (this will run training, influence calculation, and cleansing)
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode == 0:
                print("✓ Epoch-wise script completed successfully!")
                print("Output:", result.stdout[-500:])  # Last 500 chars
                return True
            else:
                print("✗ Epoch-wise script failed!")
                print("Return code:", result.returncode)
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False

        except subprocess.TimeoutExpired:
            print("✗ Epoch-wise script timed out!")
            return False
        except Exception as e:
            print(f"✗ Epoch-wise script failed with exception: {e}")
            return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TEXTATTACK INTEGRATION TESTS")
    print("=" * 60)

    # Test 1: Direct training
    test1_passed = test_textattack_training()

    # Test 2: Epoch-wise script (only if test 1 passed)
    test2_passed = False
    if test1_passed:
        test2_passed = test_epoch_wise_script()
    else:
        print("\nSkipping epoch-wise script test due to training failure")

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"TextAttack Training: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Epoch-wise Script:   {'✓ PASSED' if test2_passed else '✗ FAILED'}")

    if test1_passed and test2_passed:
        print("\n✓ All tests passed! Integration is working correctly.")
        print("\nYou can now run:")
        print(
            "  pixi run python -m scripts.epoch_wise_keep_ratio --target sentiment --model bert --save_dir sentiment_bert_wie_textattack --relabel 30 --seed 0 --gpu 1 --type wie_all_epochs --lr 2e-5 --log_level INFO --num_epoch 20"
        )
    else:
        print("\n✗ Some tests failed. Please check the errors above.")

    return 0 if (test1_passed and test2_passed) else 1


if __name__ == "__main__":
    sys.exit(main())
