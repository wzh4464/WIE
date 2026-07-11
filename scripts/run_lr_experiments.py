#!/usr/bin/env python3
import subprocess
import os
import argparse


def run_experiment(lr, n_tr=None, n_val=None):
    # Create save directory
    save_dir = f"test_result_lr_{lr:.3f}"
    if n_tr is not None:
        save_dir += f"_ntr_{n_tr}"
    if n_val is not None:
        save_dir += f"_nval_{n_val}"

    # Run experiment
    cmd = [
        "pixi",
        "run",
        "python",
        "-m",
        "scripts.epoch_wise_keep_ratio",
        "--target",
        "mnist",
        "--model",
        "dnn",
        "--save_dir",
        save_dir,
        "--relabel",
        "30",
        "--seed",
        "0",
        "--decay",
        "True",
        "--keep_ratio",
        "80",
        "--lr",
        str(lr),
        "--seed",
        "1",
    ]

    # Add custom training parameters
    if n_tr is not None:
        cmd.extend(["--n_tr", str(n_tr)])
    if n_val is not None:
        cmd.extend(["--n_val", str(n_val)])

    print(f"Running experiment with lr={lr}, n_tr={n_tr}, n_val={n_val}")
    subprocess.run(cmd, check=True)

    # Run plotting
    plot_cmd = [
        "pixi",
        "run",
        "python",
        "-m",
        "scripts.plot_wie_cleansing_epoch_wise",
        "--single_folder",
        os.path.join("experiment", save_dir),
        "--relabel_value",
        "30",
        "--keep_ratio",
        "80",
        "--seed",
        "1",
    ]

    print(f"Plotting results for lr={lr}, n_tr={n_tr}, n_val={n_val}")
    subprocess.run(plot_cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments with different learning rates and training parameters"
    )
    parser.add_argument("--n_tr", type=int, help="Number of training samples")
    parser.add_argument("--n_val", type=int, help="Number of validation samples")
    args = parser.parse_args()

    # Define the list of learning rates to test
    lr_list = [0.1, 0.08, 0.06, 0.04, 0.02, 0.01]

    for lr in lr_list:
        run_experiment(lr, args.n_tr, args.n_val)


if __name__ == "__main__":
    main()
