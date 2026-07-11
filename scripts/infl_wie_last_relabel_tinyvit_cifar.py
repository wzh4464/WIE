import os
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor

# Set paths
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
# WORK_DIR = WORK_DIR/..
WORK_DIR = os.path.join(WORK_DIR, "..")
PYTHON_ENV = os.path.join(WORK_DIR, ".venv/bin/python")

# Define configurations
NETWORKS = ["tinyvit_cifar"]
RELABELS = [6, 8, 2, 4, 10, 15, 20]


def setup_directories(network, relabel):
    """Set up and create necessary directories for the experiment."""
    # Set directory names
    hessian_dir = f"Hessian_8000_{network}"
    dest_dir = f"{hessian_dir}_cleansing_plots"
    save_dir = f"{hessian_dir}/cifar_{network}_wie_last_tracin_relabel_{relabel}_10e-1"

    # Also create a path for the experiment directory
    sec71_save_dir = os.path.join("experiment", "Sec71", save_dir)

    # Create destination directory
    os.makedirs(dest_dir, exist_ok=True)

    print(f"Save directory: {save_dir}")
    print(f"Absolute save directory: {os.path.abspath(save_dir)}")
    print(f"Sec71 save directory: {sec71_save_dir}")
    print(f"Absolute Sec71 save directory: {os.path.abspath(sec71_save_dir)}")

    # Check if the directory exists
    if not os.path.exists(save_dir):
        print(f"Warning: Directory does not exist: {save_dir}")
        print(f"Creating directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)

    # Check if the Sec71 directory exists
    if not os.path.exists(sec71_save_dir):
        print(f"Warning: Sec71 directory does not exist: {sec71_save_dir}")
        print(f"Creating directory: {sec71_save_dir}")
        os.makedirs(sec71_save_dir, exist_ok=True)

    # List files in the directory for debugging
    print(f"Files in directory {save_dir}:")
    if os.path.exists(save_dir):
        for file in os.listdir(save_dir):
            print(f"  - {file}")
    else:
        print("  Directory does not exist")

    # List files in the Sec71 directory for debugging
    print(f"Files in Sec71 directory {sec71_save_dir}:")
    if os.path.exists(sec71_save_dir):
        for file in os.listdir(sec71_save_dir):
            print(f"  - {file}")
    else:
        print("  Directory does not exist")

    return hessian_dir, dest_dir, save_dir, sec71_save_dir


def run_training(network, gpu, save_dir, relabel):
    """Run the training process."""
    print("Running training...")
    try:
        subprocess.run(
            [
                PYTHON_ENV,
                "-m",
                "wie.training.train",
                "--target",
                "cifar",
                "--model",
                network,
                "--gpu",
                str(gpu),
                "--seed",
                "0",
                "--save_dir",
                save_dir,
                "--relabel",
                str(relabel),
                "--no-loo",
                "--steps-only",
            ],
            check=True,
            env={**os.environ, "PYTHONPATH": WORK_DIR},
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running training: {str(e)}")
        print("Continuing with next steps...")
        return False


def run_wie_last_influence(network, gpu, save_dir, relabel):
    """Run the wie_last influence computation."""
    print("Running tim last influence computation...")
    try:
        subprocess.run(
            [
                PYTHON_ENV,
                "-m",
                "wie.infl",
                "--target",
                "cifar",
                "--model",
                network,
                "--gpu",
                str(gpu),
                "--seed",
                "0",
                "--save_dir",
                save_dir,
                "--relabel",
                str(relabel),
                "--type",
                "wie_last",
                "--length",
                "1",
            ],
            check=True,
            env={**os.environ, "PYTHONPATH": WORK_DIR},
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running tim last influence computation: {str(e)}")
        print("Continuing with next steps...")
        return False


def run_tracin_influence(network, gpu, save_dir, relabel):
    """Run the tracin influence computation."""
    print("Running tracin influence computation...")
    try:
        subprocess.run(
            [
                PYTHON_ENV,
                "-m",
                "wie.infl",
                "--target",
                "cifar",
                "--model",
                network,
                "--gpu",
                str(gpu),
                "--seed",
                "0",
                "--save_dir",
                save_dir,
                "--relabel",
                str(relabel),
                "--type",
                "tracin",
            ],
            check=True,
            env={**os.environ, "PYTHONPATH": WORK_DIR},
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running tracin influence computation: {str(e)}")
        print("Continuing with next steps...")
        return False


def generate_cleansing_plots(save_dir, sec71_save_dir, relabel):
    """Generate cleansing plots."""
    print(f"Generating cleansing plots for: {save_dir}")
    try:
        # Try with the original save_dir first
        subprocess.run(
            [
                PYTHON_ENV,
                "-m",
                "experiment.cleansing_plot",
                "--save_dir",
                save_dir,
                "--relabel",
                str(relabel),
            ],
            check=True,
            env={**os.environ, "PYTHONPATH": WORK_DIR},
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating cleansing plots with original save_dir: {str(e)}")
        print("Trying with Sec71 save_dir...")
        try:
            subprocess.run(
                [
                    PYTHON_ENV,
                    "-m",
                    "experiment.cleansing_plot",
                    "--save_dir",
                    sec71_save_dir,
                    "--relabel",
                    str(relabel),
                ],
                check=True,
                env={**os.environ, "PYTHONPATH": WORK_DIR},
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error generating cleansing plots with Sec71 save_dir: {str(e)}")
            print("Continuing with next steps...")
            return False


def copy_plots(hessian_dir, dest_dir, save_dir, sec71_save_dir, relabel):
    """Copy generated plots to the destination directory."""
    # Create flattened name for the plots
    flattened_name = save_dir.replace(f"{hessian_dir}/", "").replace("/", "_")

    # Copy cleansing plots
    for plot_name in [
        f"cleansing_plot_{relabel}_pct.png",
        f"cleansing_plot_{relabel}_pct_decending.png",
    ]:
        # Try original save_dir first
        src_path = os.path.join(save_dir, plot_name)
        if os.path.exists(src_path):
            print(f"Copying {plot_name} from: {save_dir}")
            dst_path = os.path.join(dest_dir, f"{flattened_name}_{plot_name}")
            subprocess.run(["cp", src_path, dst_path], check=True)
        else:
            print(f"No {plot_name} found in: {save_dir}")
            # Try Sec71 save_dir
            sec71_src_path = os.path.join(sec71_save_dir, plot_name)
            if os.path.exists(sec71_src_path):
                print(f"Copying {plot_name} from: {sec71_save_dir}")
                dst_path = os.path.join(dest_dir, f"{flattened_name}_{plot_name}")
                subprocess.run(["cp", sec71_src_path, dst_path], check=True)
            else:
                print(f"No {plot_name} found in: {sec71_save_dir}")
                # Try alternative plot names
                alt_plot_names = [
                    "cleansing_plot.png",
                    f"cleansing_plot_{relabel}.png",
                    f"cleansing_plot_{relabel}_pct.png",
                ]

                for alt_plot_name in alt_plot_names:
                    # Try original save_dir
                    alt_src_path = os.path.join(save_dir, alt_plot_name)
                    print(f"Trying alternative plot: {alt_plot_name} in {save_dir}")
                    if os.path.exists(alt_src_path):
                        print(
                            f"Copying alternative plot {alt_plot_name} from: {save_dir}"
                        )
                        dst_path = os.path.join(
                            dest_dir, f"{flattened_name}_{plot_name}"
                        )
                        subprocess.run(["cp", alt_src_path, dst_path], check=True)
                        break

                    # Try Sec71 save_dir
                    sec71_alt_src_path = os.path.join(sec71_save_dir, alt_plot_name)
                    print(
                        f"Trying alternative plot: {alt_plot_name} in {sec71_save_dir}"
                    )
                    if os.path.exists(sec71_alt_src_path):
                        print(
                            f"Copying alternative plot {alt_plot_name} from: {sec71_save_dir}"
                        )
                        dst_path = os.path.join(
                            dest_dir, f"{flattened_name}_{plot_name}"
                        )
                        subprocess.run(["cp", sec71_alt_src_path, dst_path], check=True)
                        break
                else:
                    print("No alternative plot found in either directory")

    # Copy pairwise comparison plots with TracIn
    pairwise_plot_names = [
        f"cleansing_plot_{relabel}_pct_tracin_vs_wie_last_asc.png",
        f"cleansing_plot_{relabel}_pct_tracin_vs_wie_last_desc.png",
    ]

    print("Copying pairwise comparison plots with TracIn")
    for plot_name in pairwise_plot_names:
        # Try original save_dir first
        src_path = os.path.join(save_dir, plot_name)
        if os.path.exists(src_path):
            print(f"Copying {plot_name} from: {save_dir}")
            dst_path = os.path.join(dest_dir, f"{flattened_name}_{plot_name}")
            subprocess.run(["cp", src_path, dst_path], check=True)
        else:
            print(f"No {plot_name} found in: {save_dir}")
            # Try Sec71 save_dir
            sec71_src_path = os.path.join(sec71_save_dir, plot_name)
            if os.path.exists(sec71_src_path):
                print(f"Copying {plot_name} from: {sec71_save_dir}")
                dst_path = os.path.join(dest_dir, f"{flattened_name}_{plot_name}")
                subprocess.run(["cp", sec71_src_path, dst_path], check=True)
            else:
                print(f"No {plot_name} found in: {sec71_save_dir}")


def run_configuration(config_idx, gpu_ids):
    try:
        # Calculate indices
        relabels_count = len(RELABELS)
        network_idx = (config_idx - 1) // relabels_count
        relabel_idx = (config_idx - 1) % relabels_count

        # Get configuration values
        network = NETWORKS[network_idx]
        relabel = RELABELS[relabel_idx]

        # Set GPU
        gpu = gpu_ids[(config_idx - 1) % len(gpu_ids)]

        print(f"\nStarting configuration {config_idx}")
        print(f"Using GPU: {gpu}")
        print(f"Network: {network}, Relabel: {relabel}")

        # Set up directories
        hessian_dir, dest_dir, save_dir, sec71_save_dir = setup_directories(
            network, relabel
        )

        # Run training
        run_training(network, gpu, save_dir, relabel)

        # Run tim last influence computation
        run_wie_last_influence(network, gpu, save_dir, relabel)

        # Run tracin influence computation
        run_tracin_influence(network, gpu, save_dir, relabel)

        # Generate cleansing plots
        generate_cleansing_plots(save_dir, sec71_save_dir, relabel)

        # Copy plots
        copy_plots(hessian_dir, dest_dir, save_dir, sec71_save_dir, relabel)

        print(f"Configuration {config_idx} completed successfully")
        return True

    except Exception as e:
        print(f"Error in configuration {config_idx}: {str(e)}")
        # Don't raise the exception, just log it and continue with other configurations
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_workers", type=int, default=3, help="Number of parallel workers"
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        required=True,
        help="Comma-separated list of GPU IDs to use (e.g., '0,2,4')",
    )
    args = parser.parse_args()

    print(f"Starting script with GPU IDs: {args.gpu_ids}")

    # Parse GPU IDs
    gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(",")]
    print(f"Parsed GPU IDs: {gpu_ids}")

    # Calculate total number of configurations
    total_configs = len(NETWORKS) * len(RELABELS)
    print(f"Total configurations to run: {total_configs}")

    # Run configurations in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Create a list of configuration indices
        config_indices = list(range(1, total_configs + 1))
        # Submit each configuration to the executor
        futures = [
            executor.submit(run_configuration, idx, gpu_ids) for idx in config_indices
        ]
        # Wait for all futures to complete
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in future execution: {str(e)}")
                results.append(False)

    # Print summary of results
    successful = sum(bool(r) for r in results)
    print(
        f"\nExecution completed. {successful}/{len(results)} configurations completed successfully."
    )


if __name__ == "__main__":
    main()
