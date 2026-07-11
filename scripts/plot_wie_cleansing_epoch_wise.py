import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import argparse

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 22
matplotlib.rcParams["axes.titlesize"] = 26
matplotlib.rcParams["axes.labelsize"] = 24
matplotlib.rcParams["xtick.labelsize"] = 22
matplotlib.rcParams["ytick.labelsize"] = 22
matplotlib.rcParams["legend.fontsize"] = 16
matplotlib.rcParams["figure.titlesize"] = 28


def plot_single_folder(
    folder_path, relabel, keep_ratio=None, seed=0, infl_type="wie_all_epochs"
):
    """Process plotting for a single folder"""
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a directory")
        return

    plt.figure(figsize=(10, 7))

    min_acc = None  # Track dynamic minimum across plotted curves
    line_count = 0  # Counter to track the number of plotted curves

    # Plot Full Training curve
    relabel_csv = os.path.join(
        folder_path, f"relabel_{relabel:03d}_pct_metrics_{seed:03d}.csv"
    )
    if os.path.exists(relabel_csv):
        df_relabel = pd.read_csv(relabel_csv)
        df_relabel = df_relabel[df_relabel["epoch"] >= 1]
        df_relabel["epoch"] = df_relabel["epoch"].astype(int)
        epochs = df_relabel["epoch"].tolist()
        # Prefer validation accuracy for full training
        accs = (
            df_relabel["val_accuracy"].tolist()
            if "val_accuracy" in df_relabel.columns
            else df_relabel["test_accuracy"].tolist()
        )
        if accs:
            min_acc = min(accs) if min_acc is None else min(min_acc, min(accs))
            plt.plot(
                epochs,
                accs,
                marker="s",
                color="#0000FF",
                linestyle="--",
                label="Full Training",
            )
            line_count += 1

    # Plot WIE cleansing curve
    if infl_type == "wie_all_epochs":
        tim_csv = os.path.join(folder_path, f"tim_cleansing_performance_{seed:03d}.csv")
    elif infl_type == "sgd":
        tim_csv = os.path.join(folder_path, f"tim_cleansing_performance_{seed:03d}.csv")
    else:
        raise ValueError(f"Unsupported influence type: {infl_type}")

    if not os.path.exists(tim_csv):
        print(f"Warning: Missing file {tim_csv}")
        if line_count == 0:
            print("Warning: No curves to plot")
            return

    df_tim = pd.read_csv(tim_csv)
    # WIE performance is recorded as 0..N-1, shift to 1..N
    df_tim["epoch"] = df_tim["epoch"].astype(int) + 1
    df_tim = df_tim[df_tim["epoch"] >= 1]
    epochs = df_tim["epoch"].tolist()
    accs = df_tim["test_accuracy"].tolist()
    if accs:
        min_acc = min(accs) if min_acc is None else min(min_acc, min(accs))

    # If keep_ratio is provided, use it directly; otherwise, extract from folder name
    if keep_ratio is None:
        folder_name = os.path.basename(folder_path)
        try:
            keep_ratio = int(folder_name.split("_")[-1])
        except (IndexError, ValueError):
            print(
                f"Warning: Cannot extract keep_ratio from folder name {folder_name}, using default value 80"
            )
            keep_ratio = 80

    plt.plot(epochs, accs, marker="o", label=f"Prune: -{100 - keep_ratio}%/epoch")
    line_count += 1

    if line_count == 1:
        print("Warning: Only one curve found, possible lack of comparison data")

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.xlim(left=1)
    if min_acc is not None:
        plt.ylim(bottom=max(0.0, min_acc - 0.01), top=1)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Let MaxNLocator choose integer ticks; no fixed [0,5,10,...]

    # Extract decay, learning rate, and training parameter information from the folder path
    folder_name = os.path.basename(folder_path)
    decay = os.path.basename(os.path.dirname(folder_path))
    lr = folder_name.split("_")[-1] if "lr_" in folder_name else "default"

    # Build output filename
    out_name = f"all_epoch_accuracy_curves_{decay}_relabel_{relabel:03d}_seed_{seed}_lr_{lr}_type_{infl_type}"

    # Add training parameter information to the filename
    if "ntr_" in folder_name:
        n_tr = folder_name.split("ntr_")[1].split("_")[0]
        out_name += f"_ntr_{n_tr}"
    if "nval_" in folder_name:
        n_val = folder_name.split("nval_")[1].split("_")[0]
        out_name += f"_nval_{n_val}"

    out_name += ".png"
    plt.savefig(out_name, dpi=200)
    plt.close()
    print(f"Saved: {out_name}")


def process_all_folders(
    base_dir, decay_options, relabels, seed=0, infl_type="wie_all_epochs"
):
    """Process all folders matching the pattern"""
    for decay in decay_options:
        decay_dir = os.path.join(base_dir, decay)
        # Automatically collect all keep_ratio folders
        all_folders = [
            f for f in os.listdir(decay_dir) if f.startswith("epochwise_relabel_")
        ]
        # Extract all keep_ratios and relabels
        folder_info = []
        for folder in all_folders:
            # Folder name format: epochwise_relabel_XXX_keep_ratio_YYY
            parts = folder.split("_")
            relabel = int(parts[2])
            keep_ratio = int(parts[-1])
            folder_info.append((relabel, keep_ratio, os.path.join(decay_dir, folder)))

        for relabel in relabels:
            # Select only all keep_ratios for the current relabel
            relabel_folders = [
                (kr, path) for r, kr, path in folder_info if r == relabel
            ]
            if not relabel_folders:
                print(f"Skip relabel={relabel} in {decay_dir}, no data.")
                continue

            plt.figure(figsize=(10, 7))
            min_acc = 1.0  # Initialize minimum accuracy

            # Plot Full Training curve
            # Choose any keep_ratio folder
            relabel_folder = relabel_folders[0][1]
            relabel_csv = os.path.join(
                relabel_folder, f"relabel_{relabel:03d}_pct_metrics_{seed:03d}.csv"
            )
            if os.path.exists(relabel_csv):
                df_relabel = pd.read_csv(relabel_csv)
                df_relabel = df_relabel[df_relabel["epoch"] >= 1]
                df_relabel["epoch"] = df_relabel["epoch"].astype(int)
                epochs = df_relabel["epoch"].tolist()
                accs = (
                    df_relabel["val_accuracy"].tolist()
                    if "val_accuracy" in df_relabel.columns
                    else df_relabel["test_accuracy"].tolist()
                )
                if accs:
                    min_acc = min(accs) if min_acc is None else min(min_acc, min(accs))
                    plt.plot(
                        epochs,
                        accs,
                        marker="s",
                        color="#0000FF",
                        linestyle="--",
                        label="Full Training",
                    )

            # Plot WIE cleansing curve
            for keep_ratio, folder in sorted(relabel_folders):
                if infl_type == "wie_all_epochs":
                    tim_csv = os.path.join(
                        folder, f"tim_cleansing_performance_{seed:03d}.csv"
                    )
                elif infl_type == "sgd":
                    tim_csv = os.path.join(
                        folder, f"tim_cleansing_performance_{seed:03d}.csv"
                    )
                else:
                    raise ValueError(f"Unsupported influence type: {infl_type}")

                if not os.path.exists(tim_csv):
                    print(f"Warning: Missing file {tim_csv}")
                    continue
                df_tim = pd.read_csv(tim_csv)
                df_tim["epoch"] = df_tim["epoch"].astype(int) + 1
                df_tim = df_tim[df_tim["epoch"] >= 1]
                epochs = df_tim["epoch"].tolist()
                accs = df_tim["test_accuracy"].tolist()
                if accs:
                    min_acc = min(accs) if min_acc is None else min(min_acc, min(accs))
                    plt.plot(
                        epochs,
                        accs,
                        marker="o",
                        label=f"Prune: -{100 - keep_ratio}%/epoch",
                    )

            plt.xlabel("Epoch")
            plt.ylabel("Test Accuracy")
            plt.xlim(left=1)
            if min_acc is not None:
                plt.ylim(bottom=max(0.0, min_acc - 0.01), top=1)
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            # Use integer ticks chosen by locator; no fixed [0,5,10,...]
            out_name = f"all_epoch_accuracy_curves_{decay}_relabel_{relabel:03d}_seed_{seed}_type_{infl_type}.png"
            plt.savefig(out_name, dpi=200)
            plt.close()
            print(f"Saved: {out_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot WIE cleansing epoch-wise results"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="experiment",
        help="Base directory containing experiment results",
    )
    parser.add_argument(
        "--decay",
        type=str,
        nargs="+",
        default=["test_result"],
        help="Decay options to process",
    )
    parser.add_argument(
        "--relabel", type=int, nargs="+", default=[20], help="Relabel values to process"
    )
    parser.add_argument(
        "--single_folder",
        type=str,
        help="Process a single folder instead of all folders",
    )
    parser.add_argument(
        "--relabel_value", type=int, help="Relabel value for single folder mode"
    )
    parser.add_argument(
        "--keep_ratio", type=int, help="Keep ratio for single folder mode"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed value for the experiment (default: 0)"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="wie_all_epochs",
        choices=["wie_all_epochs", "sgd"],
        help="Influence type: wie_all_epochs or sgd",
    )

    args = parser.parse_args()

    if args.single_folder:
        # If relabel_value is provided, use it directly
        if args.relabel_value is not None:
            plot_single_folder(
                args.single_folder,
                args.relabel_value,
                args.keep_ratio,
                args.seed,
                args.type,
            )
            return

        # Otherwise, try to extract relabel from the folder name
        folder_name = os.path.basename(args.single_folder)
        try:
            relabel = int(folder_name.split("_")[2])
            plot_single_folder(
                args.single_folder, relabel, args.keep_ratio, args.seed, args.type
            )
        except (IndexError, ValueError):
            print("Error: Cannot extract relabel value from folder name")
            print(
                "Please use the --relabel_value parameter to specify the relabel value, for example:"
            )
            print(
                "python plot_wie_cleansing_epoch_wise.py --single_folder <folder_path> --relabel_value 20"
            )
    else:
        process_all_folders(
            args.base_dir, args.decay, args.relabel, args.seed, args.type
        )


if __name__ == "__main__":
    main()
