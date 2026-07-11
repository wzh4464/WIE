# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 22
matplotlib.rcParams["axes.titlesize"] = 26
matplotlib.rcParams["axes.labelsize"] = 24
matplotlib.rcParams["xtick.labelsize"] = 22
matplotlib.rcParams["ytick.labelsize"] = 22
matplotlib.rcParams["legend.fontsize"] = 16
matplotlib.rcParams["figure.titlesize"] = 28

# %%
base_dir = "experiment/result"
tim_folder = "tim_epoch_wise_cleaning_lr_0_15_n_tr_2000"
sgd_folder = "sgd_epoch_wise_cleaning_lr_0_15_n_tr_2000"
seed = 1
relabel = 30
keep_ratio = 70

# %%
plt.figure(figsize=(10, 7))
min_acc = 0.4

# Plot Full Training curve (using data from WIE folder)
relabel_csv = os.path.join(
    base_dir, tim_folder, f"relabel_{relabel:03d}_pct_metrics_{seed:03d}.csv"
)
print(f"Checking file: {relabel_csv}")
if os.path.exists(relabel_csv):
    df_relabel = pd.read_csv(relabel_csv)
    print("Relabel data:")
    print(df_relabel.head())
    df_relabel = df_relabel[df_relabel["epoch"] >= 1]
    df_relabel["epoch"] = df_relabel["epoch"].astype(int)
    epochs = [0] + df_relabel["epoch"].tolist()
    accs = [0.49609375] + df_relabel["val_accuracy"].tolist()
    min_acc = min(min_acc, min(accs))
    plt.plot(
        epochs,
        accs,
        marker="s",
        color="black",
        linestyle="--",
        label="Training with corrupted data",
    )

# Plot WIE cleansing curve
tim_csv = os.path.join(
    base_dir, tim_folder, f"tim_cleansing_performance_{seed:03d}.csv"
)
print(f"Checking file: {tim_csv}")
if os.path.exists(tim_csv):
    df_tim = pd.read_csv(tim_csv)
    print("WIE data:")
    print(df_tim.head())
    df_tim["epoch"] = df_tim["epoch"] + 1
    df_tim = df_tim[df_tim["epoch"] >= 1]
    df_tim["epoch"] = df_tim["epoch"].astype(int)
    epochs = [0] + df_tim["epoch"].tolist()
    accs = [0.49609375] + df_tim["test_accuracy"].tolist()
    min_acc = min(min_acc, min(accs))
    plt.plot(
        epochs,
        accs,
        marker="o",
        color="red",
        label=f"WIE cleansing (keep {keep_ratio}%)",
    )

# Plot SGD cleansing curve
sgd_csv = os.path.join(
    base_dir, sgd_folder, f"tim_cleansing_performance_{seed:03d}.csv"
)
print(f"Checking file: {sgd_csv}")
if os.path.exists(sgd_csv):
    df_sgd = pd.read_csv(sgd_csv)
    print("SGD data:")
    print(df_sgd.head())
    df_sgd["epoch"] = df_sgd["epoch"] + 1
    df_sgd = df_sgd[df_sgd["epoch"] >= 1]
    df_sgd["epoch"] = df_sgd["epoch"].astype(int)
    epochs = [0] + df_sgd["epoch"].tolist()
    accs = [0.49609375] + df_sgd["test_accuracy"].tolist()
    min_acc = min(min_acc, min(accs))
    plt.plot(
        epochs,
        accs,
        marker="^",
        color="blue",
        label=f"SGD cleansing (keep {keep_ratio}%)",
    )

plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.xlim(left=0)
plt.ylim(bottom=min_acc - 0.01, top=1)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks([0, 5, 10, 15, 20])

# Save image
out_name = f"compare_wie_sgd_cleansing_lr_0_15_n_tr_2000_relabel_{relabel}_keep_{keep_ratio}_seed_{seed}.png"
plt.savefig(out_name, dpi=200)
plt.close()
print(f"Saved: {out_name}")

# %%
