import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 22
matplotlib.rcParams["axes.titlesize"] = 26
matplotlib.rcParams["axes.labelsize"] = 24
matplotlib.rcParams["xtick.labelsize"] = 22
matplotlib.rcParams["ytick.labelsize"] = 22
matplotlib.rcParams["legend.fontsize"] = 16
matplotlib.rcParams["figure.titlesize"] = 28

keep_ratios = [98, 96, 94, 92, 90]
folders = [
    f"experiment/epoch_wise_cleansing_keep_ratio_{ratio:03d}" for ratio in keep_ratios
]

final_accs = []
for folder in folders:
    tim_csv = os.path.join(folder, "tim_cleansing_performance_000.csv")
    if not os.path.exists(tim_csv):
        print(f"Warning: Missing file in {folder}")
        final_accs.append(None)
        continue
    df = pd.read_csv(tim_csv)
    acc = df["test_accuracy"].iloc[-1]
    final_accs.append(acc)

# Get the final accuracy of Full Training
full_train_acc = None
for folder in folders:
    relabel_csv = os.path.join(folder, "relabel_0_pct_metrics_000.csv")
    if os.path.exists(relabel_csv):
        df_relabel = pd.read_csv(relabel_csv)
        full_train_acc = df_relabel["test_accuracy"].iloc[-1]
        break

plt.figure(figsize=(8, 6))

# Bar chart
bar_positions = range(len(keep_ratios))
plt.bar(bar_positions, final_accs, width=0.6, color="#FF7F0E", label="Prune with WIE")
plt.xticks(bar_positions, [f"{r}%" for r in keep_ratios])
plt.xlabel("Keep Ratio")
plt.ylabel("Final Test Accuracy")

# Full Training dashed line
if full_train_acc is not None:
    plt.axhline(
        full_train_acc,
        color="#0000FF",
        linestyle="--",
        linewidth=2,
        label="Full Training",
    )

plt.ylim(bottom=0.49609375, top=1)
plt.legend()
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("final_accuracy_bar.png", dpi=200)
# plt.show()
