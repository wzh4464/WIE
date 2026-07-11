import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read data
tim_df = pd.read_csv("tim_cleansing_performance_001.csv")
sgd_df = pd.read_csv("sgd_cleansing_performance_001.csv")
relabel_df = pd.read_csv("relabel_030_pct_metrics_001.csv")

# Academic style color scheme
academic_colors = {
    "blue": "#1f77b4",
    "brown": "#8c564b",
    "red": "#d62728",
}

# Shift epoch to the right by 1
tim_df["epoch"] += 1
sgd_df["epoch"] += 1

# Get the maximum epoch to unify the horizontal axis
max_epoch = int(
    max(tim_df["epoch"].max(), sgd_df["epoch"].max(), relabel_df["epoch"].max())
)

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(
    tim_df["epoch"],
    tim_df["test_accuracy"],
    label="WIE per epoch",
    color=academic_colors["blue"],
    marker="s",
    markersize=10,
    linestyle="-",
)

plt.plot(
    sgd_df["epoch"],
    sgd_df["test_accuracy"],
    label="SGD influence",
    color=academic_colors["brown"],
    marker="v",
    markersize=10,
    linestyle="-",
)

plt.plot(
    relabel_df["epoch"],
    relabel_df["val_accuracy"],
    label="Train with corrupted data",
    color=academic_colors["red"],
    marker="o",
    markersize=10,
    linestyle="-",
)

# Axis and legend settings
plt.xlabel("Epoch", fontsize=28)
plt.ylabel("Test Accuracy", fontsize=28)
plt.legend(fontsize=18)
plt.xticks(np.arange(3, max_epoch + 1, 2), fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(left=3)
plt.ylim(bottom=0.5)

plt.grid(True)
plt.tight_layout()

# Save image
plt.savefig("accuracy_comparison_academic_colors.png", dpi=300)
