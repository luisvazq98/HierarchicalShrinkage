import matplotlib.pyplot as plt
import numpy as np

# Dataset labels
datasets = [
    "CIFAR-10", "Fashion-MNIST", "Oxford Pets",
    "Adult Income", "Titanic", "Credit Card",
    "Student Dropout", "Student Perf.", "Gait Data",
    "GitHub MUSAE", "Internet Ads"
]

# Accuracy values
dt =         [27.85, 79.04, 26.50, 80.93, 77.87, 72.33, 74.35, 89.56, 40.28, 71.02, 89.23]
pca_dt =     [24.45, 74.95, 23.75, 77.54, 66.43, 69.97, 70.93, 85.40, 33.02, 69.33, 86.33]
hs_dt =      [27.01, 78.06, 27.85, 85.33, 82.12, 82.07, 74.58, 93.11, 43.48, 77.38, 91.89]
pca_hs_dt =  [25.95, 73.86, 24.88, 82.12, 73.08, 78.18, 72.13, 90.05, 37.39, 75.29, 87.89]

x = np.arange(len(datasets))
width = 0.35

# --- Plot 1: DT vs PCA-DT ---
plt.figure(figsize=(12, 5))
plt.bar(x - width/2, dt, width, label='DT', color='#1f77b4')
plt.bar(x + width/2, pca_dt, width, label='PCA-DT', color='#ff7f0e')
plt.xticks(x, datasets, rotation=45, ha='right')
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("dt")
plt.show()

# --- Plot 2: HS-DT vs PCA-HS-DT ---
plt.figure(figsize=(12, 5))
plt.bar(x - width/2, hs_dt, width, label='HS-DT', color='#2ca02c')
plt.bar(x + width/2, pca_hs_dt, width, label='PCA-HS-DT', color='#d62728')
plt.xticks(x, datasets, rotation=45, ha='right')
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("hs")
plt.show()

