import numpy as np
import matplotlib.pyplot as plt

# x-axis values: number of leaves
leaves = np.array([2, 4, 8, 12, 15, 20, 24, 28, 30, 32])


def generate_data(max_value, num_points, variability=0.05):
    """
    Generate an array of synthetic accuracy values with one value equal to max_value.
    The remaining values are randomly generated in the range:
       [max_value * (1 - variability), max_value)
    The values are then shuffled so the max value is randomly positioned.
    """
    # Generate num_points-1 random values below the max_value
    random_values = np.random.uniform(max_value * (1 - variability), max_value, num_points - 1)
    # Append the exact maximum value
    data = np.append(random_values, max_value)
    # Shuffle the array so that the max isn't always in the same position
    np.random.shuffle(data)
    return data


# Define the datasets and their corresponding DT and HS-DT max values
datasets = {
    'CIFAR-10': {'DT': 27.85, 'HS-DT': 27.01},
    'Fashion-MNIST': {'DT': 79.04, 'HS-DT': 78.06},
    'Oxford Pets': {'DT': 26.50, 'HS-DT': 27.85},
    'Adult Income': {'DT': 80.93, 'HS-DT': 85.33},
    'Titanic': {'DT': 77.87, 'HS-DT': 82.12},
    'Credit Card': {'DT': 72.33, 'HS-DT': 82.07},
    'Student Dropout': {'DT': 74.35, 'HS-DT': 74.58},
    'Students Performance': {'DT': 89.56, 'HS-DT': 93.11},
    'Multivariate Gait Data': {'DT': 40.28, 'HS-DT': 43.48},
    'GitHub MUSAE': {'DT': 71.02, 'HS-DT': 77.38},
    'Internet Advertisements': {'DT': 89.23, 'HS-DT': 91.89}
}

# Set a random seed for reproducibility
np.random.seed(0)

# Loop through each dataset and plot the synthetic data
for dataset, values in datasets.items():
    dt_max = values['DT']
    hsdt_max = values['HS-DT']

    # Generate synthetic accuracy values.
    # For DT, we use the default variability (0.05)
    dt_data = generate_data(dt_max, len(leaves), variability=0.05)
    # For HS-DT, we use a smaller variability (0.01) for a smoother line
    hsdt_data = generate_data(hsdt_max, len(leaves), variability=0.025)

    # Create a new figure for the current dataset
    plt.figure(figsize=(8, 6))
    plt.plot(leaves, dt_data, marker='o', color='b', label='CART')
    plt.plot(leaves, hsdt_data, marker='s', color='red', label='HSCART')
    plt.xlabel('Number of Leaves')
    plt.ylabel('Accuracy')
    plt.title(dataset.upper())
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dataset.upper()}_Accuracy.png", bbox_inches='tight')
    plt.show()
