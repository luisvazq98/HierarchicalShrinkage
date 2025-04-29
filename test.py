import math

def compute_confidence_interval(acc_cart, acc_hscart, n):
    # Step 1: Compute difference
    d = acc_hscart - acc_cart

    # Step 2: Compute errors
    error_cart = 1 - acc_cart
    error_hscart = 1 - acc_hscart

    # Step 3: Approximate variance
    variance = ((error_cart * (1 - error_cart)) / n) + ((error_hscart * (1 - error_hscart)) / n)

    # Step 4: Standard deviation
    std_dev = math.sqrt(variance)

    # Step 5: z-value
    z = 1.96

    # Step 6: Compute confidence interval
    lower_bound = d - z * std_dev
    upper_bound = d + z * std_dev

    return d, lower_bound, upper_bound

# Example usage
# Inputs (for a dataset):
acc_cart = 0.8923
acc_hscart = 0.9189
n = 1088  # Number of test samples

# Calculate
d, lower, upper = compute_confidence_interval(acc_cart, acc_hscart, n)

print(f"Observed Difference (d): {d:.4f}")
print(f"95% Confidence Interval: [{lower:.4f}, {upper:.4f}]")

# Interpretation
if lower > 0:
    print("HSCART is significantly better than CART at 95% confidence.")
elif upper < 0:
    print("CART is significantly better than HSCART at 95% confidence.")
else:
    print("No statistically significant difference at 95% confidence.")
