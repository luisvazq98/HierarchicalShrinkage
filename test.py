# Define the noise levels as percentages (converted to fractions)
noise_levels = [0.01, 0.05, 0.10, 0.15, 0.30, 0.45, 0.49]

# Dictionaries to store the average AUC results for each model
results_noisy_auc = {'CART': [], 'HSCART': []}
results_denoised_auc = {'CART': [], 'HSCART': []}

for noise in noise_levels:
    # Inject noise at the current noise level
    Y_noisy, noisy_indices = introduce_label_noise(Y, noise_ratio=noise)

    # -------------------------
    # Experiment Part 1: Train on Noisy Data
    # -------------------------
    results_noisy = training_models(X, Y_noisy, cart_hscart_estimators)
    # Compute mean AUC per model over the different splits
    noisy_group = results_noisy.groupby('Model')['AUC'].mean()
    results_noisy_auc['CART'].append(noisy_group['CART'])
    results_noisy_auc['HSCART'].append(noisy_group['HSCART'])

    # -------------------------
    # Denoising Step: Use Cleanlab to filter probable label issues
    # -------------------------
    train_x_full, _, train_y_noisy, _ = train_test_split(X, Y_noisy, test_size=1 / 3, random_state=0)
    if SOURCE != 'imodels':
        train_x_full = train_x_full.reset_index(drop=True).to_numpy()
        train_y_noisy = train_y_noisy.reset_index(drop=True).to_numpy()
    train_y_noisy = train_y_noisy.astype(int)

    # Train a simple base model for Cleanlab
    base_model = DecisionTreeClassifier(max_leaf_nodes=30, random_state=0)
    base_model.fit(train_x_full, train_y_noisy)
    probas = base_model.predict_proba(train_x_full)
    noise_indices_est = find_label_issues(labels=train_y_noisy, pred_probs=probas,
                                          return_indices_ranked_by='normalized_margin')

    # Create denoised training set
    mask = np.ones(len(train_y_noisy), dtype=bool)
    mask[noise_indices_est] = False
    train_x_denoised = train_x_full[mask]
    train_y_denoised = train_y_noisy[mask]

    # -------------------------
    # Experiment Part 2: Retrain on Denoised Data
    # -------------------------
    _, test_x, _, test_y = train_test_split(X, Y_noisy, test_size=1 / 3, random_state=0)
    if SOURCE != 'imodels':
        test_x = test_x.reset_index(drop=True).to_numpy()
        test_y = test_y.reset_index(drop=True).to_numpy()

    results_denoised = training_models_denoised(train_x_denoised, train_y_denoised, test_x, test_y,
                                                cart_hscart_estimators)
    denoised_group = results_denoised.groupby('Model')['AUC'].mean()
    results_denoised_auc['CART'].append(denoised_group['CART'])
    results_denoised_auc['HSCART'].append(denoised_group['HSCART'])

# -------------------------
# Plotting the Comparison Graph
# -------------------------
plt.figure(figsize=(10, 6))
# Convert noise levels to percentage for x-axis labels
noise_percents = [100 * n for n in noise_levels]
plt.plot(noise_percents, results_noisy_auc['CART'], marker='o', linestyle='-', label='CART Noisy')
plt.plot(noise_percents, results_noisy_auc['HSCART'], marker='o', linestyle='-', label='HSCART Noisy')
plt.plot(noise_percents, results_denoised_auc['CART'], marker='o', linestyle='-', label='CART Denoised')
plt.plot(noise_percents, results_denoised_auc['HSCART'], marker='o', linestyle='-', label='HSCART Denoised')

plt.xlabel("Noise Level (%)")
plt.ylabel("Average AUC")
plt.title("Effect of Noise Level on Model Performance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
