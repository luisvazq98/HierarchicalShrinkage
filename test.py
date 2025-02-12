hs_models = models[:,9]
cart_models = models[10,:]


results = []
for i in range(0, 10):
    auc_scores_dt = []
    acc_scores_dt = []
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=1 / 3, random_state=i)

    for model in hs_models:
        model.fit(train_x, train_y)
        y_pred_proba = model.predict_proba(test_x)[:,1]
        auc_cart = roc_auc_score(test_y, y_pred_proba)

        # Append CART results
        results.append({
            'Model': 'HSCART',
            'AUC': auc_cart,
            'Split Seed': i
        })
    for model in cart_models:
        model.fit(train_x, train_y)
        y_pred_proba = model.predict_proba(test_x)[:, 1]
        auc_hscart = roc_auc_score(test_y, y_pred_proba)
        results.append({
            'Model': 'CART',
            'AUC': auc_hscart,
            'Split Seed': i
        })


results_df = pd.DataFrame(results)

