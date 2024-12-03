import pickle
import json
MODEL_PATH = "/Users/luisvazquez/Documents/cell_classification.pkl"

with open(MODEL_PATH, "rb") as f:
    loaded_model = pickle.load(f)

list = [[0.2, 0.5, 0.1, 0.3],
        [0.2, 0.5, 0.1, 0.3],]

# Model predictions
predictions = loaded_model.predict(list)

# Convert predictions to a list and then to JSON to return to the main script
predictions_list = predictions.tolist()  # Convert numpy array to a regular list

# Print the predictions as a JSON string
print(json.dumps(predictions_list))