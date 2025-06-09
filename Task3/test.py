import pickle

regions = ['tibia', 'femur', 'background']
# --- Configuration ---
# Path to your saved pickle file
features_filepath = '/Users/teksinghayer/Desktop/Naamii-selection-task/Task3/extracted_features.pkl'
with open(features_filepath, 'rb') as f:
    temp_features = pickle.load(f)
    print("Example layer keys:", temp_features)