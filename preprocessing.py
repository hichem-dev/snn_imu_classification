import numpy as np

def normalize(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric columns found for normalization")

    max_vals = numeric_df.abs().max()
    max_vals[max_vals == 0] = 1.0
    return numeric_df / max_vals

def create_windows(df, window_size=50):
    windows = []
    for i in range(0, len(df) - window_size + 1, window_size):
        windows.append(df.iloc[i:i+window_size])
    return windows

def encode_spikes(window, threshold=0.2):
    return (window.values > threshold).astype(float)

def preprocess_dataset(data, labels, label_map):
    X = []
    y = []

    for df, label in zip(data, labels):
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            continue

        numeric_df = normalize(numeric_df)
        windows = create_windows(numeric_df)

        for w in windows:
            spikes = encode_spikes(w)
            X.append(spikes)
            y.append(label_map[label])

    return np.array(X), np.array(y)