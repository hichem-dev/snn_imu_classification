import os
import pandas as pd

def load_dataset(dataset_path, selected_classes):
    data = []
    labels = []

    for label in selected_classes:
        folder = os.path.join(dataset_path, label)

        for file in os.listdir(folder):
            if file.lower().endswith(".csv"):
                df = pd.read_csv(os.path.join(folder, file), sep=';', header=0)
                data.append(df)
                labels.append(label)

    return data, labels

