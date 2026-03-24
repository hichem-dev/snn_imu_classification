import argparse
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from data_loader import load_dataset
from preprocessing import preprocess_dataset
from snn_model import SNNModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train SNN IMU classifier")
    parser.add_argument("--dataset", default="dataset", help="Dataset root path")
    parser.add_argument("--classes", nargs="+", default=["still", "up_down_slow", "up_down_fast"], help="List of class subfolders")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    label_map = {c: i for i, c in enumerate(args.classes)}

    data, labels = load_dataset(args.dataset, args.classes)
    X, y = preprocess_dataset(data, labels, label_map)
    print("Data shape:", X.shape)  # (samples, time, features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    model = SNNModel(input_size=X_train.shape[2], num_classes=len(args.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        output = model(X_train)
        output = output[:, -1, :]

        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        output = model(X_test)
        output = output[:, -1, :]

        preds = torch.argmax(output, dim=1)
        acc = (preds == y_test).float().mean()

        print("Test Accuracy:", acc.item())


if __name__ == "__main__":
    main()