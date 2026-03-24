import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from data_loader import load_dataset
from preprocessing import preprocess_dataset
from cnn_model import CNNClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN IMU classifier")
    parser.add_argument("--dataset", default="dataset", help="Dataset root path")
    parser.add_argument("--classes", nargs="+", default=["still", "up_down_slow", "up_down_fast"], help="List of class subfolders")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    label_map = {c: i for i, c in enumerate(args.classes)}

    raw_data, raw_labels = load_dataset(args.dataset, args.classes)
    X, y = preprocess_dataset(raw_data, raw_labels, label_map)
    print("Data shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=max(args.batch_size * 2, 32))

    model = CNNClassifier(num_features=X_train.shape[2], num_classes=len(args.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        running_loss /= len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                out = model(xb)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / total
        print(f"Epoch {epoch}/{args.epochs}, loss={running_loss:.4f}, test_acc={acc:.4f}")

    print("Done")


if __name__ == "__main__":
    main()
