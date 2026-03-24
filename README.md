<<<<<<< HEAD
# snn_imu_classification
IMU activity classification with CNN and SNN models
=======
# SNN/IMU Classification

A lightweight research repository for IMU-based activity classification using CNN and Spiking Neural Network (SNN) models.

## Project structure

- `dataset/`: raw CSV recordings grouped by label (`still`, `up_down_slow`, `up_down_fast`, etc.)
- `data_loader.py`: load class subfolders and parse CSV files
- `preprocessing.py`: normalization, sliding window, spike encoding
- `cnn_model.py`: CNN model definition
- `snn_model.py`: SNN model definition
- `cnn_train.py`: CNN training and evaluation pipeline
- `snn_train.py`: SNN training and evaluation pipeline

## Quick start

1. `python -m pip install -r requirements.txt`
2. `python cnn_train.py --dataset dataset --classes still up_down_slow up_down_fast --epochs 20 --batch-size 64`
3. `python snn_train.py --dataset dataset --classes still up_down_slow up_down_fast --epochs 10`

## CLI options

- `--dataset`: path to dataset root folder
- `--classes`: list of labels to include
- `--epochs`: training epochs
- `--batch-size`: training batch size (CNN only)
- `--lr`: learning rate
- `--test-size`: ratio for test split
- `--seed`: random seed

## Dataset format

Evaluate CSVs from `dataset/<label>/*.csv`. Columns should be numeric sensor channels with semicolon delimiter.

## Maintenance

- Use branches for features (`feature/`, `bugfix/`)
- Follow `pytest` for unit tests
- PRs run GitHub CI on Python 3.11

## License

MIT
>>>>>>> 8075f57 (chore: initial project import + CLI, tests, CI)
