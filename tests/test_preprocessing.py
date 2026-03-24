import numpy as np
import pandas as pd

from preprocessing import normalize, create_windows, encode_spikes, preprocess_dataset


def test_normalize():
    df = pd.DataFrame({"x": [1, 2], "y": [-1, 1]})
    out = normalize(df)
    assert np.allclose(out.max().values, np.array([1.0, 1.0]))


def test_create_windows():
    df = pd.DataFrame({"x": list(range(10))})
    windows = create_windows(df, window_size=5)
    assert len(windows) == 2
    assert windows[0].shape == (5, 1)


def test_encode_spikes():
    df = pd.DataFrame({"x": [0.0, 0.3, 0.5]})
    spikes = encode_spikes(df, threshold=0.2)
    assert spikes.tolist() == [[0.0], [1.0], [1.0]]


def test_preprocess_dataset():
    df = pd.DataFrame({
        "x": np.linspace(0.0, 1.0, 50),
        "y": np.linspace(1.0, 0.0, 50),
    })
    X, y = preprocess_dataset([df], ["still"], {"still": 0})
    assert X.ndim == 3
    assert y.tolist() == [0]
