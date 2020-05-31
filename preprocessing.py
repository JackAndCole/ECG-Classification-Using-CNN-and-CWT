import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import scipy.signal as sg
import wfdb

PATH = Path("dataset")
sampling_rate = 360

# non-beat labels
invalid_labels = ['|', '~', '!', '+', '[', ']', '"', 'x']

# for correct R-peak location
tol = 0.05


def worker(record):
    # read ML II signal & r-peaks position and labels
    signal = wfdb.rdrecord((PATH / record).as_posix(), channels=[0]).p_signal[:, 0]

    annotation = wfdb.rdann((PATH / record).as_posix(), extension="atr")
    r_peaks, labels = annotation.sample, np.array(annotation.symbol)

    # filtering uses a 200-ms width median filter and 600-ms width median filter
    baseline = sg.medfilt(sg.medfilt(signal, int(0.2 * sampling_rate) - 1), int(0.6 * sampling_rate) - 1)
    filtered_signal = signal - baseline

    # remove non-beat labels
    indices = [i for i, label in enumerate(labels) if label not in invalid_labels]
    r_peaks, labels = r_peaks[indices], labels[indices]

    # align r-peaks
    newR = []
    for r_peak in r_peaks:
        r_left = np.maximum(r_peak - int(tol * sampling_rate), 0)
        r_right = np.minimum(r_peak + int(tol * sampling_rate), len(filtered_signal))
        newR.append(r_left + np.argmax(filtered_signal[r_left:r_right]))
    r_peaks = np.array(newR, dtype="int")

    # remove inter-patient variation
    normalized_signal = filtered_signal / np.mean(filtered_signal[r_peaks])

    # AAMI categories
    AAMI = {
        "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,  # N
        "A": 1, "a": 1, "S": 1, "J": 1,  # SVEB
        "V": 2, "E": 2,  # VEB
        "F": 3,  # F
        "/": 4, "f": 4, "Q": 4  # Q
    }
    categories = [AAMI[label] for label in labels]

    return {
        "record": record,
        "signal": normalized_signal, "r_peaks": r_peaks, "categories": categories
    }


if __name__ == "__main__":
    # for multi-processing
    cpus = 22 if joblib.cpu_count() > 22 else joblib.cpu_count() - 1

    train_records = [
        '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
        '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
        '223', '230'
    ]
    print("train processing...")
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        train_data = [result for result in executor.map(worker, train_records)]

    test_records = [
        '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
        '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
        '233', '234'
    ]
    print("test processing...")
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        test_data = [result for result in executor.map(worker, test_records)]

    with open((PATH / "mitdb.pkl").as_posix(), "wb") as f:
        pickle.dump((train_data, test_data), f, protocol=4)

    print("ok!")
