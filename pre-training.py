import numpy as np
import pywt
import torch
from sklearn.metrics import classification_report, confusion_matrix
from skorch import NeuralNetClassifier

from main import MyModule, load_data

if __name__ == "__main__":
    sampling_rate = 360

    wavelet = "mexh"  # mexh, morl, gaus8, gaus4
    scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 101, 1)

    (x1_train, x2_train, y_train, groups_train), (x1_test, x2_test, y_test, groups_test) = load_data(
        wavelet=wavelet, scales=scales, sampling_rate=sampling_rate)

    net = NeuralNetClassifier(
        MyModule,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    net.initialize()
    net.load_params(f_params="./models/model_{}.pkl".format(wavelet))

    y_true, y_pred = y_test, net.predict({"x1": x1_test, "x2": x2_test})

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))
