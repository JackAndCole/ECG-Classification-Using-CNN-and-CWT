# Code for: Automatic ECG Classification Using Convolutional Neural Network and Continuous Wavelet Transform
## Dataset

[MIT-BIH Arrhythmia database](https://www.physionet.org/content/mitdb/1.0.0/)

## Usage

- Reproduce the results

> A pre-trained model is provided in the model directory. To reproduce our results, you should, first, download the MIT-BIH arrhythmia database from the above link and save it in the dataset directory. Then, execute  `preprocessing.py` to obtain the training and test dataset. After that, just run `pre-training.py` and you will get the results of our paper.

- Re-train the model

> Just replace the last step of run `pre-training.py` with `main.py`.
>
> Our CNN is implemented using PyTorch v1.4.0  and trained on the NVIDIA GeForce RTX 2080Ti graphical processing unit. Noted that different versions of PyTorch and CUDA may cause the results to be slightly different from the results in the article.

## Cite

If our work is helpful to you, please cite:

Wang, T.; Lu, C.; Sun, Y.; Yang, M.; Liu, C.; Ou, C. Automatic ECG Classification Using Continuous Wavelet Transform and Convolutional Neural Network. Entropy 2021, 23, 119. https://doi.org/10.3390/e23010119

## Email:

If you have any questions, please email to: [wtustc@mail.ustc.edu.cn](mailto:wtustc@mail.ustc.edu.cn)
