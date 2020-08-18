# Classification

## Dataset

For classification purposes, full dataset is filtered down to contain only solo songs.
Lines are then filtered so there is no duplicate transcripts in the dataset. 
These transformations bring the total number of lines to **11867**. \
Distribution of song lines per artist is as follows:

![Artists](https://github.com/milanilic332/tranSLATor/blob/master/classification/images/artists.png)

Next histogram contains the information about the line length (in ms) of the dataset:

![Artists](https://github.com/milanilic332/tranSLATor/blob/master/classification/images/length.png)

## Training

Lines represented as *.wav* files, were transformed into spectograms,
resized to fixed size and then fed to the convolutional neural network. \
Four different CNNs were evaluated with the same hyperparameter settings and results are shown bellow.

| Parameter | Train/Val | Input shape   | Epochs | Batch size | Optimizer | Scheduler        | Loss |
|-----------|-----------------|---------------|--------|------------|-----------|------------------|------|
| **Value**     | 80/20     | (128, 256, 3) | 200    | 4          | RAdam     | Cosine Annealing | CCE  |


| Architecture      | Augmentations | Slice Accuracy | Song Accuracy* |
|-------------------|---------------|----------------|---------------|
| MobileNetV2       | SpecAugment   | 86.93%          | **100%**           |
| InceptionV3       | SpecAugment   | 88.17%          | **100%**           |
| Resnet50V2        | SpecAugment   | 89.27%          | **100%**           |
| DenseNet121       | SpecAugment   | **89.62%**          | 97.78%         |
| Ensemble (4 nets) | -             | **91.87%**          | **100%**           |

Song Accuracy* is calculated by getting the majority vote of predictions for lines in the song.

