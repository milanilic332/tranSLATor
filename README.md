# tranSLATor

Music classification and recognition using neural netowrks.


## Dataset

Dataset consists of 437 different songs collected from 8 artists. Those songs are split into 32166 lines using timestamp information of each line.

## Classifiaction

The goal of classification is to predict the artist who said the input line. Dataset is filtered by the transcript of the line, because of the many repeated lines which are the part of chorus and some adlibs. Also due to labeling limitations, only lines from solo songs were used for the dataset.


| Architecture   | Augmentations | Slice Accuracy (%) | Song Accuracy (%) |
|----------------|---------------|--------------------|-------------------|
| MobileNetV2    | SpecAugment   | 86.93              | 100               |
| DenseNet121    | SpecAugment   | **89.62**              | 97.78             |
| InceptionV3    | SpecAugment   | 88.17              | 100               |
| ResNet50V2     | SpecAugment   | 89.27              | 100               |
| Ensemble (all) | -             | **91.87**              | 100               |

## Recognition


## Progress


## References
