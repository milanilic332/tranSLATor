from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet_v2 import ResNet50V2
from efficientnet.keras import EfficientNetB2
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Input
from keras.models import Model
from keras_radam import RAdam

import argparse
import pandas as pd
import librosa as lr
import numpy as np
import cv2
import os

from tools.specaugment import spec_augment
from tools.cosine_annealing import CosineAnnealingScheduler
from classification.labels import unique_labels


def yield_data(df, batch_size, input_shape, label_dict, dataset_type, to_augment=False):
    """Data generator

    @param df:                              dataframe of dataset
    @param batch_size:                      number of samples in batch
    @param input_shape:                     neural network input shape
    @param label_dict:                      dictionary mapping rapper names to integers
    @param dataset_type:                    either 'train' or 'val'
    @param to_augment:                      enable specaugment or not
    """
    df = df[df['artist'].isin(label_dict.keys())].reset_index(drop=True)
    while True:
        if dataset_type == 'train':
            df = df.sample(frac=1).reset_index(drop=True)
        for batch_num in range(df.shape[0] // batch_size):
            xs = np.zeros((batch_size, *input_shape))
            ys = np.zeros((batch_size,))
            for i in range(batch_size):
                clip, sr = lr.load(df.loc[[batch_num * batch_size + i]]['wav_filename'].values[0], sr=None)

                mel = lr.feature.melspectrogram(y=clip, sr=sr)

                if dataset_type == 'train' and to_augment:
                    mel = spec_augment(mel)

                mel = lr.power_to_db(mel, ref=np.max)
                mel = cv2.resize(mel, (input_shape[1], input_shape[0]))

                xs[i] = np.dstack([mel, mel, mel]) / 255.

                label = label_dict[df.loc[[batch_num * batch_size + i]]['artist'].values[0]]
                ys[i] = label

            yield xs, ys


def build_model(backbone_name, input_shape, n_classes):
    """Build neural network architecture

    @param backbone_name:                       name of the backbone from implemented names
    @param input_shape:                         neural network input shape
    @param n_classes:                           number of classes
    @return:                                    classification model
    """
    inputs = Input(shape=input_shape)

    if backbone_name == 'inceptionv3':
        backbone = InceptionV3(weights=None, include_top=True, classes=n_classes)(inputs)
    elif backbone_name == 'densenet121':
        backbone = DenseNet121(weights=None, include_top=True, classes=n_classes)(inputs)
    elif backbone_name == 'efficientnetb2':
        backbone = EfficientNetB2(weights=None, include_top=True, classes=n_classes)(inputs)
    elif backbone_name == 'mobilenetv2':
        backbone = MobileNetV2(weights=None, include_top=True, classes=n_classes)(inputs)
    elif backbone_name == 'resnet50':
        backbone = ResNet50V2(weights=None, include_top=True, classes=n_classes)(inputs)
    else:
        raise ValueError(f'Not implemented for {backbone_name} backbone.')

    final_model = Model(inputs, backbone)

    final_model.summary()

    return final_model


def main(train_csv, val_csv, backbone_name, input_shape, epochs, batch_size, to_augment):
    """Training loop

    @param train_csv:                           path to train .csv file
    @param val_csv:                             path to val .csv file
    @param backbone_name:                       name of the backbone from implemented names
    @param input_shape:                         neural network input shape
    @param epochs:                              number of training epochs
    @param batch_size:                          number of samples in batch
    @param to_augment:                          enable specaugment or not
    """
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    label_dict = {k: v for (k, v) in zip(unique_labels, [i for i in range(len(unique_labels))])}

    model = build_model(backbone_name, input_shape, len(label_dict.keys()))

    model.compile(loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'],
                  optimizer=RAdam(0.001))

    train_gen = yield_data(train_df, batch_size, input_shape, label_dict, 'train', to_augment)
    val_gen = yield_data(val_df, batch_size, input_shape, label_dict, 'val')

    es_callback = EarlyStopping(patience=100)
    mc_callback = ModelCheckpoint(f'models/{backbone_name}.h5', monitor='val_sparse_categorical_accuracy',
                                  mode='max', save_best_only=True, save_weights_only=False)
    ca_callback = CosineAnnealingScheduler(40, 0.0005, 0.00005)
    tb_callback = TensorBoard(f'logs/{backbone_name}/')

    model.fit_generator(generator=train_gen,
                        steps_per_epoch=train_df.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=val_gen,
                        validation_steps=val_df.shape[0] // batch_size,
                        callbacks=[es_callback, ca_callback, mc_callback, tb_callback])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training hyperparams')
    parser.add_argument('--train_csv', type=str, help='Path to train csv',
                        default=os.path.join('..', 'dataset', 'data', 'train_classification.csv'))
    parser.add_argument('--val_csv', type=str, help='Path to val csv',
                        default=os.path.join('..', 'dataset', 'data', 'val_classification.csv'))
    parser.add_argument('--backbone_name', type=str, help='Network backbone name', default='inceptionv3')
    parser.add_argument('--input_shape', type=str, help='Input shape for network', default='(128, 256, 3)')
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=200)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=4)
    parser.add_argument('--augment', type=bool, help='If you want to augment train dataset', default=True)

    args = parser.parse_args()

    main(train_csv=args.train_csv,
         val_csv=args.val_csv,
         backbone_name=args.backbone_name,
         input_shape=eval(args.input_shape),
         epochs=args.epochs,
         batch_size=args.batch_size,
         to_augment=args.augment)
