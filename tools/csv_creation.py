from ast import literal_eval
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import librosa
from mutagen.mp3 import EasyMP3 as MP3
from pydub import AudioSegment


def create_full_csv(data_dir, min_seconds=1, max_seconds=10, offsets=None):
    """Splits .mp3 files, saves .wav slices and creates .csv file containing full dataset

    @param data_dir:                    path to directory containing all datasets
    @param min_seconds:                 lower time length limit for slices
    @param max_seconds:                 upper time length limit for slices
    @param offsets:                     left and right offset (in ms) for slices
    """
    if offsets is None:
        offsets = [0, 300]

    lrc_path = os.path.join(data_dir, 'dataset_lrc')
    mp3_path = os.path.join(data_dir, 'dataset_mp3_full')

    dataset_csv = pd.DataFrame(columns=['wav_filename', 'wav_filesize', 'song_name', 'artist', 'length', 'transcript'])
    for artist in tqdm(os.listdir(lrc_path), desc='Creating final dataset'.ljust(50)):
        for song in os.listdir(os.path.join(lrc_path, artist)):
            try:
                mutagen_file = MP3(os.path.join(mp3_path, artist, song[:-3] + 'mp3'))
                pydub_file = AudioSegment.from_mp3(os.path.join(mp3_path, artist, song[:-3] + 'mp3'))
            except IOError:
                logging.warning('No such mp3 file in directory: {} skipped'.format(song[:-3] + 'mp3'))
                continue

            with open(os.path.join(lrc_path, artist, song), 'r') as f:
                lines = f.read().split('\n')[:-1]

            for line_num, line in enumerate(lines):
                timestamp = literal_eval(line[line.find('['):line.find(']') + 1])
                transcript = line[line.find(']') + 2:]
                if min_seconds * 1000 < timestamp[1] - timestamp[0] < max_seconds * 1000:
                    mp3_slice = pydub_file[timestamp[0] + offsets[0]:
                                           np.min([timestamp[1] + offsets[1], int(mutagen_file.info.length * 1000)])]

                    wav_filename = os.path.abspath(os.path.join(data_dir, 'dataset_wav_lines',
                                                                f'{artist} {song[:-4]} {line_num}.wav'))
                    mp3_slice.export(wav_filename, format='wav')

                    if os.path.getsize(wav_filename) != 44:
                        sorted_artists = sorted(mutagen_file.get('artist')[0].lower().split('; '))

                        dataset_csv = dataset_csv.append({'wav_filename': wav_filename,
                                                          'wav_filesize': os.path.getsize(wav_filename),
                                                          'song_name': song[:-4],
                                                          'artist': sorted_artists,
                                                          'length': timestamp[1] - timestamp[0] + sum(offsets),
                                                          'transcript': transcript}, ignore_index=True)

    dataset_csv.to_csv(os.path.join(data_dir, 'dataset_all.csv'), index=False)


def create_stratified_openseq2seq_train_val_csv(data_dir, csv_name):
    """Creates stratified train and val dataset for translation purposes

    @param data_dir:                    path to directory containing all datasets
    @param csv_name:                    name of full datasets .csv file
    """
    df = pd.read_csv(os.path.join(data_dir, csv_name))
    df = df.drop_duplicates(['transcript'])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1920)

    x = df.copy()
    y = (df['artist'].apply(lambda a: literal_eval(a)) + df['length'].apply(lambda l: [int(l) // 1000])).map(str)

    train_index, val_index = next(skf.split(x, y))
    train_df, val_df = x.iloc[train_index].reset_index(drop=True), x.iloc[val_index].reset_index(drop=True)

    train_df.to_csv(os.path.join(data_dir, 'train_openseq2seq.csv'), index=False)

    val_df.to_csv(os.path.join(data_dir, 'val_openseq2seq.csv'), index=False)


def create_stratified_classification_train_val_csv(data_dir, csv_name, val_frac=0.2):
    """Creates stratified train and val datasets for classification purposes

    @param data_dir:                    path to directory containing all datasets
    @param csv_name:                    name of full dataset .csv file
    @param val_frac:                    frac of samples in val dataset
    """
    df = pd.read_csv(os.path.join(data_dir, csv_name))
    df = df.drop_duplicates(['transcript'])

    artist_list = [literal_eval(x) for x in list(df['artist'].unique())]
    artist_list = list(filter(lambda x: len(x) == 1, artist_list))
    artist_list = list(map(lambda x: str(x), artist_list))

    train_df = pd.DataFrame(columns=df.columns)
    val_df = pd.DataFrame(columns=df.columns)
    for artist in artist_list:
        df_artist = df[df['artist'] == artist].sample(frac=1, random_state=1920).reset_index(drop=True)
        song_list = list((df_artist['song_name']).unique())

        train_song_list = song_list[:int(np.round((1 - val_frac) * len(song_list)))]
        val_song_list = song_list[int(np.round((1 - val_frac) * (len(song_list)))):]

        train_df = pd.concat([train_df, df_artist[df_artist['song_name'].isin(train_song_list)]], axis=0)
        val_df = pd.concat([val_df, df_artist[df_artist['song_name'].isin(val_song_list)]], axis=0)

    train_df.to_csv(os.path.join(data_dir, 'train_classification.csv'), index=False)

    val_df.to_csv(os.path.join(data_dir, 'val_classification.csv'), index=False)
