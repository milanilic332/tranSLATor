import os
from tqdm import tqdm
from itertools import chain


def get_data(path):
    """Gets the dictionary of artists songs

    @param path:                        dataset path
    @return:                            dictionary of songs for all artists
    """
    return {artist_name: [song_name[:song_name.rfind('.')] for song_name in os.listdir(os.path.join(path, artist_name))]
            for artist_name in os.listdir(path) if os.path.isdir(os.path.join(path, artist_name))}


def test_equality(data_dir):
    """Tests if html dataset has the same files as lrc dataset

    @param data_dir:                    path to directory containing all datasets
    """
    lrc_path = os.path.join(data_dir, 'dataset_lrc')
    mp3_path = os.path.join(data_dir, 'dataset_mp3_full')

    lrc_data, mp3_data = get_data(lrc_path), get_data(mp3_path)

    lrc_artists, mp3_artists = lrc_data.keys(), mp3_data.keys()

    for artist in tqdm(lrc_artists, desc='Checking if lrc and mp3 songs are the same'.ljust(50)):
        if set(lrc_data[artist]) != set(mp3_data[artist]):
            print(f'Equality warning in artist {artist} - {set(lrc_data[artist]) ^ set(mp3_data[artist])}')


def test_labels(data_dir):
    """Tests if raw html was correctly copied

    @param data_dir:                    path to directory containing all datasets
    """
    lrc_path = os.path.join(data_dir, 'dataset_lrc')

    for artist in tqdm(os.listdir(lrc_path), desc='Testing if labels are correct'.ljust(50)):
        texts = {}
        for song in os.listdir(os.path.join(lrc_path, artist)):
            with open(os.path.join(lrc_path, artist, song), 'r') as f:
                texts[song] = f.read()

        rev_texts = {}
        for key, value in texts.items():
            rev_texts.setdefault(value, set()).add(key)

        dups = set(chain.from_iterable(vals for key, vals in rev_texts.items() if len(vals) > 1))

        if dups:
            print(f'Labels warning (duplicate labels) in {artist}: {dups}')

        if '' in list(texts.values()):
            print(f'Labels warning (empty labels) in {artist}')


