import os
import string
from tqdm import tqdm
from bs4 import BeautifulSoup
from mutagen.mp3 import MP3


def clean_dirs(dataset_html, dataset_lrc):
    """Creates directory structure in .lrc dataset

    @param dataset_html:                path to directory containing .html files
    @param dataset_lrc:                 path to directory containing .lrc files
    """
    for artist in tqdm(os.listdir(dataset_html), desc='Cleaning directories'.ljust(50)):
        if artist not in os.listdir(dataset_lrc):
            os.mkdir(os.path.join(dataset_lrc, artist))
        else:
            for file in os.listdir(os.path.join(dataset_lrc, artist)):
                os.remove(os.path.join(dataset_lrc, artist, file))


def html2lrc(data_dir):
    """Converts raw .html dataset to .lrc dataset

    @param data_dir:                    path to directory containing all datasets
    """
    html_path = os.path.join(data_dir, 'dataset_html')
    lrc_path = os.path.join(data_dir, 'dataset_lrc')
    mp3_path = os.path.join(data_dir, 'dataset_mp3_full')

    clean_dirs(html_path, lrc_path)
    for artist in tqdm(os.listdir(html_path), desc='Creating .lrc from .html'.ljust(50)):
        for song in os.listdir(os.path.join(html_path, artist)):
            try:
                mp3_file = MP3(os.path.join(mp3_path, artist, song[:-4] + 'mp3'))
            except:
                print('No such mp3 file in directory: {} skipped'.format(song[:-4] + 'mp3'))
                continue

            with open(os.path.join(html_path, artist, song), 'r') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')

            timestamps = [link.get('data-milliseconds') for link in soup.find_all('a')]
            timestamps = [[timestamps[i], timestamps[i + 1]] for i in range(len(timestamps) - 1)] + \
                         [[timestamps[-1], int(mp3_file.info.length * 1000)]]

            allowed = list(string.ascii_letters) + ['\'', ' ']
            bars = [''.join(x for x in link.get_text() if x in allowed)
                    for link in soup.find_all('a')]

            lines = [f'{timestamps[i]}'.replace('\'', '') + f' {bars[i]}\n'.lower() for i in range(len(bars))]

            with open(os.path.join(lrc_path, artist, song[:-4] + 'lrc'), 'w') as f:
                f.writelines(lines)
