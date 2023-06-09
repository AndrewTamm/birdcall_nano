from tensorflow.keras.models import Model, save_model
import os
import json
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import audiomentations as AA
from PIL import Image
from scipy.fft import fft, fftfreq
from multiprocessing import Pool, TimeoutError
import threading
import json
import librosa
import os
import numpy as np
from lib.timing import timing_decorator
from scipy import ndimage
from typing import Tuple
import matplotlib.pyplot as plt
from tensorflow.keras.utils import image_dataset_from_directory

def __create_spectrogram(y, sr, target_size):
    mfcc = librosa.power_to_db(librosa.feature.melspectrogram(
                np.float32(y), sr=sr, n_fft=2048, hop_length=512, n_mels=target_size[0]), ref=np.max)
    return np.array(Image.fromarray(mfcc).resize(target_size))

def __save_spectrogram(spectrogram, path):
    spectrogram = np.flip(spectrogram, axis=0)
    plt.imsave(path, spectrogram, cmap='gray')
    

def __filter_noise(spectrogram: np.ndarray) -> np.ndarray:
    ## adapted from https://github.com/josafatburmeister/BirdSongIdentification/blob/main/src/spectrograms/spectrograms.py#L103
    # normalize spectrogram to [0, 1]
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))

    # apply median blur with kernel size 5
    filtered_spectrogram = ndimage.median_filter(spectrogram, size=5)

    # apply median filtering
    # pixels that are 1.5 times larger than the row and the column median are set to black
    # all other pixels are set to white
    row_median = np.median(filtered_spectrogram, axis=0)
    col_median = np.median(filtered_spectrogram, axis=1)

    filtered_spectrogram[filtered_spectrogram < row_median * 1.5] = 0.0
    filtered_spectrogram[filtered_spectrogram < col_median * 1.5] = 0.0
    filtered_spectrogram[filtered_spectrogram > 0] = 1.0

    # create matrix that indicates for each pixel to which region it belongs
    # a region is a connected area of black pixels
    struct = np.ones((3, 3), dtype=np.int)
    region_labels, num_regions = ndimage.label(filtered_spectrogram, structure=struct)

    # calculate size (number of black pixels) of each region
    region_sizes = np.array(ndimage.sum(filtered_spectrogram, region_labels, range(num_regions + 1)))

    # set isolated black pixels to zero
    region_mask = (region_sizes == 1)
    filtered_spectrogram[region_mask[region_labels]] = 0

    # apply morphology closing
    struct = np.ones((5, 5))
    filtered_spectrogram = ndimage.morphology.binary_closing(filtered_spectrogram, structure=struct).astype(np.int)

    return filtered_spectrogram

def __contains_signal(spectrogram: np.ndarray, signal_threshold: int = 3, noise_threshold: int = 1) -> Tuple[bool, bool]:

    assert noise_threshold <= signal_threshold

    filtered_spectrogram = __filter_noise(spectrogram)

    row_max = np.max(filtered_spectrogram, axis=1)

    # apply binary dilation to array with max values
    # see https://github.com/kahst/BirdCLEF2017/blob/f485a3f9083b35bdd7a276dcd1c14da3a9568d85/birdCLEF_spec.py#L120
    row_max = ndimage.morphology.binary_dilation(row_max, iterations=2).astype(row_max.dtype)

    # count rows with signal
    rows_with_signal = row_max.sum()

    return rows_with_signal >= signal_threshold, rows_with_signal < noise_threshold


def load_mfccs(directory, metadata_file):
    target_size = (224, 224)

    with open(os.path.join(directory, metadata_file)) as m:
        metadata = json.load(m)
        audio_path = os.path.join(directory, metadata['filename'])
        try: 
            y, sr = librosa.load(audio_path)
        except:
            print(f"An error occurred with file {audio_path}. Skipping file.")
            return [], "", []

        label = f"{metadata['genus']} {metadata['species']}"
        mfccs = []
        noise = []

        # split audio into 2 second chunks
        # Calculate the number of samples in each chunk
        samples_per_chunk = 2 * sr

        # Split the audio signal into chunks
        chunks = [y[i : i + samples_per_chunk] for i in range(0, len(y), samples_per_chunk)]

        for chunk in chunks:
            mfcc = __create_spectrogram(chunk, sr, target_size)
            is_signal, is_noise = __contains_signal(mfcc)

            if is_signal and not is_noise:
                mfccs.append(mfcc)
                ## save a copy of the spectrogram in the spectrograms folder with the species name and a chunk tag
                filepart = os.path.splitext(metadata['filename'])[0]
                os.makedirs(f"spectrograms/{label}", exist_ok=True)
                __save_spectrogram(mfcc, f"spectrograms/{label}/{filepart[:20]}_{len(mfccs)}.png")
            elif is_noise:
                noise.append(mfcc)
            
        return mfccs, label, noise
    
def __generate_dataset():
    try:
        # load images from directory and generate labels using image_dataset_from_directory
        training_set = image_dataset_from_directory(
            'spectrograms/',
            validation_split=0.2,
            subset='training',
            seed=123,
            image_size=(224, 224),
            batch_size=32,
            color_mode='rgb'
        )
    except Exception as e:
        print(e)

    # load images for the validation set
    validation_set = image_dataset_from_directory(
        'spectrograms/',
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(224, 224),
        batch_size=32,
        color_mode='rgb'
    )

    return training_set, validation_set


@timing_decorator
def multi_load(directory, test_size=0.2, random_state=None, pool_size=4, file_limit=None, use_cache=True):
    
    metadata_files = [filename for filename in os.listdir(directory) if filename.endswith('.json')]
    if file_limit:
        metadata_files = metadata_files[:file_limit]

    with Pool(processes=pool_size) as pool:
        labels = []
        mfccs = []

        lock = threading.Lock()

        def collect(result):
            chunk_mfccs, label, _ = result
            # concurrently extend chunk_mfccs and labels to the global lists
            with lock:
                labels.extend([label] * len(chunk_mfccs))
                mfccs.extend(chunk_mfccs)

        if not use_cache:
            multiple_results = [pool.apply_async(load_mfccs, (directory, metadata_file,)) for metadata_file in metadata_files]
            [collect(res.get(timeout=300)) for res in multiple_results]

        return __generate_dataset()