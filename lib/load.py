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
from collections import defaultdict
from multiprocessing import Pool, TimeoutError
import threading
from collections import defaultdict
import json
import librosa
import os
import numpy as np
from lib.timing import timing_decorator

def split_audio(y, sr, chunk_size, freq_range, snr_threshold):
    
    # Calculate the number of samples in each chunk
    samples_per_chunk = chunk_size * sr

    # Split the audio signal into chunks
    chunks = [y[i : i + samples_per_chunk] for i in range(0, len(y), samples_per_chunk)]

    filtered_chunks = []
    
    for chunk in chunks:
        # Compute FFT
        yf = fft(chunk)
        xf = fftfreq(len(chunk), 1 / sr)
        
        # Filter the frequencies in the desired range
        signal_freqs = (xf > freq_range[0]) & (xf < freq_range[1])
        noise_freqs = ~signal_freqs
        
        # Compute the power of the signal and the noise
        signal_power = np.sum(np.abs(yf[signal_freqs]) ** 2)
        noise_power = np.sum(np.abs(yf[noise_freqs]) ** 2)
        
        # Compute the SNR
        snr = 10 * np.log10(signal_power / noise_power)  # in dB

        # If the SNR exceeds the threshold, save the chunk
        if snr > snr_threshold:
            filtered_chunks.append(chunk)
    
    return filtered_chunks or chunks

def reduce_noises(y, sr):

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = round(np.median(cent))*1.5

    #less_noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=500, slope=0.8).highshelf(gain=-12.0, frequency=threshold_h, slope=0.5)#.limiter(gain=6.0)
    #y_clean = less_noise(y)

    #return y_clean
    speech_booster = AA.Compose([
        AA.LowShelfFilter(min_center_freq=1500, max_center_freq=1500, min_gain_db=-20, max_gain_db=-20, min_q=.5, max_q=.5),
        AA.HighShelfFilter(min_center_freq=threshold_h, max_center_freq=threshold_h),
        #AA.Limiter(max_threshold_db=8.0)
    ])

    try:
        y_speach_boosted = speech_booster(y, sample_rate=sr)
    except np.linalg.LinAlgError:
        # Handle the exception here - you could return the original audio, log an error message, etc.
        print(f"An error occurred with file at sample rate {sr}. Using unprocessed audio.")
        y_speach_boosted = y  # or however you wish to handle this

    return (y_speach_boosted)

def load_mfccs(directory, metadata_file):
    target_size = (224, 224)

    print(metadata_file)

    with open(os.path.join(directory, metadata_file)) as m:
        metadata = json.load(m)
        audio_path = os.path.join(directory, metadata['filename'])
        y, sr = librosa.load(audio_path)

        label = f"{metadata['genus']} {metadata['species']}"
        mfccs = []

        chunks = split_audio(y, sr, 2, (2000,6000), -5)
        print(f"{metadata_file} - {len(chunks)}")

        for chunk in chunks:
            # clean up the noise
            augmented_y = reduce_noises(chunk, sr)

            # Compute MFCCs
            mfcc =librosa.power_to_db(librosa.feature.melspectrogram(
                np.float32(augmented_y), sr=sr, n_fft=2048, hop_length=512, n_mels=target_size[0]), ref=np.max)
            mfcc = Image.fromarray(mfcc).resize(target_size)
            mfccs.append(mfcc)
            label = f"{metadata['genus']} {metadata['species']}"
        return mfccs, label 

@timing_decorator
def multi_load(directory, test_size=0.2, random_state=42, pool_size=4):
    
    metadata_files = [filename for filename in os.listdir(directory) if filename.endswith('.json')]

    with Pool(processes=pool_size) as pool:
        labels = []
        mfccs = []
        class_counts = defaultdict(int)

        lock = threading.Lock()

        def collect(result):
            chunk_mfccs, label = result
            # concurrently extend chunk_mfccs and labels to the global lists
            with lock:
                class_counts[label] += len(chunk_mfccs)
                labels.extend([label] * len(chunk_mfccs))
                mfccs.extend(chunk_mfccs)

        multiple_results = [pool.apply_async(load_mfccs, (directory, metadata_file,)) for metadata_file in metadata_files[:1000]]
        [collect(res.get(timeout=300)) for res in multiple_results]

        mask = np.vectorize(class_counts.get)(labels) > 1
        labels = np.array(labels)[mask]
        mfccs = np.array(mfccs)[mask]

        # Encode labels to integers
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        mfccs = np.stack(mfccs)
        # Split the data into training and test datasets
        X_train, X_test, y_train, y_test = train_test_split(mfccs, encoded_labels, test_size=test_size, random_state=random_state, stratify=encoded_labels)
        
        return X_train, X_test, y_train, y_test, label_encoder