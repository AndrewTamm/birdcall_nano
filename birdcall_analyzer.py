import argparse
import librosa
import numpy as np
import queue
import signal
import skimage.transform
import sounddevice as sd
import sys
import tensorflow as tf
import threading
from tensorflow.keras.models import load_model


class AudioClassifier:
    def __init__(self, sample_rate=44100, duration=2.0, overlap_factor=0.5, model_path="quantized_mobilenetv3_mini.tflite", labels_path="labels.txt"):
        self.sample_rate = sample_rate
        self.duration = duration
        self.buffer_size = int(self.duration * self.sample_rate)
        self.overlap_factor = overlap_factor
        self.q = queue.Queue()
        self.stop_flag = False

        # Setup interpreter
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.q.put(indata.copy())

    def infer_from_stream(self):
        buffer = np.zeros(self.buffer_size)  

        while not self.q.empty():
            self.q.get()

        while not self.stop_flag:
            try:
                chunk = self.q.get()
                chunk_size = len(chunk)
                buffer[:-chunk_size] = buffer[chunk_size:]
                buffer[-chunk_size:] = np.squeeze(chunk)

                spectrogram = librosa.feature.melspectrogram(buffer, sr=self.sample_rate, n_mels=self.input_details['shape'][1], n_fft=2048, hop_length=512, fmin=20, fmax=self.sample_rate/2, power=1.0)
                spectrogram = skimage.transform.resize(spectrogram, (224, 224), anti_aliasing=True)
                spectrogram = np.expand_dims(spectrogram, axis=0) 

                if self.input_details['dtype'] == np.uint8 or self.input_details['dtype'] == np.int8:
                    input_scale, input_zero_point = self.input_details["quantization"]
                    spectrogram = spectrogram / input_scale + input_zero_point

                spectrogram = spectrogram.astype(self.input_details["dtype"])
                spectrogram = np.expand_dims(spectrogram, axis=-1)
                spectrogram = np.repeat(spectrogram, 3, axis=-1)

                self.interpreter.set_tensor(self.input_details['index'], spectrogram)
                self.interpreter.invoke()

                output = self.interpreter.tensor(self.output_details['index'])
                prediction_index = np.argmax(output()[0])

                prediction_label = self.labels[prediction_index]

                print(prediction_label)  
            except Exception as e:
                print(e)
                exit()

    def run(self):
        threading.Thread(target=self.infer_from_stream).start()
        with sd.InputStream(callback=self.audio_callback, device=1, channels=1, samplerate=self.sample_rate, blocksize=int(self.buffer_size*self.overlap_factor)) as stream:
            signal.signal(signal.SIGINT, self.signal_handler)
            while not self.stop_flag:
                pass

    def signal_handler(self, signal, frame):
        self.stop_flag = True
        print('Stopping stream...')
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time audio classification")
    parser.add_argument("-sr", "--sample_rate", type=int, default=44100, help="Sample rate")
    parser.add_argument("-d", "--duration", type=float, default=2.0, help="Duration of audio chunks")
    parser.add_argument("-of", "--overlap_factor", type=float, default=0.5, help="Overlap factor for audio chunks")
    parser.add_argument("-m", "--model_path", type=str, default="quantized_mobilenetv3_mini.tflite", help="Path to TFLite model")
    parser.add_argument("-l", "--labels_path", type=str, default="labels.txt", help="Path to labels file")
    args = parser.parse_args()

    classifier = AudioClassifier(sample_rate=args.sample_rate, duration=args.duration, overlap_factor=args.overlap_factor)
    classifier.run()