# Birdcall Identification Tool
This tool trains various machine learning models to identify bird species based on their calls. The models focus on minimizing model size with a goal of executing on an Arduino Nano BLE rather than focusing on correctness, so while there may be significantly more accurate models, they are also necessarily much larger. 

## Dependencies
This project requires Tensorflow and the custom libraries as specified in the import section of the main application.

<pre>
pip install requests tensorflow tensorflow_model_optimization librosa scikit-learn audiomentations pillow scipy matplotlib temppathlib
</pre>

This command should install the following packages:

* `requests` for making HTTP requests.
* `tensorflow` for machine learning functionalities.
* `tensorflow_model_optimization` for TensorFlow model optimization tools.
* `librosa` for music and audio analysis.
* `scikit-learn` for machine learning and data analysis tools.
* `audiomentations` for audio data augmentation.
* `pillow` for handling images.
* `scipy` for scientific computations.
* `matplotlib` for data visualization.
* `temppathlib` for temporary file and directory management.

## Usage
Use the command line interface to specify options for training the model:

<pre>
python main.py --species top_20_species --pool_size 10 --epochs 100 --file_limit 100 --plot_samples --plot_training_history --quantize --prune --f1_score --model_squeezenet
</pre>

## Options
Here are descriptions of the command line options:

* `--species`: Name of the species to be classified. Default is 'top_20_species'.
* `--pool_size`: Number of processes to spawn. Default is 10.
* `--epochs`: Number of epochs to train for. Default is 100.
* `--file_limit`: Maximum number of files to process. Default is None which processes all files.
* `--plot_samples`: Show a plot of sample spectrograms.
* `--plot_training_history`: Show a plot of training history.
* `--quantize`: Quantize the model.
* `--prune`: Prune the model.
* `--f1_score`: Compute F1 Score.
* `--model_squeezenet`: Use SqueezeNet model.
* `--model_squeezenext`: Use SqueezeNext model.
* `--model_mobilenetv3`: Use MobileNetV3 model.

Note: You must specify one of the model options: `--model_squeezenet`, `--model_squeezenext`, or `--model_mobilenetv3`.

## Downloading Birdcall Data
This tool allows you to download birdcall audio files and their associated metadata from the Xeno-Canto website. It is capable of fetching the top N bird species and storing their audio files in a specified directory.

### Usage
Ensure you have `Python 3.7+` installed on your system.

Install required Python packages if they are not already installed:

<pre>
pip install requests
</pre>
Run the script by typing the following command in your terminal:

<pre>
python lib/download.py --subdirectory "directory_name" --num_species N
</pre>
Replace `"directory_name"` with the name of the directory where you want the downloaded files to be saved. Replace `N` with the number of top bird species you want to fetch.

For example, to fetch the top 25 species and save them in a directory named `"top_species"`, use:

<pre>
python script_name.py --subdirectory "top_species" --num_species 25
</pre>
If you want to use the default parameters (`subdirectory = "top_20_species"` and `num_species = 20`), you can simply run:

<pre>
python script_name.py
</pre>
The script will create the specified directory if it doesn't exist. For each birdcall, it downloads the audio file and a corresponding JSON file containing metadata for the birdcall. It skips any files that have already been downloaded, so it's safe to run the script multiple times - it will only download new files.

The script automatically filters out `'Mystery'` species, so only identifiable bird species are fetched. It also provides an output of the top N species it has downloaded, represented as genus-species pairs.

Please note that this script is intended for educational and research purposes. Always respect the Xeno-Canto Terms of Service when using their data.

## Real-time Birdcall Classifier
This program utilizes TensorFlow and sounddevice to classify bird calls in real-time by taking continuous input from the microphone. The machine learning model used for classification is based on a MobileNetV3 architecture.

# Getting Started
These instructions will guide you on how to get the application up and running on your local machine.

# Prerequisites
Make sure to have the following installed on your system:

* Python 3.x
* sounddevice
* TensorFlow
* Librosa
* skimage
* Matplotlib
* Numpy
You can install the required packages using pip:

<pre>
pip install sounddevice tensorflow librosa scikit-image matplotlib numpy
</pre>

## Usage
You can run the program with default parameters using the following command:

<pre>
python birdcall_classifier.py
</pre>
## Command Line Arguments
You can customize the sample rate, duration of audio chunks, and overlap factor for audio chunks with command line arguments. Below is a list of available arguments:

* `-sr` or `--sample_rate`: Specify the sample rate (default: 44100)
* `-d` or `--duration`: Specify the duration of audio chunks (default: 2.0 seconds)
* `-of` or `--overlap_factor`: Specify the overlap factor for audio chunks (default: 0.5)
* `-l` or `--labels_path`: Specify the path for the label file
* `-m` or `--model_path`: Specify the path for the TFLite model

For example, to set the sample rate to 48000, duration to 2.5 seconds, and overlap factor to 0.6, use the following command:

<pre>
python birdcall_classifier.py -sr 48000 -d 2.5 -of 0.6
</pre>

## Acknowledgement

* This work would not be possible without the amazing work done by the volunteers at [xeno-canto](https://xeno-canto.org/) who have collected and categorized so many samples. 

* The ongoing [BirdCLEF competition](BirdCLEF 2023) by the Cornell School of Ornithology has many brilliant people who have much better solutions to categorizations, you should see their work. 

* An extra special thanks to Josafat-Mattias Burmeister to Maximilian Götz whose [BirdSongIdentification](https://github.com/josafatburmeister/BirdSongIdentification) helped me understand how to work with some of these large datasets without attempting to allocate 120 gig arrays on a laptop. In particular, their algorithm to identify calls out of the noise was significantly more accurate than my own first guess and made a massive jump in training. 
