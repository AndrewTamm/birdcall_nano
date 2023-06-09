from lib import load
import os

def sanitize_directory(directory_path):
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for dirname in dirnames:
            try:
                dirname.encode('utf-8')
            except UnicodeEncodeError:
                print(f"UnicodeEncodeError: Unable to encode directory {dirname} in directory {dirpath}")   

        for filename in filenames:
            if "é" in filename or "è" in filename or "à" in filename or "ç" in filename:
                new_name = ''.join(char for char in filename if ord(char) < 128)
                print(f"Renaming to: {new_name}")
                os.rename(os.path.join(dirpath, filename), os.path.join(dirpath, new_name))
            
if __name__ == "__main__":
    sanitize_directory("spectrograms")
