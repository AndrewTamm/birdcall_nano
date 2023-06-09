import temppathlib
import zipfile

def get_tflite_model_size(tflite_model):
    with temppathlib.NamedTemporaryFile() as f:
        f.file.write(tflite_model)
        f.file.flush()
        size = f.path.stat().st_size
    return size

def save_tflite_model(tflite_model, save_path):
    with open(save_path, 'wb') as f:
        f.write(tflite_model)

def load_tflite_model(save_path):
    with open(save_path, 'rb') as f:
        tflite_model = f.read()
    return tflite_model

def get_gzipped_model_size(tflite_model):
    # Returns size of gzipped model, in bytes.

    with temppathlib.NamedTemporaryFile() as f:
        with temppathlib.NamedTemporaryFile() as zipped_file:
            f.file.write(tflite_model)
            f.file.flush()
            with zipfile.ZipFile(zipped_file.file, 'w', compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr("model.tflite", tflite_model)
            zipped_file.file.flush()
            return zipped_file.path.stat().st_size

if __name__ == "__main__":
    # call get_tflite_model_size with 100 byte string
    print(get_tflite_model_size(b'0'*100))

    # call get_gzipped_model_size with 100000 byte string
    print(get_gzipped_model_size(b'0'*100000))

