import requests
from pathlib import Path
import os
import json

url = "https://xeno-canto.org/api/2/recordings?query=cnt:%22United%20States%22"  # Replace with the actual URL

# Function to process each recording object
def process_recording(recording):
    return {
        'genus': recording['gen'],
        'species': recording['sp'],
        'name': recording['en'],
        'file': recording['file']
    }

def fetch_data():
    page = 4
    results = []

    while True:
        response = requests.get(url, params={"page": page})
        if response.status_code == 200:
            data = response.json()
            num_pages = int(data["numPages"])
            recordings = data["recordings"]

            # Process each recording object
            results.extend([{
                'genus': recording['gen'],
                'species': recording['sp'],
                'name': recording['en'],
                'file': recording['file'],
                'filename': recording['file-name']
            } for recording in recordings])

            if page == num_pages:
                break  # Exit the loop if all pages have been fetched

            page += 1  # Increment page for the next request
        else:
            print("Error:", response.status_code)
            break  # Stop fetching data in case of an error
    return results


def download_files(results, subdirectory):
    os.makedirs(subdirectory, exist_ok=True)  # Create subdirectory if it doesn't exist

    for result in results:
        file_url = result['file']
        filename = result['filename']
        file_path = os.path.join(subdirectory, filename)  # Construct the file path

        file = Path(file_path)
        if file.is_file():
            continue

        try:
            # Download and save the file
            response = requests.get(file_url)
            if response.status_code == 200:
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                    print(f"File '{filename}' downloaded successfully.")
            else:
                print(f"Failed to download file '{filename}'. Status code: {response.status_code}")
                continue
        except Exception as e:
            print(f"Exception while attempting to download file '{filename}'. Exception: {e}")
            continue

        # Save metadata as a separate JSON file
        metadata_filename = f"{filename}.json"
        metadata_path = os.path.join(subdirectory, metadata_filename)
        with open(metadata_path, 'w') as metadata_file:
            json.dump(result, metadata_file, indent=4)
            print(f"Metadata file '{metadata_filename}' saved successfully.")


if __name__ == "__main__":
    results = fetch_data()
    download_files(results, 'calls')
