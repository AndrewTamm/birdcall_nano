import requests
from pathlib import Path
import os
import json
from collections import defaultdict
import argparse

URL = "https://xeno-canto.org/api/2/recordings?query=cnt:%22United%20States%22"  # Replace with the actual URL

def process_recording(recording):
    """Processes the given recording into a dictionary."""
    return {
        'genus': recording['gen'],
        'species': recording['sp'],
        'name': recording['en'],
        'file': recording['file'],
        'filename': recording['file-name']
    }

def fetch_data(url):
    """Fetches the data from the given URL."""
    page = 1
    results = []

    while True:
        response = requests.get(url, params={"page": page})
        if response.status_code != 200:
            print("Error:", response.status_code)
            break

        data = response.json()
        num_pages = int(data["numPages"])
        recordings = data["recordings"]
        results.extend(map(process_recording, recordings))

        if page == num_pages:
            break
        page += 1

    return results

def count_pairs(results):
    """Counts the number of occurrences of each (genus, species) pair."""
    count_map = defaultdict(int)
    for result in results:
        pair = (result['genus'], result['species'])
        count_map[pair] += 1
    return dict(count_map)

def download_files(results, subdirectory, top_pairs=None):
    """Downloads the files based on the results."""
    os.makedirs(subdirectory, exist_ok=True)

    for result in results:
        if top_pairs is not None and (result['genus'], result['species']) not in top_pairs:
            continue

        file_path = os.path.join(subdirectory, result['filename'])
        if Path(file_path).is_file():
            continue

        try:
            response = requests.get(result['file'])
            response.raise_for_status()
        except (requests.HTTPError, requests.ConnectionError) as e:
            print(f"Failed to download file '{result['filename']}': {e}")
            continue

        try:
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"File '{result['filename']}' downloaded successfully.")

            metadata_path = os.path.join(subdirectory, f"{result['filename']}.json")
            with open(metadata_path, 'w') as metadata_file:
                json.dump(result, metadata_file, indent=4)
            print(f"Metadata file '{result['filename']}.json' saved successfully.")
        except Exception as e:
            print(f"Exception while attempting to save file '{result['filename']}': {e}")
            continue

def main(subdirectory, num_species):
    results = fetch_data(URL)
    count_map = count_pairs(results)
    sorted_count_map = sorted(count_map.items(), key=lambda x: x[1], reverse=True)
    top_pairs = [pair for pair, _ in sorted_count_map if pair[1] != "Mystery"][:num_species]
    print(top_pairs)
    download_files(results, subdirectory, top_pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download birdcall data.")
    parser.add_argument("--subdirectory", default="top_20_species", help="The directory to save the files.")
    parser.add_argument("--num_species", type=int, default=20, help="The number of top species to download.")
    args = parser.parse_args()

    main(args.subdirectory, args.num_species)