# type: ignore
#
# CREDIT: https://stackoverflow.com/a/39225039
#

import requests
from tqdm import tqdm


def progress_bar(some_iter):
    try:
        return tqdm(some_iter)
    except ModuleNotFoundError:
        return some_iter


def download_file_from_google_drive(file_id, destination):
    print(f"Trying to fetch {destination}")

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in progress_bar(response.iter_content(CHUNK_SIZE)):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python download.py drive_file_id destination_file_path")
    else:
        # TAKE ID FROM SHAREABLE LINK
        FILE_ID = sys.argv[1]
        # DESTINATION FILE ON YOUR DISK
        DESTINATION = sys.argv[2]
        download_file_from_google_drive(FILE_ID, DESTINATION)
