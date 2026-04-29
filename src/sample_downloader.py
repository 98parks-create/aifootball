import requests
import os

def download_sample_video():
    url = "https://drive.google.com/uc?export=download&id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"
    save_path = os.path.join("data", "uploads", "sample.mp4")
    
    if os.path.exists(save_path):
        print(f"File already exists at {save_path}")
        return

    print(f"Downloading sample video from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete!")
    else:
        print(f"Failed to download video. Status code: {response.status_code}")

if __name__ == "__main__":
    download_sample_video()
