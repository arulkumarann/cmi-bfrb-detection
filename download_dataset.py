import os
import subprocess
         

competition_name = "cmi-detect-behavior-with-sensor-data"
files_to_download = [
    "train.csv",
    "train_demographics.csv",
    "test.csv",
    "test_demographics.csv"
    ]
download_path = "./dataset"

os.makedirs(download_path, exist_ok=True)

for file_name in files_to_download:
    print(f"Downloading {file_name}...")
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", competition_name,
        "-f", file_name,
        "-p", download_path,
        "--force"  
    ], check=True)
    print(f"{file_name} downloaded!")

print("All files downloaded successfully.")
