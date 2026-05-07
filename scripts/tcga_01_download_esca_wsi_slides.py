import requests
import os
import subprocess
import shutil
import random

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = ROOT / "data" / "raw" / "tcga_esca_wsi"
CANCER_DIR = BASE_DIR / "cancerous"
NORMAL_DIR = BASE_DIR / "non_cancerous"

CANCER_DIR.mkdir(parents=True, exist_ok=True)
NORMAL_DIR.mkdir(parents=True, exist_ok=True)

FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_CLIENT = "gdc-client"  # change to full path if needed


def query_files(sample_type, max_files):
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": ["TCGA-ESCA"]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_type",
                    "value": ["Slide Image"]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_format",
                    "value": ["SVS"]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "cases.samples.sample_type",
                    "value": [sample_type]
                }
            }
        ]
    }

    params = {
        "filters": filters,
        "fields": "file_id,file_name",
        "format": "JSON",
        "size": "1000"
    }

    response = requests.post(FILES_ENDPOINT, json=params)
    response.raise_for_status()

    files = response.json()["data"]["hits"]

    import random

    if len(files) == 0:
        raise ValueError(f"No files found for {sample_type}")

    # 🔴 Tumor → only DX1
    if sample_type == "Primary Tumor":
        filtered_files = [f for f in files if "DX1" in f["file_name"]]
        print(f"[INFO] Found {len(filtered_files)} DX1 files for {sample_type}")

        if len(filtered_files) == 0:
            raise ValueError("No DX1 tumor files found!")

    # 🟢 Normal → allow all (TS1/TSA etc.)
    else:
        filtered_files = files
        print(f"[INFO] Found {len(filtered_files)} normal slide files")

    # ⚠️ Warning if less than requested
    if len(filtered_files) < max_files:
        print(f"[WARNING] Only {len(filtered_files)} files available for {sample_type}")

    # 🎯 Final sampling
    return random.sample(filtered_files, min(len(filtered_files), max_files))


def download_with_gdc_client(files, save_dir, max_files):
    existing = [f for f in os.listdir(save_dir) if f.endswith(".svs")]

    if len(existing) >= max_files:
        print(f"[INFO] Already have {len(existing)} files. Skipping download.")
        return

    for f in files:
        file_id = f["file_id"]
        file_name = f["file_name"]
        final_path = os.path.join(save_dir, file_name)

        if os.path.exists(final_path):
            print(f"[SKIP] {file_name} already exists")
            continue

        print(f"[DOWNLOAD] {file_name}")

        try:
            subprocess.run(
                [
                    GDC_CLIENT,
                    "download",
                    file_id,
                    "-d",
                    str(Path(save_dir).resolve())   # 🔥 ABSOLUTE PATH
                ],
                check=True
            )
        except subprocess.CalledProcessError:
            print(f"[ERROR] Failed to download {file_name}")
            continue

        # Move file out of UUID folder
        subfolder = os.path.join(save_dir, file_id)
        downloaded_file = os.path.join(subfolder, file_name)

        if os.path.exists(downloaded_file):
            shutil.move(downloaded_file, final_path)
            shutil.rmtree(subfolder, ignore_errors=True)

def main():
    print("🔍 Querying cancerous WSIs...")
    tumor_files = query_files("Primary Tumor", max_files=40)

    print("🔍 Querying normal WSIs...")
    normal_files = query_files("Solid Tissue Normal", max_files=15)

    print("\n⬇ Downloading cancerous WSIs...")
    download_with_gdc_client(tumor_files, CANCER_DIR, max_files=40)

    print("\n⬇ Downloading normal WSIs...")
    download_with_gdc_client(normal_files, NORMAL_DIR, max_files=15)

    print("\n✅ DONE: All files downloaded.")


if __name__ == "__main__":
    main()