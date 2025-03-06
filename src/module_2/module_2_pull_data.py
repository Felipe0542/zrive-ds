import boto3
import botocore
import logging
import os

# Logging configuration
logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("s3_download.log"),
        logging.StreamHandler()
    ]
)

# S3 bucket details
bucket_name = "zrive-ds-data"
local_directory = "/mnt/d/zrive-ds/src/module_2/files_downloaded"  # WSL-compatible path

# List of files to download with their S3 prefixes
files_to_download = {
    "groceries/sampled-datasets/orders.parquet": "orders.parquet",
    "groceries/sampled-datasets/regulars.parquet": "regulars.parquet",
    "groceries/sampled-datasets/abandoned_carts.parquet": "abandoned_carts.parquet",
    "groceries/sampled-datasets/inventory.parquet": "inventory.parquet",
    "groceries/sampled-datasets/users.parquet": "users.parquet",
    "groceries/box_builder_dataset/feature_frame.csv": "feature_frame.csv"
}

# Create S3 client
s3 = boto3.client('s3')

# Function to download specific files from S3
def download_selected_files():
    try:
        os.makedirs(local_directory, exist_ok=True)  # Create local folder if it doesn't exist
        logging.info("Starting download of selected files...")

        for s3_key, local_file_name in files_to_download.items():
            local_path = os.path.join(local_directory, local_file_name)
            logging.info(f"Downloading {s3_key} to {local_path}...")
            s3.download_file(bucket_name, s3_key, local_path)
            logging.info(f"Successfully downloaded: {local_file_name}")

        logging.info("All files downloaded successfully.")

    except botocore.exceptions.ClientError as e:
        logging.error(f"Error accessing S3: {e}")

if __name__ == "__main__":
    download_selected_files()
