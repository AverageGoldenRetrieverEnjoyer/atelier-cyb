import zipfile
import os

source_dir = "/path/to/zip/files"
output_base_dir = "/path/to/extracted"

# Make sure the output directory exists
os.makedirs(output_base_dir, exist_ok=True)

for file_name in os.listdir(source_dir):
    if file_name.endswith(".zip"):
        zip_path = os.path.join(source_dir, file_name)
        # Create a unique folder for each zip
        folder_name = os.path.splitext(file_name)[0]
        output_dir = os.path.join(output_base_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extracted {file_name} to {output_dir}")
