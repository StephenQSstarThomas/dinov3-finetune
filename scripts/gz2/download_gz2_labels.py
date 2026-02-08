#!/usr/bin/env python3
"""
Download Galaxy Zoo 2 morphology labels (Hart et al. 2016)
"""

import requests
import gzip
import shutil
from pathlib import Path

# Download URL
url = "https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz"
output_file = "gz2_hart16.csv"

print(f"Downloading GZ2 labels from: {url}")

# Download
response = requests.get(url, stream=True)
gz_file = output_file + ".gz"

with open(gz_file, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print(f"Downloaded to: {gz_file}")

# Extract
print(f"Extracting...")
with gzip.open(gz_file, 'rb') as f_in:
    with open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Clean up
Path(gz_file).unlink()

print(f"\nLabels saved to: {output_file}")
print(f"Contains morphology classifications for 239,695 galaxies")
