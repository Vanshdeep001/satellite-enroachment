"""
LEVIR-CD Dataset Download Utility.
Downloads the official LEVIR-CD split archives (train.zip, val.zip, test.zip)
from the official Google Drive distribution hosted by the dataset authors.
"""

import os
import sys
import time
import zipfile
import shutil
import urllib.request
import re

OFFICIAL_FILES = {
    'val.zip': '1BqSt4ueO7XAyQ_84mUjswUSJt13ZBuzG',
    'test.zip': '1jj3qJD_grJlgIhUWO09zibRGJe0R4Tn0',
    'train.zip': '18GuoCuBn48oZKAlEo-LrNwABrFhVALU-'
}

def get_confirm_download_url(file_id):
    """Retrieve the direct download link with confirmation token from Google Drive."""
    base_url = f'https://drive.google.com/uc?id={file_id}&export=download'
    req = urllib.request.Request(base_url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        action_match = re.search(r'action=\"(https://drive\.usercontent\.google\.com/download[^\"]*)\"', html)
        uuid_match = re.search(r'name=\"uuid\" value=\"([^\"]*)\"', html)
        if action_match and uuid_match:
            action = action_match.group(1)
            uuid = uuid_match.group(1)
            return f'{action}?id={file_id}&export=download&confirm=t&uuid={uuid}'
            
        # Check if direct download was already served
        cl = resp.headers.get('Content-Length')
        ct = resp.headers.get('Content-Type')
        if ct and 'octet-stream' in ct:
            return base_url
            
    raise RuntimeError(f"Could not parse direct download URL for file ID: {file_id}")

def download_file(url, target_path, filename):
    """Download a file with chunked streaming and progress reporting."""
    print(f"[*] Connecting for {filename}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req, timeout=60) as resp:
        total_size = resp.headers.get('Content-Length')
        total_size = int(total_size) if total_size else 0
        total_mb = total_size / (1024 * 1024)
        
        print(f"[*] Downloading {filename} (Size: {total_mb:.1f} MB)...")
        chunk_size = 1024 * 1024  # 1MB chunks
        downloaded = 0
        last_log = time.time()
        start_time = time.time()
        
        with open(target_path, 'wb') as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                
                now = time.time()
                if now - last_log > 5.0 or downloaded == total_size:
                    elapsed = now - start_time
                    speed_mb = (downloaded / (1024 * 1024)) / max(elapsed, 0.1)
                    percent = (downloaded / total_size * 100) if total_size else 0
                    print(f"    [{filename}] {downloaded / (1024*1024):.1f}/{total_mb:.1f} MB ({percent:.1f}%) @ {speed_mb:.2f} MB/s")
                    last_log = now

    print(f"[+] Download complete: {filename}")

def verify_zip(zip_path):
    """Verify integrity of a zip archive."""
    print(f"[*] Verifying integrity of {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        bad_file = zf.testzip()
        if bad_file:
            raise ValueError(f"Corrupted file inside archive {zip_path}: {bad_file}")
        num_files = len(zf.namelist())
    print(f"[+] Integrity verified. Archive contains {num_files} entries.")
    return True

if __name__ == '__main__':
    raw_dir = os.path.abspath('data/raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    file_to_download = sys.argv[1] if len(sys.argv) > 1 else 'val.zip'
    if file_to_download not in OFFICIAL_FILES:
        print(f"Unknown file: {file_to_download}. Choose from: {list(OFFICIAL_FILES.keys())}")
        sys.exit(1)
        
    fid = OFFICIAL_FILES[file_to_download]
    out_path = os.path.join(raw_dir, file_to_download)
    
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000000:
        print(f"[*] File {file_to_download} already exists ({os.path.getsize(out_path)/(1024*1024):.1f} MB). Verifying...")
        try:
            verify_zip(out_path)
            print("[+] Existing file is valid, skipping download.")
            sys.exit(0)
        except Exception as e:
            print(f"[-] Existing file is invalid ({e}), redownloading...")
            os.remove(out_path)
            
    dl_url = get_confirm_download_url(fid)
    download_file(dl_url, out_path, file_to_download)
    verify_zip(out_path)

