import os
import csv
import urllib.parse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger("benchmark.dataset")

class CSVDataset:
    def __init__(self, csv_path: str, cache_dir: str):
        self.csv_path = csv_path
        self.cache_dir = cache_dir
        self.urls = []
        self.local_paths = []
        
        # Load CSV and extract URLs
        if os.path.exists(csv_path):
            try:
                with open(csv_path, mode='r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    # Check if 'Image URL' column is present, otherwise fallback to first column
                    col_name = 'Image URL'
                    if reader.fieldnames and col_name not in reader.fieldnames:
                        col_name = reader.fieldnames[0]
                    
                    for row in reader:
                        url = row.get(col_name)
                        if url and url.startswith("http"):
                            self.urls.append(url.strip())
                logger.info(f"Loaded {len(self.urls)} image URLs from {csv_path}")
            except Exception as e:
                logger.error(f"Error reading CSV file {csv_path}: {e}")
        else:
            logger.warning(f"CSV file {csv_path} does not exist. Benchmark will run in mock mode or require local folder.")

    def download_and_cache(self, max_download: int = 1000, num_workers: int = 16) -> list:
        """
        Downloads images in parallel to the local cache directory.
        Returns the list of local file paths that exist.
        """
        if not self.urls:
            # Check if there are already images in cache_dir
            if os.path.exists(self.cache_dir):
                existing = [os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if existing:
                    logger.info(f"No URLs loaded, but found {len(existing)} existing images in cache.")
                    self.local_paths = existing
                    return self.local_paths
            return []

        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Select URLs up to max_download
        urls_to_process = self.urls[:max_download]
        
        # Check which files already exist
        to_download = []
        local_files = []
        
        for url in urls_to_process:
            parsed_url = urllib.parse.urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename:
                filename = f"image_{hash(url)}.jpg"
            
            dest_path = os.path.abspath(os.path.join(self.cache_dir, filename))
            local_files.append(dest_path)
            
            if not os.path.exists(dest_path):
                to_download.append((url, dest_path))
        
        if to_download:
            logger.info(f"Downloading {len(to_download)} new images to {self.cache_dir} (Parallel workers: {num_workers})...")
            
            def download_one(url_dest):
                url, dest = url_dest
                try:
                    r = requests.get(url, timeout=15)
                    if r.status_code == 200:
                        with open(dest, 'wb') as f:
                            f.write(r.content)
                        return True
                    else:
                        logger.debug(f"Failed downloading {url}: HTTP {r.status_code}")
                except Exception as e:
                    logger.debug(f"Error downloading {url}: {e}")
                return False

            success_count = 0
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(download_one, item): item for item in to_download}
                for fut in as_completed(futures):
                    if fut.result():
                        success_count += 1
            
            logger.info(f"Successfully downloaded {success_count}/{len(to_download)} images.")
        else:
            logger.info("All selected images already cached locally.")

        # Gather final list of successfully downloaded/existing local files
        self.local_paths = [path for path in local_files if os.path.exists(path)]
        logger.info(f"Dataset has {len(self.local_paths)} valid cached images ready for load testing.")
        return self.local_paths
