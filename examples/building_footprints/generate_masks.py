import argparse
import os
import glob

import tqdm
from joblib import Parallel, delayed

import numpy as np

from utils import read_tile, mask_from_tile

def main(args):
    data_path = args.data_path

    # Process code for paralleization
    def process_file(file):
        # Get the filename for the mask data
        mask_path = os.path.join(data_path, "testmask")
        filename = os.path.basename(file)
        keyname = filename.split('.')[0]
        label_name = keyname.replace("8band_", "Mask_") + ".npy"
        write_path = os.path.join(mask_path, label_name)
        
        # Don't process existing files
        if os.path.exists(write_path):
            return
        
        try:
            data, transform = read_tile(file)
            mask = mask_from_tile(file, data.shape, transform)
        except:
            print(f"Failed to create mask for {file}")
            return
                
        np.save(write_path, mask)

    mask_path = os.path.join(data_path, "testmask")
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
       
    # Run the job in parallel
    files = glob.glob(os.path.join(data_path, "8band/*.tif"))
    Parallel(n_jobs=16)(delayed(process_file)(file) for file in tqdm.tqdm(files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a set of masks from geojson labels.")

    parser.add_argument(
        "data_path", type=str, help="Path to the root of the SpaceNet-1 data."
    )
    args = parser.parse_args()

    main(args)