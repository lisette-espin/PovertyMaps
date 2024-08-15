# export PYTHONPATH="${PYTHONPATH}:../libs"

import os
import time
import argparse
import pandas as pd
from tqdm import tqdm
from pqdm.processes import pqdm

from utils import ios
from maps.reversegeocode import ReverseGeocode

##############################################################################################################
# Functions
##############################################################################################################

CHUNK_SIZE = 1000000
CHUNK_DIRNAME = 'chunks'
CHUNK_FNAME = 'chunk'
COUNTRY_DIRNAME = 'country'


def run(fn_zip, n_jobs):
    fn_zip = os.path.abspath(fn_zip)

    # assign country to each cell tower
    print("=========================================")
    ios.printf("1. Decompressing and Chunking files")
    chunks_path = split_file_into_chunks(fn_zip)
    chunk_files = [os.path.join(chunks_path, fn) for fn in os.listdir(chunks_path) if fn.startswith(CHUNK_FNAME)]
    print(f"{len(chunk_files)} chunk files in {chunks_path}")

    # assign country to each cell tower
    print("=========================================")
    ios.printf("2. Assigning country to each cell tower")
    assign_country(chunk_files, chunks_path, n_jobs)

    # create file for each country
    print("=========================================")
    ios.printf("3. Create file for each country")
    split_by_country(chunk_files, chunks_path)

    # remove chunks
    print("=========================================")
    ios.printf("4. Remove chunks (tmp files)")
    remove_chunks(chunks_path)

    # copy country files into main directory
    print("=========================================")
    ios.printf("5. Copy country files into main directory")
    move_country_files(fn_zip)
    return


def move_country_files(fn_zip):
    root = os.path.dirname(fn_zip)
    cmd = f"""
            cp -r {root}/{COUNTRY_DIRNAME} {root}/../../
            """
    execute(cmd)


def split_file_into_chunks(fn_zip):
    # decompress
    root = os.path.dirname(fn_zip)
    fn_csv = fn_zip.replace('.gz', '')
    cmd = f"""
            cd {root}
            gzip -d < {fn_zip} > {fn_csv}
            """
    execute(cmd)

    # split
    cmd = f"""
            cd {root}
            mkdir -p {CHUNK_DIRNAME}
            split -{CHUNK_SIZE} {os.path.basename(fn_csv)} {CHUNK_DIRNAME}/{CHUNK_FNAME}.
            ls {CHUNK_DIRNAME}/* | wc -lc
            """
    execute(cmd)
    return os.path.join(root, CHUNK_DIRNAME)


def get_chunk_files(lst, n):
    ### https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks?page=1&tab=votes#tab-top
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def remove_chunks(chunks_path):
    root = os.path.dirname(chunks_path)

    # merge files into a single one with country
    cmd = f"""
            cd {chunks_path}
            cat {CHUNK_FNAME}.?? > {root}/cell_towers_full.csv
            """
    execute(cmd)

    # remove chunk files
    cmd = f"""
            cd {root}
            rm -rf {CHUNK_DIRNAME}
            """
    execute(cmd)


def assign_country(chunk_files, chunks_path, n_jobs):
    _ = pqdm(chunk_files, ReverseGeocode.assign_country, n_jobs=n_jobs)


def split_by_country(chunk_files, chunks_path):
    root = os.path.dirname(chunks_path)
    output = os.path.join(root, COUNTRY_DIRNAME)
    ios.validate_path(output)

    for fn in tqdm(chunk_files):
        df_chunk = pd.read_csv(fn, index_col=None)
        for country, df in df_chunk.groupby('country'):
            new_fn = os.path.join(output, "cell_towers_{}.csv".format(country))
            df.to_csv(new_fn, mode='a', header=not os.path.exists(new_fn))


def execute(cmd):
    print(cmd)
    stdout = os.system(cmd)
    print(stdout)


##############################################################################################################
# Main
##############################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fnzip", help="File where the cell_towers.csv file is located", type=str, required=True)
    parser.add_argument("-njobs", help="Number of processes to run in parallel", type=int, default=1, required=False)

    args = parser.parse_args()
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))

    start_time = time.time()
    run(args.fnzip, args.njobs)
    print("--- %s seconds ---" % (time.time() - start_time))
