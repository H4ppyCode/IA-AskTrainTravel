from typing import TYPE_CHECKING
import argparse
import subprocess
import pandas as pd

if TYPE_CHECKING:
    from dataclasses import dataclass
    
    @dataclass
    class Args(argparse.Namespace):
        nlp_output: str
        gtfs_path: str
        graph_name: str
        cache_path: str


def main():
    parser = argparse.ArgumentParser(
                    prog='Nlp to SNCF path')
    # NLP
    parser.add_argument('nlp_output', type=str, help='Nlp output filepath in csv format: sequence_id,start,dest')

    # Pathfinding
    parser.add_argument('-g', '--gtfs-path', type=str, default='gtfs', help='Path to the data folder containing the GTFS files')
    parser.add_argument('-n', '--graph-name', type=str, default='sncf-light', help='Name of the graph to build')
    parser.add_argument('--cache-path', type=str, default='gtfs-cache', help='Path to the cache folder')

    args: Args = parser.parse_args()

    # Read the nlp output
    df = pd.read_csv(args.nlp_output)
    for index, row in df.iterrows():
        if row['departure'] == 'NOT_TRIP' or row['departure'] == 'NOT_FRENCH':
            continue
        print(f"{row['sequence_id']}: {row['departure']} -> {row['arrival']}")
        subprocess.run(f"python src/pathfinding/main.py -g {args.gtfs_path} -n {args.graph_name} --cache-path {args.cache_path} {row['departure']} {row['arrival']} ", shell=True)
        print("\n")


        

if __name__ == "__main__":
    main()
