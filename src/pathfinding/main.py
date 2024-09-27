from typing import TYPE_CHECKING
import os
import sys
import argparse
import time

from GTFSGraph import GTFSGraph


if TYPE_CHECKING:
    from dataclasses import dataclass
    
    @dataclass
    class Args(argparse.Namespace):
        # Pathfinding arguments
        start: str
        dest: str
        time: str
        # Display arguments
        no_display_console: bool
        output_html: str
        follow_railways: bool
        # Graph building arguments
        gtfs_path: str
        graph_name: str
        cache_path: str
        force_rebuild_all: bool
        force_rebuild_gtfs: bool
        force_rebuild_railways: bool
        # Misc arguments
        verbose: bool

def main():
    parser = argparse.ArgumentParser(
                    prog='SNCF-Pathfinding',
                    description='Compute the shortest path between two cities in France')

    # Pathfinding arguments
    parser.add_argument('start', type=str, help='Name of the departure city')
    parser.add_argument('dest', type=str, help='Name of the destination city')
    parser.add_argument('-t', '--time', dest='time', type=str, default=None, help='Time of departure. Default to now. Format is HH:MM:SS.')

    # Display arguments
    parser.add_argument('--no-display-console', default=False, action='store_true', help='Do not print the path instructions on the console')
    parser.add_argument('--output-html', type=str, default=None, help='Path to a html output file. If this is not set the path will not be saved to a file')
    parser.add_argument('--follow-railways', default=False, action='store_true', help='Follow the railways instead of drawing straight lines between stations (requires --output-html)')

    # Graph building arguments
    parser.add_argument('-g', '--gtfs-path', type=str, default='gtfs', help='Path to the data folder containing the GTFS files')
    parser.add_argument('-n', '--graph-name', type=str, default='sncf-light', help='Name of the graph to build')
    parser.add_argument('--cache-path', type=str, default='gtfs-cache', help='Path to the cache folder')
    parser.add_argument('--force-rebuild-all', default=False, action='store_true', help='Force the rebuild of all the graphs')
    parser.add_argument('--force-rebuild-gtfs', default=False, action='store_true', help='Force the rebuild of the gtfs graph')
    parser.add_argument('--force-rebuild-railways', default=False, action='store_true', help='Force the rebuild of the railways graph')
    
    # Misc arguments
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase output verbosity')

    args: "Args" = parser.parse_args()


    data_path = os.path.abspath(args.gtfs_path)
    graph = GTFSGraph(data_path, args.graph_name, args.cache_path, args.verbose)

    if args.verbose:
        print("Building/Loading the gtfs graph...", file=sys.stderr)
        start = time.time()
    graph.build_graph(force=args.force_rebuild_all or args.force_rebuild_gtfs)
    graph.clear_build_context()
    if args.verbose:
        print("Took %s seconds to build/load the gtfs graph" % (time.time() - start), file=sys.stderr)
    
    if args.follow_railways:
        if args.verbose:
            start = time.time()
        graph.build_tracks_graph(force=args.force_rebuild_all or args.force_rebuild_railways)
        if args.verbose:
            print("Took %s seconds to build/load the railways graph" % (time.time() - start), file=sys.stderr)

    weight, path = graph.compute_path_from_names(args.start, args.dest, False, args.time)
    if args.output_html or not args.no_display_console:
        graph.display_path(path, weight, not args.no_display_console, args.output_html)

if __name__ == '__main__':
    main()
