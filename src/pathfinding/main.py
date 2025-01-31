from typing import TYPE_CHECKING
import os
import sys
import argparse
import time
import logging

from pathfinding.GTFSGraph import GTFSGraph


class PathfindingNamespace(argparse.Namespace):
    # Pathfinding arguments
    start: str
    dest: str
    time: str = None
    # Display arguments
    no_display_console: bool
    output_html: str = False
    follow_railways: bool = False
    # Graph building arguments
    gtfs_path: str
    graph_name: str
    cache_path: str
    force_rebuild_all: bool
    force_rebuild_gtfs: bool
    force_rebuild_railways: bool
    # Misc arguments
    verbose: bool

def fill_path_parser(parser: argparse.ArgumentParser, args_prefix: str = "") -> None:
    # Pathfinding arguments
    if not args_prefix:
        parser.add_argument('start', type=str, help='Name of the departure city')
        parser.add_argument('dest', type=str, help='Name of the destination city')
    parser.add_argument('-t', f'--{args_prefix}time', dest='time', type=str, default=None, help='Time of departure. Default to now. Format is HH:MM:SS.')

    # Display arguments
    parser.add_argument(f'--{args_prefix}no-display-console', default=False, action='store_true', help='Do not print the path instructions on the console')
    parser.add_argument(f'--{args_prefix}output-html', type=str, default=None, help='Path to a html output file. If this is not set the path will not be saved to a file')
    parser.add_argument(f'--{args_prefix}follow-railways', default=False, action='store_true', help='Follow the railways instead of drawing straight lines between stations (requires --output-html)')

    # Graph building arguments
    parser.add_argument('-g', f'--{args_prefix}gtfs-path', type=str, default=None, help='Path to the data folder containing the GTFS files')
    parser.add_argument('-n', f'--{args_prefix}graph-name', type=str, default='sncf-light', help='Name of the graph to build')
    parser.add_argument(f'--{args_prefix}cache-path', type=str, default=None, help='Path to the cache folder')
    parser.add_argument(f'--{args_prefix}force-rebuild-all', default=False, action='store_true', help='Force the rebuild of all the graphs')
    parser.add_argument(f'--{args_prefix}force-rebuild-gtfs', default=False, action='store_true', help='Force the rebuild of the gtfs graph')
    parser.add_argument(f'--{args_prefix}force-rebuild-railways', default=False, action='store_true', help='Force the rebuild of the railways graph')
    
    # Misc arguments
    parser.add_argument('-v', f'--{args_prefix}verbose', default=False, action='store_true', help='Increase output verbosity')

class Pathfinder:
    def __init__(self, args: PathfindingNamespace):
        if not args.gtfs_path:
            args.gtfs_path = os.path.join(os.path.dirname(__file__), '..','..', 'data', 'pathfinding')
        if not args.cache_path:
            args.cache_path = os.path.join(os.path.dirname(__file__), '..','..', 'data', 'pathfinding', 'cache')
        self.args = args
        self.data_path = os.path.abspath(args.gtfs_path)
        self._graph : GTFSGraph = None


    @property
    def graph(self) -> GTFSGraph:
        if not self._graph:
            if self.args.verbose:
                print("Building/Loading the gtfs graph...", file=sys.stderr)
                start = time.time()
            self._graph = GTFSGraph(self.data_path, self.args.graph_name, self.args.cache_path, self.args.verbose)
            self._graph.build_graph(force=self.args.force_rebuild_all or self.args.force_rebuild_gtfs)
            self._graph.clear_build_context()
            if self.args.verbose:
                print("Took %s seconds to build/load the gtfs graph" % (time.time() - start), file=sys.stderr)

            if self.args.follow_railways:
                if self.args.verbose:
                    start = time.time()
                self._graph.build_tracks_graph(force=self.args.force_rebuild_all or self.args.force_rebuild_railways)
                if self.args.verbose:
                    print("Took %s seconds to build/load the railways graph" % (time.time() - start), file=sys.stderr)
            
        return self._graph
    
    def compute_path(self, start: str, dest: str):
        weight, path = self.graph.compute_path_from_names(start, dest, False, self.args.time)
        if self.args.output_html or not self.args.no_display_console:
            self.graph.display_path(path, weight, not self.args.no_display_console, self.args.output_html)

    
    def run(self):
        self.compute_path(self.args.start, self.args.dest)


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] - %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
                    prog='SNCF-Pathfinding',
                    description='Compute the shortest path between two cities in France')
    fill_path_parser(parser)
    args: PathfindingNamespace = parser.parse_args()
    pth = Pathfinder(args)
    pth.run()

if __name__ == '__main__':
    main()
