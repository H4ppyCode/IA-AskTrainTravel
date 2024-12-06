from typing import Optional, Union, Any, Dict, List
import os
import sys
import logging
import pandas as pd
import networkx as nx
import time
import pickle
import datetime
import bisect
import json
import re
from geopy.distance import geodesic
from rtree import index
from shapely.geometry import Point

from enum import Enum
from dataclasses import dataclass

import folium
from folium.plugins import MarkerCluster


@dataclass
class GPSPosition:
    lat: float
    lon: float

    def __eq__(self, other: 'GPSPosition'):
        return self.lat == other.lat and self.lon == other.lon
    
@dataclass
class NodeDisplayData:
    pos: GPSPosition
    name: str
    
class NodeType(Enum):
    # Stop area (station) with no specific time.
    # ID: [area_id]
    # name: stop_name of the area
    # pos: GPSPosition of the area
    AREA = "area"
    # Stop area (station) at a given time specific time. 
    # ID: [area_id]_[time]
    AREA_TIME_POINT = "area_time_point"
    # Stop for a specific trip (so specific time)
    # ID: [trip_id]_[stop_id]
    # time: arrival time at the stop
    # area: area_id of the stop
    TRIP_STOP = "trip_stop"

    def __eq__(self, other: 'NodeType'):
        return self.value == other.value
    
    

class EdgeType(Enum):
    TRIP = "trip" # Edge between two trip stops
    TRANSFER = "transfer" # Edge between two stops times (transfer between two trips)
    TIMEPOINT = "timepoint" # Bidirectionnal edge between a trip stop and a time point, always with 0 weight

    def __eq__(self, other: 'EdgeType'):
        return self.value == other.value

def time_to_seconds(time_str, enforce_24h: bool = False):
    h, m, s = map(int, time_str.split(':'))
    res = h * 3600 + m * 60 + s
    if enforce_24h and h > 23:
        res -= 24 * 3600
    return res

def seconds_to_time(seconds: int) -> datetime.time:
    return datetime.time(seconds // 3600, (seconds // 60) % 60, seconds % 60)


def find_closest_timepoint(source_time: Union[str, datetime.time], time_objects: List[datetime.time]) -> datetime.time:
    if isinstance(source_time, str):
        source_time = datetime.datetime.strptime(source_time, '%H:%M:%S').time()
    
    idx = bisect.bisect_left(time_objects, source_time)
    
    # If no time is found after the given time, return the first time (of the next day)
    if idx < len(time_objects):
        return time_objects[idx] 
    else:
        return time_objects[0]

def normalize_tracks(tracks):
    for track in tracks:
        geometry = track['geo_shape']['geometry']
        coords = geometry['coordinates']
        if not isinstance(coords[0][0], list):
            coords = [coords]
        coords = [list(map(lambda x: (x[1], x[0]), line)) for line in coords]
        geometry['coordinates'] = coords

class GTFSGraph:
    def __init__(self, data_directory: str, name: str = None, cache_directory: str = None, verbose: bool = False):
        self.data_directory: str = os.path.abspath(data_directory)
        self.verbose: bool = verbose
        if cache_directory:
            self.cache_directory = os.path.abspath(cache_directory)
        else:
            self.cache_directory: str = os.path.join(self.data_directory, 'cache')
        self.name: str = name
        if not name:
            self.name = os.path.basename(data_directory)
        self.graph: Optional[nx.DiGraph] = None
        # Key: stop_id, Value: List of node_id for time points at this stop area
        self.time_points: Dict[str, List[str]] = {}
        
        ### Lazy loaded properties
        self._stops_df: pd.DataFrame = None
        self._stop_times_df: pd.DataFrame = None
        self._trips_df: pd.DataFrame = None
        self._routes_df: pd.DataFrame = None
        # Key: stop_id, Value: stop data
        self._stops_dict: Dict[str, Dict] = None

        ### Data for the tracks graph
        self.tracks_graph: Optional[nx.Graph] = None
        self.rtree_idx: Optional[index.Index] = None
        # Key: index in the R-tree, Value: (track, line index, point index)
        self.points_map: Optional[Dict[int, Any]] = None

    def clear_build_context(self):
        self.time_points = {}
        self._stops_dict = None

    @property
    def stops_df(self) -> pd.DataFrame:
        if self._stops_df is None:
            self._stops_df = self.read_csv_file('stops.txt')
            self._stops_df.drop_duplicates(subset='stop_id', keep='first', inplace=True)
        return self._stops_df
    
    @property
    def stop_times_df(self) -> pd.DataFrame:
        if self._stop_times_df is None:
            self._stop_times_df = self.read_csv_file('stop_times.txt')
            self._stop_times_df.sort_values(['trip_id', 'stop_sequence'], inplace=True)
        return self._stop_times_df
    
    @property
    def trips_df(self) -> pd.DataFrame:
        if self._trips_df is None:
            self._trips_df = self.read_csv_file('trips.txt')
        return self._trips_df
    
    @property
    def routes_df(self) -> pd.DataFrame:
        if self._routes_df is None:
            self._routes_df = self.read_csv_file('routes.txt')
        return self._routes_df

    @property
    def stops_dict(self) -> Dict[str, Dict]:
        if not self._stops_dict:
            self._stops_dict = self.stops_df.set_index('stop_id').to_dict('index')
        return self._stops_dict
    
    @property
    def graph_filepath(self):
        return os.path.join(self.cache_directory, 'gtfs-graph-%s.gpickle' % self.name)

    def read_csv_file(self, filename: str):
        return pd.read_csv(os.path.join(self.data_directory, filename), comment='#')

    def find_area_time_point(self, stop, time: str):
        if not (isinstance(stop['parent_station'], str) and stop['parent_station'] != ''):
            raise Exception(f"Stop {stop['stop_id']} has no parent station")
        
        area = stop['parent_station']
        node_id = f"{area}_{time}"
        if not self.graph.has_node(node_id):
            node_name = self.stops_dict[area]['stop_name']
            self.graph.add_node(node_id, type=NodeType.AREA_TIME_POINT)
            if area not in self.time_points:
                self.time_points[area] = []
                # Create the area node for destination node
                self.graph.add_node(area, name=node_name, pos=GPSPosition(stop['stop_lat'], stop['stop_lon']), type=NodeType.AREA)
            self.graph.add_edge(node_id, area, weight=0, type=EdgeType.TIMEPOINT)
            self.time_points[area].append(node_id)

        return node_id
    
    def add_trips_nodes(self):
        if self.verbose:
            start_time = time.time()
            nb_trips = len(self.stop_times_df['trip_id'].unique())
            logging.debug(f"Adding edges for {nb_trips} trips")
            i = 0

        for _, trip in self.stop_times_df.groupby('trip_id'):
            previous_row = None
            if self.verbose:
                i += 1
                if (i % 1000) == 0:
                    logging.debug(f"{i} trips done ({time.time() - start_time:.2f}s)")
            for _, row in trip.iterrows():
                # get the associated stop 
                stop = self.stops_dict[row['stop_id']] 
                # TODO: Handle delta time between arrival and departure of a stop
                time_point_id = self.find_area_time_point(stop, row['arrival_time'])

                # Create the node and link it to the area time point
                node_id = f"{row['trip_id']}_{row['stop_id']}"
                self.graph.add_node(node_id, area=time_point_id.rsplit('_', 1)[0], time=row['arrival_time'], type=NodeType.TRIP_STOP)
                self.graph.add_edge(node_id, time_point_id, weight=0, type=EdgeType.TIMEPOINT)
                self.graph.add_edge(time_point_id, node_id, weight=0, type=EdgeType.TIMEPOINT)
                    
                # If this is not the first stop of the trip, add an edge between the previous stop and this one
                if previous_row is not None:
                    previous_node_id = f"{previous_row['trip_id']}_{previous_row['stop_id']}"
                    weight = time_to_seconds(row['departure_time']) - time_to_seconds(previous_row['arrival_time'])
                    self.graph.add_edge(previous_node_id, node_id, weight=weight, trip=row['trip_id'], type=EdgeType.TRIP)
                previous_row = row
    
    def sort_time_points(self):
        # Sort time points by time
        for area, points in self.time_points.items():
            def time_key(x: str):
                time_part: str = x.rsplit('_', 1)[1]
                hpart = int(time_part.split(':', 1)[0])
                if hpart > 23:
                    time_part = f"{hpart - 24:02d}{time_part[2:]}"
                return time_part

            self.time_points[area] = sorted(points, key=time_key)

    def add_transfer_edges(self):
        for points in self.time_points.values():
            if len(points) < 2:
                continue
            first_tp_time = points[0].rsplit('_', 1)[-1]
            tp_time = first_tp_time
            for i in range(len(points) - 1):
                next_tp_time = points[i + 1].rsplit('_', 1)[-1]
                weight = time_to_seconds(next_tp_time, enforce_24h=True) - time_to_seconds(tp_time, enforce_24h=True)
                self.graph.add_edge(points[i], points[i + 1], weight=weight, type=EdgeType.TRANSFER)
                tp_time = next_tp_time
            # Add a transfer edge between the last and first time points considering a new day to allow overnight transfers
            # Formula: time between last time point and midnight (ie 24h - last time) + time between midnight and first time point (ie first time)
            weight = (24 * 3600) - time_to_seconds(tp_time, enforce_24h=True) + time_to_seconds(first_tp_time, enforce_24h=True)
            self.graph.add_edge(points[i], points[i + 1], weight=weight, type=EdgeType.TRANSFER)

    def build_graph(self, force: bool = False) -> nx.DiGraph:
        # Check if the graph already exists
        if not force and os.path.exists(self.graph_filepath):
            if self.verbose:
                logging.debug("Loading existing graph from %s" % self.graph_filepath)
            self.graph = pickle.load(open(self.graph_filepath, 'rb'))
            return self.graph

        # Create a directed graph
        self.graph = nx.DiGraph()

        # Add nodes for each trip stop
        self.add_trips_nodes()

        # Sort time points by time
        self.sort_time_points()

        # Add edges between time points
        self.add_transfer_edges()
        
        # Save the graph
        if not os.path.isdir(os.path.dirname(self.graph_filepath)):
            os.makedirs(os.path.dirname(self.graph_filepath))
        pickle.dump(self.graph, open(self.graph_filepath, 'wb'))
        return self.graph
    
    def build_tracks_graph(self, force: bool = False):
        """
        Build an R-tree index of all points from the LineString geometries in the tracks.
        """
        rtree_directory = os.path.join(self.cache_directory, 'rtree')
        graph_filepath = os.path.join(rtree_directory, 'tracks_graph.gpickle')
        rtree_filepath = os.path.join(rtree_directory, 'rtree')
        point_map_filepath = os.path.join(rtree_directory, 'point_map.gpickle')

        if not os.path.isdir(rtree_directory):
            os.makedirs(rtree_directory)
        elif force:
            if os.path.isfile(rtree_filepath + '.dat'):
                os.remove(rtree_filepath + '.dat')
                os.remove(rtree_filepath + '.idx')

        self.rtree_idx = index.Index(rtree_filepath)
        # Check if the graph already exists
        if not force and os.path.isfile(graph_filepath):
            if self.verbose:
                logging.debug("Loading existing tracks graph from %s" % graph_filepath)
            self.tracks_graph = pickle.load(open(graph_filepath, 'rb'))
            self.points_map = pickle.load(open(point_map_filepath, 'rb'))
            return
        
        self.points_map = {}
        self.tracks_graph = nx.Graph()
        lines_indexes: Dict[int, index.Index] = {}
        counter = 0

        # Load the tracks data (railway shapes and speeds)
        railway_speed_path = os.path.join(self.data_directory, 'vitesse-maximale-nominale-sur-ligne.json')
        if not os.path.exists(railway_speed_path):
            raise FileNotFoundError("Railway speed data missing at: %s" % os.path.abspath(railway_speed_path))
        with open(railway_speed_path, 'r') as f:
            tracks = json.load(f)
        normalize_tracks(tracks)

        if self.verbose:
            start_time = time.time()
            logging.debug(f"Building tracks graph for {len(tracks)} tracks")
            j = 0
        
        for track in tracks:
            if self.verbose:
                j += 1
                if (j % 100) == 0:
                    logging.debug(f"{j} tracks done ({time.time() - start_time:.2f}s)")

            coords = track['geo_shape']['geometry']['coordinates']

            # km/h to m/s
            vmax = track['v_max']
            if not vmax:
                vmax = 160 # Average speed ??
            vmax = float(vmax) / 3.6
            for line_idx, line in enumerate(coords):
                last_pos = None
                for i, (lat, lon) in enumerate(line):
                    point = Point(lat, lon) 
                    current_pos = f"{lat}_{lon}"

                    closest = self.find_closest_rtree_points(lat, lon, convert_to_positions=True)
                    for closest_point in closest:
                        closest_point_str = f"{closest_point[0]}_{closest_point[1]}"
                        if current_pos == closest_point_str:
                            continue
                        dist = geodesic((lat, lon), closest_point).meters
                        # Add an edge between the two points if they are close enough
                        # This is required because sometimes, points are spaced by a fucking 10th of millimeter, causing a stop in the tracks graph
                        if dist < 10:
                            self.tracks_graph.add_edge(current_pos, closest_point_str, weight=0)
                    
                    # Build lines indexes to create edge between the different lines of a same station
                    line_code = int(track["code_ligne"])
                    if line_code not in lines_indexes:
                        lines_indexes[line_code] = index.Index()
                    lines_indexes[line_code].insert(counter, (point.x, point.y, point.x, point.y))
                    self.rtree_idx.insert(counter, (point.x, point.y, point.x, point.y))
                    self.points_map[counter] = (track, line_idx, i)  # Map the index to the track and point index

                    counter += 1
                    if last_pos is not None:
                        self.tracks_graph.add_edge(last_pos, current_pos, weight=geodesic(last_pos.split('_'), (lat, lon)).meters / vmax)
                    last_pos = current_pos

        # Ensure the stations are linked to all their associated lines
        stations_list_path = os.path.join(self.data_directory, 'liste-des-gares.geojson')
        if not os.path.exists(stations_list_path):
            raise FileNotFoundError("Stations list data missing at: %s" % os.path.abspath(stations_list_path))
        with open(stations_list_path, 'r') as f:
            stations = json.load(f)
        if self.verbose:
            start_time = time.time()
            logging.debug(f"Linking {len(stations['features'])} stations to their associated lines")
            i = 0
        for station in stations['features']:
            if self.verbose:
                i += 1
                if (i % 100) == 0:
                    logging.debug(f"{i} stations done ({time.time() - start_time:.2f}s)")
            station_line = int(station["properties"]["code_ligne"])
            if station_line in lines_indexes:
                # Get (lat, lon) of the station
                station_point = station['geometry']['coordinates']
                station_point.reverse()
                station_idx_point = self.find_closest_rtree_points(*station_point, k=1, convert_to_positions=True)[0]
                station_idx_point_str = f"{station_idx_point[0]}_{station_idx_point[1]}"
                # Get the closest point on the line
                closest_idx_line_point = self.find_closest_rtree_points(*station_point, k=1, convert_to_positions=True, tree=lines_indexes[station_line])[0]
                closest_idx_line_point_str = f"{closest_idx_line_point[0]}_{closest_idx_line_point[1]}"

                # If the station is already the closest point on the line, skip it (no need to link it to itself)
                if station_idx_point_str == closest_idx_line_point_str:
                    continue
                # Add an edge between the station and the closest point on the line
                self.tracks_graph.add_edge(station_idx_point_str, closest_idx_line_point_str, weight=0)

        # Save the processed data
        pickle.dump(self.tracks_graph, open(graph_filepath, 'wb'))
        pickle.dump(self.points_map, open(point_map_filepath, 'wb'))
        self.rtree_idx.close()
        self.rtree_idx = index.Index(rtree_filepath)

    def find_closest_rtree_points(self, lat, lon, k=1, convert_to_positions: bool = False, tree: Optional[index.Index] = None):
        """
        Use the R-tree to find the closest point to a given lat/lon.
        """
        if (tree is None) and (self.rtree_idx is None):
            raise Exception("The R-tree index is not loaded")
        if tree is None:
            tree = self.rtree_idx
        
        point = Point(lat, lon) 
        nearest = list(tree.nearest((point.x, point.y, point.x, point.y), num_results=k))
        
        # Retrieve the corresponding track and point index from point_map
        closest_points = []
        for idx in nearest:
            track, line_idx, point_idx = self.points_map[idx]
            if convert_to_positions:
                closest_points.append(track['geo_shape']['geometry']['coordinates'][line_idx][point_idx])
            else:
                closest_points.append((track, line_idx, point_idx))
        
        return closest_points


    def find_railway_tracks_path(self, prev_lat, prev_lon, lat, lon, fmap = None, marker_cluster = None):
        """
        Find the shape of the railway tracks between two points of the railway network.
        """
        if not self.rtree_idx:
            raise Exception("The R-tree index is not loaded")
        closest_prev_points = self.find_closest_rtree_points(prev_lat, prev_lon)
        closest_next_points = self.find_closest_rtree_points(lat, lon)

        for prev_track, prev_line_idx, prev_idx in closest_prev_points:
            for next_track, next_line_idx, next_idx in closest_next_points:
                prev_coords = prev_track['geo_shape']['geometry']['coordinates'][prev_line_idx][prev_idx]
                next_coords = next_track['geo_shape']['geometry']['coordinates'][next_line_idx][next_idx]
                try:
                    path_strs = nx.shortest_path(self.tracks_graph, source=f"{prev_coords[0]}_{prev_coords[1]}", target=f"{next_coords[0]}_{next_coords[1]}", weight='weight')
                except nx.NetworkXNoPath:
                    folium.PolyLine(locations=[prev_coords, next_coords], color="red", weight=2.5, opacity=0.8).add_to(fmap)
                    folium.Marker(
                        location=prev_coords,
                        popup=f"{prev_coords[0]}_{prev_coords[1]}",
                        icon=folium.Icon(color='blue', icon='train', prefix='fa')
                    ).add_to(marker_cluster)
                    folium.Marker(
                        location=next_coords,
                        popup=f"{next_coords[0]}_{next_coords[1]}",
                        icon=folium.Icon(color='blue', icon='train', prefix='fa')
                    ).add_to(marker_cluster)
                    continue
                        
                path = []
                last_pos = None
                for coord in path_strs:
                    pos = coord.split('_')
                    if last_pos is not None:
                        path.append([last_pos, pos])
                    last_pos = pos
                return path

        return None
    
    def get_area(self, area_name: str, is_dest: bool = False):
        area_regex_name = re.sub(r'[ -]', r'[ -]', area_name)
        area_regex_name = fr"^{area_regex_name}(?:\s|$)"
        possible_areas = self.stops_df[self.stops_df['stop_name'].str.match(area_regex_name, case=False) & (self.stops_df['location_type'] == 1)]
        if possible_areas.empty:
            raise Exception(f"Area {area_name} not found")
        if len(possible_areas) == 1:
            return possible_areas.iloc[0]

        root_area = 'area_%s' % area_name
        self.graph.add_node(root_area, name=root_area, type=NodeType.AREA)
        for _, stop in possible_areas.iterrows():
            if is_dest:
                self.graph.add_edge(stop['stop_id'], root_area, weight=0, type=EdgeType.TIMEPOINT)
            else:
                self.graph.add_edge(root_area, stop['stop_id'], weight=0, type=EdgeType.TIMEPOINT)
        return self.graph.nodes[root_area]
    
    def get_area_time_point(self, area, time: str):
        # This is a node because multiple areas are matching with the requested name
        if isinstance(area, dict):
            stop_datetimes: List[datetime.time] = []
            for stop_area in self.graph._succ[area['name']].keys():
                stop_datetimes.extend([seconds_to_time(time_to_seconds(tp.split('_')[1], True)) for tp in self.graph._pred[stop_area] if not tp.startswith('area_')])
        else:
            stop_datetimes: List[datetime.time] = [seconds_to_time(time_to_seconds(tp.split('_')[1], True)) for tp in self.graph._pred[area['stop_id']] if not tp.startswith('area_')]
        stop_datetimes.sort()
        tp = find_closest_timepoint(time, stop_datetimes)
        tp_str = tp.strftime('%H:%M:%S')
        if isinstance(area, dict):
            for stop_area in self.graph._succ[area['name']].keys():
                if f"{stop_area}_{tp_str}" in self.graph:
                    return f"{stop_area}_{tp_str}"
        return f"{area['stop_id']}_{tp_str}"

    def compute_path(self, source: str, destination: str, display: bool = False):
        weight, shortest_path = nx.bidirectional_dijkstra(self.graph, source=source, target=destination)
        if display:
            self.display_path(shortest_path, weight)
        return weight, shortest_path

    def compute_path_from_names(self, source_name: str, destination_name: str, display: bool = False, source_time: str = None):
        source_area = self.get_area(source_name, is_dest=False)
        if source_time is None:
            source_time = datetime.datetime.now().strftime('%H:%M:%S')
        source_node_name = self.get_area_time_point(source_area, source_time)
        dest = self.get_area(destination_name, is_dest=True)
        if isinstance(dest, dict):
            dest_node_name = dest['name']
        else:
            dest_node_name = dest['stop_id']
        return self.compute_path(source_node_name, dest_node_name, display)
    
    def get_node_display_data(self, node_id: str) -> NodeDisplayData:
        if not self.graph.has_node(node_id):
            return None
        node = self.graph.nodes[node_id]
        if node['type'] == NodeType.AREA:
            return NodeDisplayData(name=node['name'], pos=node['pos'])
        if node['type'] == NodeType.AREA_TIME_POINT:
            return self.get_node_display_data(node_id.rsplit('_', 1)[0])
        if node['type'] == NodeType.TRIP_STOP:
            return self.get_node_display_data(self.graph.nodes[node_id]['area'])
        return None

    def display_path(self, path: List[str], weight, display_instructions: bool = True, html_ouput_file: str = None):
        start_node_data = self.get_node_display_data(path[0])

        if display_instructions:
            first_node = self.graph.nodes[path[0]]
            if first_node['type'] == NodeType.TRIP_STOP:
                tm = self.graph.nodes[path[0]]['time']
            else:
                tm = path[0].rsplit('_', 1)[-1]
            instructions = ["Départ: %s %s" % (start_node_data.name, tm)]
        if html_ouput_file:
            map_center = [start_node_data.pos.lat, start_node_data.pos.lon]
            fmap = folium.Map(location=map_center, zoom_start=6)

            # Create a MarkerCluster to group nearby markers
            marker_cluster = MarkerCluster().add_to(fmap)

        current_trip = None
        last_node_data: NodeDisplayData = None
        for i in range(len(path) - 1):
            node_id = path[i]
            node_data = self.get_node_display_data(node_id)
            edge = self.graph.get_edge_data(node_id, path[i+1])

            # Prepare the marker for the current stop
            if html_ouput_file:
                marker_popup = f"Stop: {node_data.name}"
                icon = folium.Icon(color='blue', icon='train', prefix='fa')
            
            edge_type: EdgeType = edge['type']
            if edge_type == EdgeType.TRIP and (current_trip is None or current_trip != edge['trip']):
                current_trip = edge['trip']
                route_id = self.trips_df[self.trips_df['trip_id'] == current_trip]['route_id'].iloc[0]
                route = self.routes_df[self.routes_df['route_id'] == route_id]['route_long_name'].iloc[0]

                # Instructions
                if display_instructions:
                    start_time = self.graph.nodes[path[i]]['time']
                    instructions.append(f"- {start_time}: Prendre la ligne '{route}'")
                # Map
                if html_ouput_file:
                    icon = folium.Icon(color='green', icon='info-sign')
                    marker_popup += f"\n(Change: {route})"
            elif edge_type == EdgeType.TIMEPOINT and current_trip is not None:
                current_trip = None
                # Instructions
                if display_instructions:
                    instructions.append('- %s: Descendre à %s' % (path[i + 1].rsplit('_', 1)[-1], node_data.name))

            # Add marker for the current stop on the map
            if html_ouput_file and (last_node_data is None or last_node_data.pos != node_data.pos):
                lat, lon = node_data.pos.lat, node_data.pos.lon
                folium.Marker(
                    location=[lat, lon],
                    popup=marker_popup,
                    icon=icon
                ).add_to(marker_cluster)

                # Draw a line between the current stop and the next stop
                if last_node_data is not None:
                    # Find the corresponding LineString from the shapes (if they are loaded)
                    # Fallback on a simple line if the shapes are not available or if the matching LineString is not found
                    prev_lat, prev_lon = last_node_data.pos.lat, last_node_data.pos.lon
                    if self.rtree_idx:
                        matching_track = self.find_railway_tracks_path(prev_lat, prev_lon, lat, lon, fmap, marker_cluster)
                    else:
                        matching_track = None

                    # Draw the LineString (train tracks)
                    if matching_track:
                        folium.PolyLine(locations=matching_track, color="blue", weight=2.5, opacity=0.8).add_to(fmap)
                    else:
                        folium.PolyLine(locations=[[prev_lat, prev_lon], [lat, lon]], color="blue", weight=2.5, opacity=0.8).add_to(fmap)

            last_node_data = node_data

            if display_instructions and i == len(path) - 2:
                instructions.append(f"Arrivée: {node_data.name} {path[i].rsplit('_', 1)[-1]} (temps: {datetime.timedelta(seconds=weight)})")
        
        if display_instructions:
            print("\n".join(instructions))
        if html_ouput_file:
            fmap.save(html_ouput_file)
            if self.verbose:
                logging.info(f"Interractive map saved to {html_ouput_file}")
