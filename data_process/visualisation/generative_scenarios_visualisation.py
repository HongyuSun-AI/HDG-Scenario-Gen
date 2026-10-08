#!/usr/bin/env python

try:
    import lanelet2

    use_lanelet2_lib = True
except ImportError:
    import warnings

    string = "lanelet2 import failed"
    warnings.warn(string)
    print("Using visualization without lanelet2.")
    use_lanelet2_lib = False
    from utils import map_vis_without_lanelet

import argparse
import os
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import pickle

from utils import dataset_reader
from utils import dataset_types
from utils import map_vis_lanelet2
from utils import tracks_vis
from utils import dict_utils

def update_plot():
    global fig, timestamp, title_text, track_dictionary, patches_dict, text_dict, axes, pedestrian_dictionary

    assert (timestamp <= timestamp_max), "timestamp=%i" % timestamp
    assert (timestamp >= timestamp_min), "timestamp=%i" % timestamp
    assert (timestamp % dataset_types.DELTA_TIMESTAMP_MS == 0), "timestamp=%i" % timestamp
    percentage = (float(timestamp) / timestamp_max) * 100
    title_text.set_text('\nts={}/{} ({:.2f}%)'.format(timestamp, timestamp_max, percentage))
    tracks_vis.update_objects_plot(timestamp, patches_dict, text_dict, axes,
                                   track_dict=track_dictionary, pedest_dict=pedestrian_dictionary)
    fig.canvas.draw()


def start_playback():
    global timestamp, timestamp_min, timestamp_max, playback_stopped
    playback_stopped = False
    plt.ion()
    while timestamp < timestamp_max and not playback_stopped:
        timestamp += dataset_types.DELTA_TIMESTAMP_MS
        start_time = time.time()
        update_plot()
        end_time = time.time()
        diff_time = end_time - start_time
        plt.pause(max(0.001, dataset_types.DELTA_TIMESTAMP_MS / 1000. - diff_time))
    plt.ioff()


class FrameControlButton(object):
    def __init__(self, position, label):
        self.ax = plt.axes(position)
        self.label = label
        self.button = Button(self.ax, label)
        self.button.on_clicked(self.on_click)

    def on_click(self, event):
        global timestamp, timestamp_min, timestamp_max, playback_stopped

        if self.label == "play":
            if not playback_stopped:
                return
            else:
                start_playback()
                return
        playback_stopped = True
        if self.label == "<<":
            timestamp -= 10 * dataset_types.DELTA_TIMESTAMP_MS
        elif self.label == "<":
            timestamp -= dataset_types.DELTA_TIMESTAMP_MS
        elif self.label == ">":
            timestamp += dataset_types.DELTA_TIMESTAMP_MS
        elif self.label == ">>":
            timestamp += 10 * dataset_types.DELTA_TIMESTAMP_MS
        timestamp = min(timestamp, timestamp_max)
        timestamp = max(timestamp, timestamp_min)
        update_plot()


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_name", type=str, help='Scenario name', nargs="?")
    parser.add_argument("track_file_number", type=int, help='Track file number', default=0, nargs="?")
    parser.add_argument("load_mode", type=str, help='Vehicle, pedestrian, or both', default="both",
                        nargs="?")
    parser.add_argument("--start_timestamp", type=int, nargs="?")
    parser.add_argument("--lat_origin", type=float,
                        help='Projection reference latitude',
                        default=0.0, nargs="?")
    parser.add_argument("--lon_origin", type=float,
                        help='Projection reference longitude',
                        default=0.0, nargs="?")
    args = parser.parse_args()
    args.scenario_name = "DR_CHN_Merging_ZS"
    if args.scenario_name is None:
        raise IOError("Missing scenario")
    if args.load_mode != "vehicle" and args.load_mode != "pedestrian" and args.load_mode != "both":
        raise IOError("Invalid load command")


    error_string = ""


    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tracks_dir = os.path.join(root_dir, "recorded_trackfiles")
    maps_dir = os.path.join(root_dir, "maps")
    lanelet_map_ending = ".osm"
    lanelet_map_file = os.path.join(maps_dir, args.scenario_name + lanelet_map_ending)

    scenario_dir = os.path.join(tracks_dir, args.scenario_name)

    track_file_name = os.path.join(
        scenario_dir,
        "vehicle_tracks_" + str(args.track_file_number).zfill(3) + ".csv"
    )
    pedestrian_file_name = os.path.join(
        scenario_dir,
        "pedestrian_tracks_" + str(args.track_file_number).zfill(3) + ".csv"
    )

    if not os.path.isdir(tracks_dir):
        error_string += "Missing track file directory \"" + tracks_dir + "\"\n"
    if not os.path.isdir(maps_dir):
        error_string += "Missing map file directory \"" + tracks_dir + "\"\n"
    if not os.path.isdir(scenario_dir):
        error_string += "Missing scenario directory \"" + scenario_dir + "\"\n"
    if not os.path.isfile(lanelet_map_file):
        error_string += "Missing lanelet map file \"" + lanelet_map_file + "\"\n"
    if not os.path.isfile(track_file_name):
        error_string += "Missing track file \"" + track_file_name + "\"\n"
    if not os.path.isfile(pedestrian_file_name):
        flag_ped = 0
    else:
        flag_ped = 1


    fig, axes = plt.subplots(1, 1)
    fig.canvas.manager.set_window_title("Interaction Dataset Visualization")


    lat_origin = args.lat_origin
    lon_origin = args.lon_origin
    print("Loading map...")

    lanelet_map_file = os.path.join(maps_dir, "./map.osm")

    if use_lanelet2_lib:
        projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(lat_origin, lon_origin))
        laneletmap = lanelet2.io.load(lanelet_map_file, projector)
        map_vis_lanelet2.draw_lanelet_map(laneletmap, axes)
    else:
        map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, lat_origin, lon_origin)


    current_file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(current_file_path))
    target_dir = os.path.join(parent_dir, 'recorded_trackfiles', 'generated_scenarios')
    track_file_name = os.path.join(target_dir, f'{1}.csv')


    track_dictionary = None
    pedestrian_dictionary = None
    if args.load_mode == 'both':
        track_dictionary = dataset_reader.read_tracks(track_file_name)
        if flag_ped:
            pedestrian_dictionary = dataset_reader.read_pedestrian(pedestrian_file_name)

    elif args.load_mode == 'vehicle':
        track_dictionary = dataset_reader.read_tracks(track_file_name)
    elif args.load_mode == 'pedestrian':
        pedestrian_dictionary = dataset_reader.read_pedestrian(pedestrian_file_name)
    timestamp_min = 1e9
    timestamp_max = 0


    if track_dictionary is not None:
        for key, track in dict_utils.get_item_iterator(track_dictionary):
            timestamp_min = min(timestamp_min, track.time_stamp_ms_first)
            timestamp_max = max(timestamp_max, track.time_stamp_ms_last)
    else:
        for key, track in dict_utils.get_item_iterator(pedestrian_dictionary):
            timestamp_min = min(timestamp_min, track.time_stamp_ms_first)
            timestamp_max = max(timestamp_max, track.time_stamp_ms_last)

    if args.start_timestamp is None:
        args.start_timestamp = timestamp_min

    button_pp = FrameControlButton([0.2, 0.05, 0.05, 0.05], '<<')
    button_p = FrameControlButton([0.27, 0.05, 0.05, 0.05], '<')
    button_f = FrameControlButton([0.4, 0.05, 0.05, 0.05], '>')
    button_ff = FrameControlButton([0.47, 0.05, 0.05, 0.05], '>>')

    button_play = FrameControlButton([0.6, 0.05, 0.1, 0.05], 'play')
    button_pause = FrameControlButton([0.71, 0.05, 0.1, 0.05], 'pause')


    patches_dict = dict()
    text_dict = dict()


    print("Plotting...")
    timestamp = args.start_timestamp
    title_text = fig.suptitle("")
    playback_stopped = True

    update_plot()
    plt.show()
