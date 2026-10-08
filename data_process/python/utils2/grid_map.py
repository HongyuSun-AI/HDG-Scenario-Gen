import os
import pandas as pd
import numpy as np

import xml.etree.ElementTree as xml
import pyproj
import math

class Grid:
    def __init__(self, min_x, max_x, x_step):
        self.min_x = min_x
        self.max_x = max_x
        self.x_step = x_step

    def get_grid_number_x(self, x_value):


        if x_value <= self.min_x or x_value > self.max_x:
            raise ValueError(f"Input={x_value}; range=({self.min_x},{self.max_x})")


        grid_number = (x_value - self.min_x) // 5

        return int(grid_number)

    def get_score_in_grid(self, x_value):

        if x_value < self.min_x or x_value > self.max_x:
            raise ValueError(f"Input={x_value}; range=({self.min_x},{self.max_x})")

        grid_number = self.get_grid_number_x(x_value)


        grid_start = self.min_x + grid_number * self.x_step
        grid_end = grid_start + self.x_step


        score = (x_value - grid_start) / (grid_end - grid_start)

        return score

    def get_x_from_grid(self, grid_number, score):


        if not (0 <= score <= 1):
            raise ValueError('Invalid score')


        grid_start = self.min_x + grid_number * self.x_step
        grid_end = grid_start + self.x_step


        x_value = grid_start + score * (grid_end - grid_start)

        return x_value

    def get_lane_id(self, y1, y2, y3, y4, y_value):


        if y1 >= y_value >= y2:
            return 0
        elif y2 > y_value >= y3:
            return 1
        elif y3 > y_value >= y4:
            return 2
        else:
            raise ValueError(f"y_value {y_value} 不在任何车道范围内。")
    def convert_and_normalize(self, array):


        B, C, W = array.shape
        normalized_array = np.zeros_like(array)

        for b in range(B):
            for c in range(C):
                for w in range(W):
                    score = array[b, c, w]
                    if score == -1:
                        normalized_array[b, c, w] = -1
                    else:

                        if score > 1:
                            score = 1


                        grid_number = w
                        x_value = self.get_x_from_grid(grid_number, score)

                        normalized_value = (x_value - self.min_x) / (self.max_x - self.min_x)

                        normalized_array[b, c, w] = normalized_value

        return normalized_array


class LL2XYProjector:
    def __init__(self, lat_origin, lon_origin):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.zone = math.floor((lon_origin + 180.) / 6) + 1
        self.p = pyproj.Proj(proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84')
        [self.x_origin, self.y_origin] = self.p(lon_origin, lat_origin)

    def latlon2xy(self, lat, lon):
        [x, y] = self.p(lon, lat)
        return [x - self.x_origin, y - self.y_origin]

def get_way_ids_by_relation(e, relation_id):
    way_ids = []
    for relation in e.findall('relation'):
        if int(relation.get('id')) == relation_id:
            for member in relation.findall('member'):
                if member.get('type') == 'way':
                    way_ids.append(int(member.get('ref')))
    return way_ids

def get_coordinates_by_way(e, way_id, projector):
    coordinates = []
    way = e.find(f'way[@id="{way_id}"]')
    if way is not None:
        for nd in way.findall('nd'):
            ref_id = int(nd.get('ref'))
            node = e.find(f'node[@id="{ref_id}"]')
            if node is not None:
                lat = float(node.get('lat'))
                lon = float(node.get('lon'))
                x, y = projector.latlon2xy(lat, lon)
                coordinates.append((x, y))
    return coordinates

def find_y_for_x(x_input, e, relation_id, projector):
    way_ids = get_way_ids_by_relation(e, relation_id)
    for way_id in way_ids:
        coordinates = get_coordinates_by_way(e, way_id, projector)
        for i in range(len(coordinates) - 1):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[i + 1]
            if x1 <= x_input <= x2 or x2 <= x_input <= x1:
                if x1 == x2:
                    return y1

                y = y1 + (x_input - x1) * (y2 - y1) / (x2 - x1)
                return y
    return None

def get_y(osm_file, x_input, relation_id, lat_origin=0, lon_origin=0):
    projector = LL2XYProjector(lat_origin, lon_origin)
    e = xml.parse(osm_file).getroot()
    y_output = find_y_for_x(x_input, e, relation_id, projector)
    if y_output is not None:
        return y_output
    else:
        print("未找到对应的 y 坐标")


if __name__ == "__main__":
    osm_file = '../map.osm'
    x_input = 1000
    relation_id = 1000
    lat_origin = 0
    lon_origin = 0

    y_output = get_y(osm_file, x_input, relation_id, lat_origin, lon_origin)

    print(f"y 坐标: {y_output}")
