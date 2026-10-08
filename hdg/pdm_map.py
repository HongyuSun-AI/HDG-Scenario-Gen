from .pdm_runtime import activate
activate()
from types import SimpleNamespace
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.maps_datatypes import SemanticMapLayer as Layer


class LocalLane:
    def __init__(self, index, polygon, xy, speed_limit):
        self.id, self.index, self.polygon = f'merge_lane_{index}', index, polygon
        self.speed_limit_mps = speed_limit
        self.incoming_edges, self.outgoing_edges = [], []
        heading = np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0]))
        self.baseline_path = SimpleNamespace(
            discrete_path=[StateSE2(float(x), float(y), float(h)) for (x,y),h in zip(xy,heading)],
            length=float(LineString(xy).length), linestring=LineString(xy))

    def get_roadblock_id(self):
        return 'merge_roadblock'

    def contains_point(self, point):
        return self.polygon.covers(Point(point.x, point.y))


class MergeMapAdapter:
    def __init__(self, road, speed_limit=15., extension=200., route_end_x=None):
        self.road = road
        x = np.linspace(road.xmin - extension, road.xmax + extension, 541)
        bounds = road.bounds(x)
        self.lanes = {}
        for lane in range(3):
            polygon = Polygon(np.r_[np.c_[x, bounds[lane]], np.c_[x[::-1], bounds[lane+1, ::-1]]]).buffer(0)
            y = (bounds[lane] + bounds[lane+1]) / 2
            if lane == 2:
                widths = bounds[2] - bounds[3]
                narrow = np.flatnonzero((x >= road.xmin) & (widths < 2.5))
                if len(narrow):
                    end = float(x[narrow[0]])
                    blend = np.clip((x - (end - 30.)) / 30., 0, 1)
                    blend = blend * blend * (3 - 2 * blend)
                    y = (1-blend) * y + blend * (bounds[1] + bounds[2]) / 2
            route_x, route_y = x, y
            if route_end_x is not None:
                route_x = np.r_[x[x < route_end_x], route_end_x]
                route_y = np.interp(route_x, x, y)
            self.lanes[f'merge_lane_{lane}'] = LocalLane(lane, polygon, np.c_[route_x,route_y], speed_limit)
        self.block = SimpleNamespace(id='merge_roadblock', interior_edges=list(self.lanes.values()),
            incoming_edges=[], outgoing_edges=[], polygon=unary_union([l.polygon for l in self.lanes.values()]))

    def get_map_object(self, object_id, layer):
        if layer == Layer.ROADBLOCK:
            return self.block if object_id == self.block.id else None
        if layer == Layer.LANE:
            return self.lanes.get(object_id)
        return None

    def get_proximal_map_objects(self, point, radius, layers):
        p = Point(point.x, point.y)
        choices = {Layer.ROADBLOCK:[self.block], Layer.LANE:list(self.lanes.values())}
        return {layer:[o for o in choices.get(layer, []) if o.polygon.distance(p) <= radius] for layer in layers}

    def get_distance_to_nearest_map_object(self, point, layer):
        objects = self.get_proximal_map_objects(point, float('inf'), [layer])[layer]
        return min(((o.id, o.polygon.distance(Point(point.x, point.y))) for o in objects),
                   key=lambda pair:pair[1], default=(None, float('inf')))

    def is_in_layer(self, point, layer):
        if layer in (Layer.ROADBLOCK, Layer.DRIVABLE_AREA):
            return self.block.polygon.covers(Point(point.x, point.y))
        if layer == Layer.LANE:
            return any(l.contains_point(point) for l in self.lanes.values())
        return False
