from dataclasses import dataclass, asdict
from enum import IntEnum
import numpy as np
from .metrics import corners


class CollisionType(IntEnum):
    STOPPED_EGO_COLLISION = 0
    STOPPED_TRACK_COLLISION = 1
    ACTIVE_FRONT_COLLISION = 2
    ACTIVE_REAR_COLLISION = 3
    ACTIVE_LATERAL_COLLISION = 4


@dataclass(frozen=True)
class VehicleGeometry:
    length: float = 4.5
    width: float = 1.8
    rear_axle_to_center: float = 1.35


def box_intersects(a, b, ga, gb):
    ca, cb = corners(a, ga.length, ga.width), corners(b, gb.length, gb.width)
    for polygon in (ca, cb):
        for edge in (polygon[1] - polygon[0], polygon[2] - polygon[1]):
            axis = np.array([-edge[1], edge[0]])
            pa, pb = ca @ axis, cb @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def _front_intersects(ego, other, ego_geometry, other_geometry):

    front = corners(ego, ego_geometry.length, ego_geometry.width)[:2] - other[:2]
    c, s = np.cos(other[2]), np.sin(other[2])
    local = front @ np.array([[c, -s], [s, c]])
    delta = local[1] - local[0]
    lo, hi = 0., 1.
    for axis, half in enumerate((other_geometry.length / 2, other_geometry.width / 2)):
        if abs(delta[axis]) < 1e-12:
            if abs(local[0, axis]) > half:
                return False
        else:
            bounds = sorted(((-half - local[0, axis]) / delta[axis],
                             (half - local[0, axis]) / delta[axis]))
            lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
            if lo > hi:
                return False
    return True


def classify_collision(ego, other, ego_speed, other_speed,
                       ego_geometry=VehicleGeometry(), other_geometry=VehicleGeometry()):

    if ego_speed <= .05:
        return CollisionType.STOPPED_EGO_COLLISION
    if other_speed <= .05:
        return CollisionType.STOPPED_TRACK_COLLISION
    forward = np.array([np.cos(ego[2]), np.sin(ego[2])])
    rear_axle = ego[:2] - ego_geometry.rear_axle_to_center * forward
    relative = other[:2] - rear_axle
    norm = np.linalg.norm(relative)

    if norm > 0 and np.arccos(np.clip(relative @ forward / norm, -1., 1.)) > np.deg2rad(150):
        return CollisionType.ACTIVE_REAR_COLLISION
    if _front_intersects(ego, other, ego_geometry, other_geometry):
        return CollisionType.ACTIVE_FRONT_COLLISION
    return CollisionType.ACTIVE_LATERAL_COLLISION


def in_common_lane(state, road, geometry=VehicleGeometry()):

    points = corners(state, geometry.length, geometry.width)
    xmin, xmax = road.xmin, road.xmax
    if hasattr(road, 'lines'):
        xmin = max(line[:, 0].min() for line in road.lines)
        xmax = min(line[:, 0].max() for line in road.lines)
    if ((points[:, 0] < xmin) | (points[:, 0] > xmax)).any():
        return None
    bounds = road.bounds(points[:, 0])
    for lane in range(3):
        if ((points[:, 1] <= bounds[lane] + 1e-8) &
            (points[:, 1] >= bounds[lane + 1] - 1e-8) &
            (bounds[lane] - bounds[lane + 1] > .3)).all():
            return True
    return False


def is_at_fault(kind, common_lane):
    if kind in (CollisionType.ACTIVE_FRONT_COLLISION, CollisionType.STOPPED_TRACK_COLLISION):
        return True
    if kind == CollisionType.ACTIVE_LATERAL_COLLISION:
        return None if common_lane is None else not common_lane
    return False


class CollisionRecorder:

    def __init__(self, ego, road, geometries=None):
        self.ego, self.road = ego, road
        self.geometries = geometries
        self.seen, self.events = set(), []

    def observe(self, states, valid, speeds, frame):
        geometry = self.geometries or [VehicleGeometry()] * len(states)
        if not valid[self.ego]:
            return
        for other in np.flatnonzero(valid):
            if other == self.ego or int(other) in self.seen:
                continue
            if not box_intersects(states[self.ego], states[other], geometry[self.ego], geometry[other]):
                continue
            self.seen.add(int(other))
            kind = classify_collision(states[self.ego], states[other], speeds[self.ego], speeds[other],
                                      geometry[self.ego], geometry[other])
            common = in_common_lane(states[self.ego], self.road, geometry[self.ego])
            self.events.append(dict(frame=int(frame), other=int(other), collision_type=kind.name,
                at_fault=is_at_fault(kind, common), in_common_lane=common,
                ego_speed=float(speeds[self.ego]), other_speed=float(speeds[other])))

    def result(self):
        responsibility = [event['at_fault'] for event in self.events]
        at_fault = True if True in responsibility else (None if None in responsibility else False)
        return dict(collision=bool(self.events), at_fault=at_fault, events=self.events,
                    rule='nuPlan attribution, OSM lanes',
                    geometry=[asdict(g) for g in self.geometries] if self.geometries else asdict(VehicleGeometry()),
                    geometry_assumed=self.geometries is None,
                    map_limitation='Outside-map lateral attribution unknown')
