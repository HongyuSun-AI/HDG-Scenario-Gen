import xml.etree.ElementTree as ET
import numpy as np
from pyproj import Proj


class MergeRoad:
    def __init__(self, path='data_process/maps/merge.osm', xmin=1000., xmax=1140.):
        self.xmin, self.xmax = xmin, xmax
        root = ET.parse(path).getroot()
        proj = Proj(proj='utm', ellps='WGS84', zone=31, datum='WGS84')
        origin = np.array(proj(0, 0))
        nodes = {n.attrib['id']: np.array(proj(float(n.attrib['lon']), float(n.attrib['lat']))) - origin
                 for n in root.findall('node')}
        ways = {w.attrib['id']: np.array([nodes[n.attrib['ref']] for n in w.findall('nd')])
                for w in root.findall('way')}
        self.lines = []
        for rid in (1000, 2000, 3000, 4000):
            rel = root.find(f"relation[@id='{rid}']")
            if rel is None:
                raise ValueError(f'Missing road boundary {rid}')
            pts = np.concatenate([ways[m.attrib['ref']] for m in rel.findall('member') if m.attrib['type'] == 'way'])
            pts = pts[np.argsort(pts[:, 0], kind='stable')]
            _, ix = np.unique(pts[:, 0], return_index=True)
            self.lines.append(pts[ix])
        self.xgrid = np.linspace(xmin, xmax, 1401)
        self.centers = (self.bounds(self.xgrid)[:-1] + self.bounds(self.xgrid)[1:]) / 2
        self.arc = np.c_[np.zeros(3), np.cumsum(np.sqrt(np.diff(self.xgrid)[None] ** 2 + np.diff(self.centers, axis=1) ** 2), axis=1)]

    def bounds(self, x):
        return np.array([np.interp(x, p[:, 0], p[:, 1]) for p in self.lines])

    def lane(self, x, y):
        b = self.bounds(x)
        result = np.full(np.shape(x), -1, dtype=int)
        for lane in range(2, -1, -1):
            result = np.where((x >= self.xmin) & (x <= self.xmax) &
                              (y <= b[lane]) & (y >= b[lane + 1]) &
                              (b[lane] - b[lane + 1] > .3), lane, result)
        return result

    def path_s(self, lane, x):
        return np.interp(x, self.xgrid, self.arc[lane])

    def gap(self, lane, xi, xj, length_i=4.5, length_j=4.5):
        return abs(self.path_s(lane, xj) - self.path_s(lane, xi)) - (length_i + length_j) / 2

    def center(self, lane, x):
        b = self.bounds(x)
        return (b[lane] + b[lane + 1]) / 2

    def layout(self, states, valid, cells=28):
        # (3, cells, 3)
        result = np.full((3, cells, 3), -1., dtype=np.float32)
        collisions = 0
        for s in states[valid]:
            lane = int(self.lane(s[0], s[1]))
            if lane < 0:
                continue
            cell = min(cells - 1, int((s[0] - self.xmin) / (self.xmax - self.xmin) * cells))
            if result[lane, cell, 0] != -1:
                collisions += 1
                continue
            bounds = self.bounds(s[0])
            result[lane, cell] = [(s[0] - self.xmin) / (self.xmax - self.xmin) * cells - cell,
                                  (s[1] - bounds[lane + 1]) / (bounds[lane] - bounds[lane + 1]),
                                  (s[2] + 3.14) / 6.28]
        return result, collisions

    def decode_layout(self, layout):
        states, malformed = [], 0
        cells = layout.shape[1]
        for lane, cell in np.ndindex(layout.shape[:2]):
            value = layout[lane, cell]
            if (value < -.9).all():
                continue
            if not np.isfinite(value).all() or (value < -.1).any():
                malformed += 1
                continue
            xlocal, ylocal, heading = np.clip(value, 0, 1)
            x = self.xmin + (cell + xlocal) / cells * (self.xmax - self.xmin)
            bounds = self.bounds(x)
            if bounds[lane] - bounds[lane + 1] <= .3:
                malformed += 1
                continue
            states.append([x, bounds[lane + 1] + ylocal * (bounds[lane] - bounds[lane + 1]), heading * 6.28 - 3.14])
        return np.asarray(states).reshape(-1, 3), malformed
