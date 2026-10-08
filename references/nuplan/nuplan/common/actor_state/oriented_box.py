from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property, lru_cache
from typing import List, Optional

import numpy as np
from shapely.geometry import Polygon

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.geometry.transform import translate_longitudinally_and_laterally


class OrientedBoxPointType(IntEnum):


    FRONT_BUMPER = (1,)
    REAR_BUMPER = (2,)
    FRONT_LEFT = (3,)
    FRONT_RIGHT = (4,)
    REAR_LEFT = (5,)
    REAR_RIGHT = (6,)
    CENTER = (7,)
    LEFT = (8,)
    RIGHT = 9


@dataclass(frozen=True)
class Dimension:


    length: float
    width: float
    height: float


class OrientedBox:


    def __init__(self, center: StateSE2, length: float, width: float, height: float):

        self._center = center
        self._length = length
        self._width = width
        self._height = height

    @property
    def dimensions(self) -> Dimension:

        return Dimension(length=self.length, width=self.width, height=self.height)

    @lru_cache()
    def corner(self, point: OrientedBoxPointType) -> Point2D:

        if point == OrientedBoxPointType.FRONT_LEFT:
            return translate_longitudinally_and_laterally(self.center, self.half_length, self.half_width).point
        elif point == OrientedBoxPointType.FRONT_RIGHT:
            return translate_longitudinally_and_laterally(self.center, self.half_length, -self.half_width).point
        elif point == OrientedBoxPointType.REAR_LEFT:
            return translate_longitudinally_and_laterally(self.center, -self.half_length, self.half_width).point
        elif point == OrientedBoxPointType.REAR_RIGHT:
            return translate_longitudinally_and_laterally(self.center, -self.half_length, -self.half_width).point
        elif point == OrientedBoxPointType.CENTER:
            return self._center.point
        elif point == OrientedBoxPointType.FRONT_BUMPER:
            return translate_longitudinally_and_laterally(self.center, self.half_length, 0.0).point
        elif point == OrientedBoxPointType.REAR_BUMPER:
            return translate_longitudinally_and_laterally(self.center, -self.half_length, 0.0).point
        elif point == OrientedBoxPointType.LEFT:
            return translate_longitudinally_and_laterally(self.center, 0, self.half_width).point
        elif point == OrientedBoxPointType.RIGHT:
            return translate_longitudinally_and_laterally(self.center, 0, -self.half_width).point
        else:
            raise RuntimeError(f"Unknown point: {point}!")

    def all_corners(self) -> List[Point2D]:

        return [
            self.corner(OrientedBoxPointType.FRONT_LEFT),
            self.corner(OrientedBoxPointType.REAR_LEFT),
            self.corner(OrientedBoxPointType.REAR_RIGHT),
            self.corner(OrientedBoxPointType.FRONT_RIGHT),
        ]

    @property
    def width(self) -> float:

        return self._width

    @property
    def half_width(self) -> float:

        return self._width / 2.0

    @property
    def length(self) -> float:

        return self._length

    @property
    def half_length(self) -> float:

        return self._length / 2.0

    @property
    def height(self) -> float:

        return self._height

    @property
    def half_height(self) -> float:

        return self._height / 2.0

    @property
    def center(self) -> StateSE2:

        return self._center

    @cached_property
    def geometry(self) -> Polygon:

        corners = [tuple(corner) for corner in self.all_corners()]
        return Polygon(corners)

    def __hash__(self) -> int:

        return hash((self.center, self.width, self.height, self.length))

    def __eq__(self, other: object) -> bool:

        if not isinstance(other, OrientedBox):

            return NotImplemented
        return (
            math.isclose(self.width, other.width)
            and math.isclose(self.height, other.height)
            and math.isclose(self.length, other.length)
            and self.center == other.center
        )

    @classmethod
    def from_new_pose(cls, box: OrientedBox, pose: StateSE2) -> OrientedBox:

        return cls(pose, box.length, box.width, box.height)


def collision_by_radius_check(box1: OrientedBox, box2: OrientedBox, radius_threshold: Optional[float]) -> bool:

    if not radius_threshold:
        w1, l1 = box1.width, box1.length
        w2, l2 = box2.width, box2.length
        radius_threshold = (np.hypot(w1, l1) + np.hypot(w2, l2)) / 2.0

    distance_between_centers = box1.center.distance_to(box2.center)

    return bool(distance_between_centers < radius_threshold)


def in_collision(box1: OrientedBox, box2: OrientedBox, radius_threshold: Optional[float] = None) -> bool:

    return (
        bool(box1.geometry.intersects(box2.geometry))
        if collision_by_radius_check(box1, box2, radius_threshold)
        else False
    )
