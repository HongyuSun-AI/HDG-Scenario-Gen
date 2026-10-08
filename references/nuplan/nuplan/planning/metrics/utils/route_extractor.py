from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set

import numpy as np
import numpy.typing as npt
from shapely.geometry import Point

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import (
    GraphEdgeMapObject,
    Lane,
    LaneConnector,
    LaneGraphEdgeMapObject,
    PolylineMapObject,
)
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

logger = logging.getLogger(__name__)


@dataclass
class RouteBaselineRoadBlockPair:


    road_block: LaneGraphEdgeMapObject
    base_line: PolylineMapObject
    next: Optional[RouteBaselineRoadBlockPair] = None


@dataclass
class RouteRoadBlockLinkedList:


    head: Optional[RouteBaselineRoadBlockPair] = None


def get_current_route_objects(map_api: AbstractMap, pose: Point2D) -> List[GraphEdgeMapObject]:

    curr_lane = map_api.get_one_map_object(pose, SemanticMapLayer.LANE)
    if curr_lane is None:

        curr_lane_connectors = map_api.get_all_map_objects(pose, SemanticMapLayer.LANE_CONNECTOR)
        route_objects_with_pose = curr_lane_connectors
    else:
        route_objects_with_pose = [curr_lane]

    return route_objects_with_pose


def get_route_obj_with_candidates(
    pose: Point2D, candidate_route_objs: List[GraphEdgeMapObject]
) -> List[GraphEdgeMapObject]:

    if not len(candidate_route_objs):
        raise ValueError('Empty route candidates')


    route_objects_with_pose = [
        one_route_obj for one_route_obj in candidate_route_objs if one_route_obj.contains_point(pose)
    ]


    if not route_objects_with_pose and len(candidate_route_objs) == 1:
        route_objects_with_pose = [
            next_route_obj
            for next_route_obj in candidate_route_objs[0].outgoing_edges
            if next_route_obj.contains_point(pose)
        ]
    return route_objects_with_pose


def remove_extra_lane_connectors(route_objs: List[List[GraphEdgeMapObject]]) -> List[List[GraphEdgeMapObject]]:


    last_to_first_route_list = route_objs[::-1]
    enum = enumerate(last_to_first_route_list)
    for ind, curr_last_obj in enum:

        if ind == 0 or len(curr_last_obj) <= 1:
            continue

        if len(curr_last_obj) > len(last_to_first_route_list[ind - 1]):
            curr_route_obj_ids = [obj.id for obj in curr_last_obj]
            if all([(obj.id in curr_route_obj_ids) for obj in last_to_first_route_list[ind - 1]]):
                last_to_first_route_list[ind] = last_to_first_route_list[ind - 1]

        if len(curr_last_obj) <= 1:
            continue

        if last_to_first_route_list[ind - 1] and isinstance(last_to_first_route_list[ind - 1][0], Lane):
            next_lane_incoming_edge_ids = [obj.id for obj in last_to_first_route_list[ind - 1][0].incoming_edges]
            objs_to_keep = [obj for obj in curr_last_obj if obj.id in next_lane_incoming_edge_ids]
            if objs_to_keep:
                last_to_first_route_list[ind] = objs_to_keep

    return last_to_first_route_list[::-1]


def get_route(map_api: AbstractMap, poses: List[Point2D]) -> List[List[GraphEdgeMapObject]]:

    if not len(poses):
        raise ValueError('invalid poses passed to get_route()')

    route_objs: List[List[GraphEdgeMapObject]] = []


    curr_route_obj: List[GraphEdgeMapObject] = []

    for ind, pose in enumerate(poses):
        if curr_route_obj:


            curr_route_obj = get_route_obj_with_candidates(pose, curr_route_obj)


        if not curr_route_obj:
            curr_route_obj = get_current_route_objects(map_api, pose)


            if (
                ind > 1
                and route_objs[-1]
                and isinstance(route_objs[-1][0], LaneConnector)
                and (
                    (curr_route_obj and isinstance(curr_route_obj[0], LaneConnector))
                    or (not curr_route_obj and map_api.is_in_layer(pose, SemanticMapLayer.INTERSECTION))
                )
            ):
                previous_proximal_route_obj = [obj for obj in route_objs[-1] if obj.polygon.distance(Point(*pose)) < 5]

                if previous_proximal_route_obj:
                    curr_route_obj = previous_proximal_route_obj
        route_objs.append(curr_route_obj)


    improved_route_obj = remove_extra_lane_connectors(route_objs)
    return improved_route_obj


def get_route_simplified(route_list: List[List[LaneGraphEdgeMapObject]]) -> List[List[LaneGraphEdgeMapObject]]:


    try:

        ind = next(iteration for iteration, iter_route_list in enumerate(route_list) if iter_route_list)
    except StopIteration:
        logger.warning('Empty route list')
        return []

    route_simplified = [route_list[ind]]
    for route_object in route_list[ind + 1 :]:
        repeated_entries = [
            obj_id
            for obj_id in [prev_obj.id for prev_obj in route_simplified[-1]]
            if obj_id in [one_route_obj.id for one_route_obj in route_object]
        ]
        if route_object and not repeated_entries:
            route_simplified.append(route_object)
    return route_simplified


def get_route_baseline_roadblock_linkedlist(
    map_api: AbstractMap, expert_route: List[List[LaneGraphEdgeMapObject]]
) -> RouteRoadBlockLinkedList:


    route_baseline_roadblock_list = RouteRoadBlockLinkedList()
    prev_roadblock_id = None

    for route_object in expert_route:
        if route_object:

            roadblock_id = route_object[0].get_roadblock_id()
            if roadblock_id != prev_roadblock_id:
                prev_roadblock_id = roadblock_id
                if isinstance(route_object[0], Lane):
                    road_block = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
                else:
                    road_block = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)

                ref_baseline_path = route_object[0].baseline_path

                if route_baseline_roadblock_list.head is None:
                    prev_route_baseline_roadblock = RouteBaselineRoadBlockPair(
                        base_line=ref_baseline_path, road_block=road_block
                    )
                    route_baseline_roadblock_list.head = prev_route_baseline_roadblock
                else:
                    prev_route_baseline_roadblock.next = RouteBaselineRoadBlockPair(
                        base_line=ref_baseline_path, road_block=road_block
                    )
                    prev_route_baseline_roadblock = prev_route_baseline_roadblock.next

    return route_baseline_roadblock_list


def get_distance_of_closest_baseline_point_to_its_start(base_line: PolylineMapObject, pose: Point2D) -> float:

    return float(base_line.linestring.project(Point(*pose)))


@dataclass
class CornersGraphEdgeMapObject:


    front_left_map_objs: List[GraphEdgeMapObject]
    rear_left_map_objs: List[GraphEdgeMapObject]
    rear_right_map_objs: List[GraphEdgeMapObject]
    front_right_map_objs: List[GraphEdgeMapObject]

    def __iter__(self) -> Iterable[List[GraphEdgeMapObject]]:

        return iter(
            (self.front_left_map_objs, self.rear_left_map_objs, self.rear_right_map_objs, self.front_right_map_objs)
        )


def extract_corners_route(
    map_api: AbstractMap, ego_footprint_list: List[OrientedBox]
) -> List[CornersGraphEdgeMapObject]:

    if not len(ego_footprint_list):
        logger.warning('Invalid poses passed to extract_corners_route()')
        return []

    corners_route: List[CornersGraphEdgeMapObject] = []
    curr_candid_route_obj: List[GraphEdgeMapObject] = []

    for ind, ego_footprint in enumerate(ego_footprint_list):

        corners_route_objs = CornersGraphEdgeMapObject([], [], [], [])
        ego_corners = ego_footprint.all_corners()
        next_candid_route_obj: List[GraphEdgeMapObject] = []
        for ego_corner, corner_type in zip(ego_corners, corners_route_objs.__dict__.keys()):

            route_object = []
            if curr_candid_route_obj:
                route_object = get_route_obj_with_candidates(ego_corner, curr_candid_route_obj)


            if not route_object:
                route_object = get_current_route_objects(map_api, ego_corner)

                if ind == 1:
                    curr_candid_route_obj += [
                        obj for obj in route_object if obj.id not in [candid.id for candid in curr_candid_route_obj]
                    ]

            next_candid_route_obj += [
                obj for obj in route_object if obj.id not in [candid.id for candid in next_candid_route_obj]
            ]


            corners_route_objs.__setattr__(corner_type, route_object)

        corners_route.append(corners_route_objs)


        curr_candid_route_obj = next_candid_route_obj

    return corners_route


def get_outgoing_edges_obj_dict(corner_route_object: List[GraphEdgeMapObject]) -> dict[str, GraphEdgeMapObject]:

    return {obj_edge.id: obj_edge for obj in corner_route_object for obj_edge in obj.outgoing_edges}


def get_incoming_edges_obj_dict(corner_route_object: List[GraphEdgeMapObject]) -> dict[str, GraphEdgeMapObject]:

    return {obj_edge.id: obj_edge for obj in corner_route_object for obj_edge in obj.incoming_edges}


def get_common_route_object(
    corners_route_obj_ids: List[Set[str]], obj_id_dict: dict[str, GraphEdgeMapObject]
) -> Set[GraphEdgeMapObject]:

    return {obj_id_dict[id] for id in set.intersection(*corners_route_obj_ids)}


def get_connecting_route_object(
    corners_route_obj_list: List[List[GraphEdgeMapObject]],
    corners_route_obj_ids: List[Set[str]],
    obj_id_dict: dict[str, GraphEdgeMapObject],
) -> Set[GraphEdgeMapObject]:

    all_corners_connecting_obj_ids = set()

    front_left_route_obj, rear_left_route_obj, rear_right_route_obj, front_right_route_obj = corners_route_obj_list
    (
        front_left_route_obj_ids,
        rear_left_route_obj_ids,
        rear_right_route_obj_ids,
        front_right_route_obj_ids,
    ) = corners_route_obj_ids

    rear_right_route_obj_out_edge_dict = get_outgoing_edges_obj_dict(rear_right_route_obj)
    rear_left_route_obj_out_edge_dict = get_outgoing_edges_obj_dict(rear_left_route_obj)

    obj_id_dict = {**obj_id_dict, **rear_right_route_obj_out_edge_dict, **rear_left_route_obj_out_edge_dict}


    rear_right_obj_or_outgoing_edge = rear_right_route_obj_ids.union(set(rear_right_route_obj_out_edge_dict.keys()))
    rear_left_in_rear_right_obj_or_outgoing_edge = rear_left_route_obj_ids.intersection(rear_right_obj_or_outgoing_edge)

    rear_left_obj_or_outgoing_edge = rear_left_route_obj_ids.union(set(rear_left_route_obj_out_edge_dict.keys()))
    rear_right_in_rear_left_obj_or_outgoing_edge = rear_right_route_obj_ids.intersection(rear_left_obj_or_outgoing_edge)

    rear_corners_connecting_obj_ids = rear_left_in_rear_right_obj_or_outgoing_edge.union(
        rear_right_in_rear_left_obj_or_outgoing_edge
    )

    if len(rear_corners_connecting_obj_ids) > 0:


        front_left_route_obj_in_edge_dict = get_incoming_edges_obj_dict(front_left_route_obj)
        front_left_obj_or_incoming_edge = front_left_route_obj_ids.union(set(front_left_route_obj_in_edge_dict.keys()))
        front_left_rear_right_common_obj_ids = front_left_obj_or_incoming_edge.intersection(
            rear_right_obj_or_outgoing_edge
        )

        front_right_route_obj_in_edge_dict = get_incoming_edges_obj_dict(front_right_route_obj)
        front_right_obj_or_incoming_edge = front_right_route_obj_ids.union(
            set(front_right_route_obj_in_edge_dict.keys())
        )
        front_right_rear_left_common_obj_ids = front_right_obj_or_incoming_edge.intersection(
            rear_left_obj_or_outgoing_edge
        )

        all_corners_connecting_obj_ids = {
            obj_id_dict[id]
            for id in set.intersection(front_left_rear_right_common_obj_ids, front_right_rear_left_common_obj_ids)
        }

    return all_corners_connecting_obj_ids


def extract_common_or_connecting_route_objs(
    corners_route_obj: CornersGraphEdgeMapObject,
) -> Optional[Set[GraphEdgeMapObject]]:

    corners_route_obj_list = [*corners_route_obj.__iter__()]

    not_in_lane_or_laneconn = [
        True if len(corner_route_obj) == 0 else False for corner_route_obj in corners_route_obj_list
    ]


    if np.all(not_in_lane_or_laneconn):
        return set()


    if np.any(not_in_lane_or_laneconn):
        return None


    obj_id_dict = {obj.id: obj for corner_route_obj in corners_route_obj_list for obj in corner_route_obj}
    corners_route_obj_ids = [{obj.id for obj in corner_route_obj} for corner_route_obj in corners_route_obj_list]


    all_corners_common_obj = get_common_route_object(corners_route_obj_ids, obj_id_dict)
    if len(all_corners_common_obj) > 0:

        return all_corners_common_obj


    all_corners_connecting_obj = get_connecting_route_object(corners_route_obj_list, corners_route_obj_ids, obj_id_dict)
    if len(all_corners_connecting_obj) > 0:
        return all_corners_connecting_obj


    return None


def get_timestamps_in_common_or_connected_route_objs(
    common_or_connected_route_objs: List[Optional[Set[GraphEdgeMapObject]]], ego_timestamps: npt.NDArray[np.int32]
) -> List[int]:

    return [timestamp for route_obj, timestamp in zip(common_or_connected_route_objs, ego_timestamps) if route_obj]


def get_common_or_connected_route_objs_of_corners(
    corners_route: List[CornersGraphEdgeMapObject],
) -> List[Optional[Set[GraphEdgeMapObject]]]:

    history_common_or_connecting_route_objs: List[Optional[Set[GraphEdgeMapObject]]] = []

    prev_corners_route_obj = corners_route[0]

    corners_common_or_connecting_route_objs = extract_common_or_connecting_route_objs(prev_corners_route_obj)
    history_common_or_connecting_route_objs.append(corners_common_or_connecting_route_objs)

    for curr_corners_route_obj in corners_route[1:]:

        if curr_corners_route_obj != prev_corners_route_obj:

            corners_common_or_connecting_route_objs = extract_common_or_connecting_route_objs(curr_corners_route_obj)

        history_common_or_connecting_route_objs.append(corners_common_or_connecting_route_objs)

        prev_corners_route_obj = curr_corners_route_obj

    return history_common_or_connecting_route_objs
