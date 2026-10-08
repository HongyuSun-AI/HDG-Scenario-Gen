from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from shapely.geometry import LineString

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import in_collision
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChangeStatistics
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.metrics.utils.collision_utils import (
    CollisionType,
    ego_delta_v_collision,
    get_fault_type_statistics,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.observation.idm.utils import is_agent_behind, is_track_stopped
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks


@dataclass
class CollisionData:


    collision_ego_delta_v: float
    collision_type: CollisionType
    tracked_object_type: TrackedObjectType


@dataclass
class Collisions:


    timestamp: int

    collisions_id_data: Dict[str, CollisionData]


def _get_collision_type(
    ego_state: EgoState, tracked_object: TrackedObject, stopped_speed_threshold: float = 5e-02
) -> CollisionType:

    is_ego_stopped = ego_state.dynamic_car_state.speed <= stopped_speed_threshold


    if is_ego_stopped:
        collision_type = CollisionType.STOPPED_EGO_COLLISION


    elif is_track_stopped(tracked_object):
        collision_type = CollisionType.STOPPED_TRACK_COLLISION


    elif is_agent_behind(ego_state.rear_axle, tracked_object.box.center):
        collision_type = CollisionType.ACTIVE_REAR_COLLISION


    elif LineString(
        [
            ego_state.car_footprint.oriented_box.geometry.exterior.coords[0],
            ego_state.car_footprint.oriented_box.geometry.exterior.coords[3],
        ]
    ).intersects(tracked_object.box.geometry):
        collision_type = CollisionType.ACTIVE_FRONT_COLLISION


    else:
        collision_type = CollisionType.ACTIVE_LATERAL_COLLISION

    return collision_type


def find_new_collisions(
    ego_state: EgoState, observation: DetectionsTracks, collided_track_ids: Set[str]
) -> Tuple[Set[str], Dict[str, CollisionData]]:

    collisions_id_data: Dict[str, CollisionData] = {}

    for tracked_object in observation.tracked_objects:

        if tracked_object.track_token not in collided_track_ids and in_collision(
            ego_state.car_footprint.oriented_box, tracked_object.box
        ):


            collided_track_ids.add(tracked_object.track_token)

            collision_delta_v = ego_delta_v_collision(ego_state, tracked_object)

            collision_type = _get_collision_type(ego_state, tracked_object)

            collisions_id_data[tracked_object.track_token] = CollisionData(
                collision_delta_v, collision_type, tracked_object.tracked_object_type
            )

    return collided_track_ids, collisions_id_data


def classify_at_fault_collisions(
    all_collisions: List[Collisions],
    timestamps_in_common_or_connected_route_objs: List[int],
) -> Tuple[List[int], Dict[TrackedObjectType, List[float]]]:

    at_fault_collisions: Dict[TrackedObjectType, List[float]] = defaultdict(list)
    timestamps_at_fault_collisions: List[int] = []
    for collision in all_collisions:
        timestamp = collision.timestamp
        ego_in_multiple_lanes_or_nondrivable_area = timestamp not in timestamps_in_common_or_connected_route_objs

        for _id, collision_data in collision.collisions_id_data.items():

            collisions_at_stopped_track_or_active_front = collision_data.collision_type in [
                CollisionType.ACTIVE_FRONT_COLLISION,
                CollisionType.STOPPED_TRACK_COLLISION,
            ]


            collision_at_lateral = collision_data.collision_type == CollisionType.ACTIVE_LATERAL_COLLISION

            if collisions_at_stopped_track_or_active_front or (
                ego_in_multiple_lanes_or_nondrivable_area and collision_at_lateral
            ):

                timestamps_at_fault_collisions.append(timestamp)

                at_fault_collisions[collision_data.tracked_object_type].append(collision_data.collision_ego_delta_v)

    return timestamps_at_fault_collisions, at_fault_collisions


class EgoAtFaultCollisionStatistics(MetricBase):


    def __init__(
        self,
        name: str,
        category: str,
        ego_lane_change_metric: EgoLaneChangeStatistics,
        max_violation_threshold_vru: int = 0,
        max_violation_threshold_vehicle: int = 0,
        max_violation_threshold_object: int = 1,
        metric_score_unit: Optional[str] = None,
    ) -> None:

        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self._max_violation_threshold_vru = max_violation_threshold_vru
        self._max_violation_threshold_vehicle = max_violation_threshold_vehicle
        self._max_violation_threshold_object = max_violation_threshold_object


        self.results: List[MetricStatistics] = []
        self.all_collisions: List[Collisions] = []
        self.all_at_fault_collisions: Dict[TrackedObjectType, List[float]] = defaultdict(list)
        self.timestamps_at_fault_collisions: List[int] = []


        self._ego_lane_change_metric = ego_lane_change_metric

    def _compute_collision_score(self, number_of_collisions: int, max_violation_threshold: int) -> float:

        return max(0.0, 1.0 - (number_of_collisions / (max_violation_threshold + 1)))

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> Optional[float]:

        return (
            1
            if metric_statistics[0].value
            else self._compute_collision_score(
                metric_statistics[2].value, self._max_violation_threshold_vru
            )
            * self._compute_collision_score(
                metric_statistics[3].value, self._max_violation_threshold_vehicle
            )
            * self._compute_collision_score(
                metric_statistics[4].value, self._max_violation_threshold_object
            )
        )

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:


        assert self._ego_lane_change_metric.results, "Run ego_lane_change_metric before {}".format(
            self.name
        )
        timestamps_in_common_or_connected_route_objs: List[
            int
        ] = self._ego_lane_change_metric.timestamps_in_common_or_connected_route_objs

        all_collisions: List[Collisions] = []
        collided_track_ids: Set[str] = set()

        for sample in history.data:
            ego_state = sample.ego_state
            observation = sample.observation
            timestamp = ego_state.time_point.time_us

            collided_track_ids, collisions_id_data = find_new_collisions(ego_state, observation, collided_track_ids)


            if len(collisions_id_data):
                all_collisions.append(Collisions(timestamp, collisions_id_data))


        self.timestamps_at_fault_collisions, self.all_at_fault_collisions = classify_at_fault_collisions(
            all_collisions, timestamps_in_common_or_connected_route_objs
        )

        number_of_at_fault_collisions = sum(
            len(track_collisions) for track_collisions in self.all_at_fault_collisions.values()
        )

        statistics = [
            Statistic(
                name=f"{self.name}",
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=number_of_at_fault_collisions == 0,
                type=MetricStatisticsType.BOOLEAN,
            ),
            Statistic(
                name='number_of_all_at_fault_collisions',
                unit=MetricStatisticsType.COUNT.unit,
                value=number_of_at_fault_collisions,
                type=MetricStatisticsType.COUNT,
            ),
        ]
        statistics.extend(get_fault_type_statistics(self.all_at_fault_collisions))


        self.results = self._construct_metric_results(
            metric_statistics=statistics, time_series=None, scenario=scenario, metric_score_unit=self.metric_score_unit
        )
        self.all_collisions = all_collisions

        return self.results
