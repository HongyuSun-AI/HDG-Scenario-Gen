import logging
from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np
import numpy.typing as npt

from nuplan.common.maps.abstract_map_objects import GraphEdgeMapObject
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic
from nuplan.planning.metrics.utils.route_extractor import (
    CornersGraphEdgeMapObject,
    extract_corners_route,
    get_common_or_connected_route_objs_of_corners,
    get_outgoing_edges_obj_dict,
    get_route,
    get_timestamps_in_common_or_connected_route_objs,
)
from nuplan.planning.metrics.utils.state_extractors import extract_ego_center, extract_ego_time_point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory

logger = logging.getLogger(__name__)


@dataclass
class LaneChangeStartRecord:


    start_timestamp: int
    initial_lane: Optional[Set[GraphEdgeMapObject]]


@dataclass
class LaneChangeData:


    start_data: LaneChangeStartRecord
    duration_us: float
    final_lane: Optional[Set[GraphEdgeMapObject]]
    success: bool


def _ego_starts_lane_change(
    initial_lane: Optional[Set[GraphEdgeMapObject]], start_timestamp: int
) -> Optional[LaneChangeStartRecord]:


    return LaneChangeStartRecord(start_timestamp, initial_lane) if initial_lane else None


def _ego_ends_lane_change(
    open_lane_change: LaneChangeStartRecord, final_lane: Set[GraphEdgeMapObject], end_timestamp: int
) -> LaneChangeData:


    if not final_lane:
        return LaneChangeData(
            open_lane_change, end_timestamp - open_lane_change.start_timestamp, final_lane=None, success=False
        )

    initial_lane = open_lane_change.initial_lane
    initial_lane_ids = {obj.id for obj in initial_lane}
    initial_lane_out_edge_ids = set(get_outgoing_edges_obj_dict(initial_lane).keys())
    initial_lane_or_out_edge_ids = initial_lane_ids.union(initial_lane_out_edge_ids)
    final_lane_ids = {obj.id for obj in final_lane}

    return LaneChangeData(
        open_lane_change,
        end_timestamp - open_lane_change.start_timestamp,
        final_lane,
        success=False if len(set.intersection(initial_lane_or_out_edge_ids, final_lane_ids)) else True,
    )


def find_lane_changes(
    ego_timestamps: npt.NDArray[np.int32], common_or_connected_route_objs: List[Optional[Set[GraphEdgeMapObject]]]
) -> List[LaneChangeData]:

    lane_changes: List[LaneChangeData] = []
    open_lane_change = None

    if common_or_connected_route_objs[0] is None:
        logging.debug("Initial corners span routes")

    for prev_ind, curr_obj in enumerate(common_or_connected_route_objs[1:]):

        if open_lane_change is None:

            if curr_obj is None:
                open_lane_change = _ego_starts_lane_change(
                    initial_lane=common_or_connected_route_objs[prev_ind], start_timestamp=ego_timestamps[prev_ind + 1]
                )

        else:

            if curr_obj is not None:
                lane_change_data = _ego_ends_lane_change(
                    open_lane_change, final_lane=curr_obj, end_timestamp=ego_timestamps[prev_ind + 1]
                )
                lane_changes.append(lane_change_data)
                open_lane_change = None


    if open_lane_change:
        lane_changes.append(
            LaneChangeData(
                open_lane_change, ego_timestamps[-1] - open_lane_change.start_timestamp, final_lane=None, success=False
            )
        )

    return lane_changes


class EgoLaneChangeStatistics(MetricBase):


    def __init__(self, name: str, category: str, max_fail_rate: float) -> None:

        super().__init__(name=name, category=category)
        self._max_fail_rate = max_fail_rate


        self.ego_driven_route: List[List[Optional[GraphEdgeMapObject]]] = []
        self.corners_route: List[CornersGraphEdgeMapObject] = [CornersGraphEdgeMapObject([], [], [], [])]
        self.timestamps_in_common_or_connected_route_objs: List[int] = []
        self.results: List[MetricStatistics] = []

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:


        ego_states = history.extract_ego_state
        ego_poses = extract_ego_center(ego_states)


        self.ego_driven_route = get_route(history.map_api, ego_poses)


        ego_timestamps = extract_ego_time_point(ego_states)


        ego_footprint_list = [ego_state.car_footprint for ego_state in ego_states]


        corners_route = extract_corners_route(history.map_api, ego_footprint_list)

        self.corners_route = corners_route

        common_or_connected_route_objs = get_common_or_connected_route_objs_of_corners(corners_route)


        timestamps_in_common_or_connected_route_objs = get_timestamps_in_common_or_connected_route_objs(
            common_or_connected_route_objs, ego_timestamps
        )


        self.timestamps_in_common_or_connected_route_objs = timestamps_in_common_or_connected_route_objs


        lane_changes = find_lane_changes(ego_timestamps, common_or_connected_route_objs)

        if len(lane_changes) == 0:
            metric_statistics = [
                Statistic(
                    name=f"number_of_{self.name}",
                    unit=MetricStatisticsType.COUNT.unit,
                    value=0,
                    type=MetricStatisticsType.COUNT,
                ),
                Statistic(
                    name=f"{self.name}_fail_rate_below_threshold",
                    unit=MetricStatisticsType.BOOLEAN.unit,
                    value=True,
                    type=MetricStatisticsType.BOOLEAN,
                ),
            ]

        else:

            lane_change_durations = [lane_change.duration_us * 1e-6 for lane_change in lane_changes]
            failed_lane_changes = [lane_change for lane_change in lane_changes if not lane_change.success]
            failed_ratio = len(failed_lane_changes) / len(lane_changes)
            fail_rate_below_threshold = 1 if self._max_fail_rate >= failed_ratio else 0
            metric_statistics = [
                Statistic(
                    name=f"number_of_{self.name}",
                    unit=MetricStatisticsType.COUNT.unit,
                    value=len(lane_changes),
                    type=MetricStatisticsType.COUNT,
                ),
                Statistic(
                    name=f"max_{self.name}_duration",
                    unit="seconds",
                    value=np.max(lane_change_durations),
                    type=MetricStatisticsType.MAX,
                ),
                Statistic(
                    name=f"avg_{self.name}_duration",
                    unit="seconds",
                    value=float(np.mean(lane_change_durations)),
                    type=MetricStatisticsType.MEAN,
                ),
                Statistic(
                    name=f"ratio_of_failed_{self.name}",
                    unit=MetricStatisticsType.RATIO.unit,
                    value=failed_ratio,
                    type=MetricStatisticsType.RATIO,
                ),
                Statistic(
                    name=f"{self.name}_fail_rate_below_threshold",
                    unit=MetricStatisticsType.BOOLEAN.unit,
                    value=bool(fail_rate_below_threshold),
                    type=MetricStatisticsType.BOOLEAN,
                ),
            ]

        results: List[MetricStatistics] = self._construct_metric_results(
            metric_statistics=metric_statistics, time_series=None, scenario=scenario
        )

        self.results = results

        return results
