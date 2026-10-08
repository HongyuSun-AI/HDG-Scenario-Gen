from .pdm_runtime import activate
activate()
import hashlib
import numpy as np
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap


def make_agent(index, state, velocity, time_s, length=4.5, width=1.8):
    token = f'hdg_agent_{index}'
    return Agent(TrackedObjectType.VEHICLE, OrientedBox(StateSE2(*map(float,state)), length, width, 1.5),
        StateVector2D(*map(float,velocity)), SceneObjectMetadata(int(round(time_s*1e6)), token, index, token, 'vehicle'))


class FixedFutureTrajectory:
    # states[T,N,3]
    def __init__(self, states, ego, dt=.1, start_time_s=0.):
        x = np.array(states, dtype=np.float64, copy=True)
        mixed = np.isfinite(x).any(-1) & ~np.isfinite(x).all(-1)
        if mixed.any():
            raise ValueError('Mixed finite/nonfinite actor channels')
        x[:, ego] = np.nan
        self.states, self.ego, self.dt, self.start_time_s = x, ego, float(dt), float(start_time_s)
        self.valid = np.isfinite(x).all(-1)
        self.states.setflags(write=False); self.valid.setflags(write=False)
        self.sha256 = hashlib.sha256(x.tobytes()).hexdigest()

    def at(self, time_s):
        relative = (time_s - self.start_time_s) / self.dt
        index = int(round(relative))
        if index < 0 or abs(relative-index) > 1e-5:
            raise ValueError('Invalid replay query time')
        return self.states[min(index, len(self.states)-1)], self.valid[min(index, len(self.states)-1)], index >= len(self.states)

    def velocity(self, index, actor):
        if index >= len(self.states) or not self.valid[index, actor]:
            return np.zeros(2)
        lo, hi = max(index-1,0), min(index+1,len(self.states)-1)
        if not self.valid[lo,actor]: lo = index
        if not self.valid[hi,actor]: hi = index
        if hi == lo: return np.zeros(2)
        return (self.states[hi,actor,:2] - self.states[lo,actor,:2]) / ((hi-lo)*self.dt)


class ReplayPDMObservation(PDMObservation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, observation_sample_res=1, **kwargs)
        self.future = None
        self._objects_by_sample = None
        self._lookup_index = 0
        self.audit = {}

    def set_future(self, future):
        self.future = future

    def __getitem__(self, time_idx):
        self._lookup_index = int(time_idx)
        return super().__getitem__(time_idx)

    @property
    def unique_objects(self):
        if self._objects_by_sample is not None:
            return self._objects_by_sample[self._lookup_index]
        return super().unique_objects

    def update(self, ego_state, observation, traffic_light_data, route_lane_dict):
        self._objects_by_sample = None

        super().update(ego_state, observation, traffic_light_data, route_lane_dict)
        if self.future is None:
            self.audit = {'mode':'constant_velocity', 'known_future_used':False}
            return
        now = ego_state.time_point.time_s
        current, valid, _ = self.future.at(now)
        for obj in observation.tracked_objects:
            index = int(obj.track_token.rsplit('_',1)[1])
            if not valid[index] or not np.allclose(current[index], obj.center.serialize(), atol=1e-5):
                raise ValueError('Replay observation mismatch')
        tokens, lights = self._get_traffic_light_geometries(traffic_light_data, route_lane_dict)
        maps, objects, padded = [], [], 0
        for sample in range(self._observation_samples + 1):
            time_s = now + sample*self._sample_interval
            states, valid, extended = self.future.at(time_s)
            index = int(round((time_s-self.future.start_time_s)/self.future.dt))
            frame_objects = {}
            for actor in np.flatnonzero(valid):
                agent = make_agent(int(actor), states[actor], self.future.velocity(index, int(actor)), time_s)
                frame_objects[agent.track_token] = agent
            keys = list(frame_objects)
            maps.append(PDMOccupancyMap(keys+tokens, [frame_objects[k].box.geometry for k in keys]+lights))
            objects.append(frame_objects)
            padded += int(extended)
        self._occupancy_maps, self._objects_by_sample = maps, objects
        self._lookup_index = 0
        self.audit = dict(mode='fixed_future_replay', known_future_used=True, sha256=self.future.sha256,
            replay_start_s=self.future.start_time_s, replay_end_s=self.future.start_time_s+(len(self.future.states)-1)*self.future.dt,
            padded_forecast_frames=padded, tail_policy='Stationary beyond known horizon', ego_excluded=True)
