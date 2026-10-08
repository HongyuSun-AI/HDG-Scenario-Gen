from .pdm_runtime import activate
activate()
from dataclasses import dataclass, asdict
import gc
import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput, PlannerInitialization
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from .pdm_map import MergeMapAdapter
from .pdm_replay import FixedFutureTrajectory, ReplayPDMObservation, make_agent


@dataclass(frozen=True)
class PDMConfig:
    dt: float = .1
    trajectory_steps: int = 80
    proposal_steps: int = 40
    speed_limit_mps: float = 15.
    map_radius_m: float = 50.


class PDMClosedAdapter:
    stop_at_destination = False
    name = 'PDM-Closed (official tuplan_garage; merge.osm adapter)'

    def __init__(self, road, config=None):
        self.road, self.config = road, config or PDMConfig()
        self.map_api = MergeMapAdapter(road, self.config.speed_limit_mps)
        self.vehicle = VehicleParameters(width=1.8, front_length=3.6, rear_length=.9,
            cog_position_from_rear_axle=1.35, wheel_base=2.7, vehicle_name='HDG_assumed',
            vehicle_type='car', height=1.5)

    def describe(self):
        return dict(name=self.name, config=asdict(self.config), controller='nuPlan LQRTracker + KinematicBicycleModel',
                    geometry=dict(length=4.5,width=1.8,rear_axle_to_center=1.35,wheel_base=2.7),
                    map='OSM merge lanes, extended ends',
                    termination='ego centre exits longitudinal crop', route_extension_m=200.)

    def reset(self, initial, ego, destination):
        self.ego, self.destination = int(ego), np.array(destination, copy=True)


        self.map_api = MergeMapAdapter(self.road, self.config.speed_limit_mps)
        self.initial_speeds = np.zeros(len(initial))
        self.previous_observed = None
        self.state = None
        self.frame = 0
        self.audits = []
        c = self.config
        trajectory = TrajectorySampling(num_poses=c.trajectory_steps, interval_length=c.dt)
        proposals = TrajectorySampling(num_poses=c.proposal_steps, interval_length=c.dt)
        policies = BatchIDMPolicy(fallback_target_velocity=c.speed_limit_mps,
            speed_limit_fraction=[.2,.4,.6,.8,1.], min_gap_to_lead_agent=1., headway_time=1.5,
            accel_max=1.5, decel_max=3.)
        self.core = PDMClosedPlanner(trajectory, proposals, policies, [-1.,1.], c.map_radius_m)
        self.core._observation = ReplayPDMObservation(trajectory, proposals, c.map_radius_m)
        self.core.initialize(PlannerInitialization(['merge_roadblock'], StateSE2(self.road.xmax+200.,float(destination[1]),float(destination[2])), self.map_api))
        self.tracker = LQRTracker(q_longitudinal=[10.], r_longitudinal=[1.], q_lateral=[1.,10.,0.],
            r_lateral=[1.], discretization_time=c.dt, tracking_horizon=10, jerk_penalty=1e-4,
            curvature_rate_penalty=1e-2, stopping_proportional_gain=.5, stopping_velocity=.2,
            vehicle=self.vehicle)
        self.motion = KinematicBicycleModel(self.vehicle)

    def set_initial_speeds(self, speeds):
        speeds = np.asarray(speeds, dtype=float)
        self.initial_speeds = speeds.copy()

    def step(self, observed, valid, speed, dt, future_env=None):
        observed, valid = np.asarray(observed, dtype=float), np.asarray(valid, dtype=bool)
        time_s = self.frame*dt
        time = TimePoint(int(round(time_s*1e6)))
        if self.state is None:
            self.state = EgoState.build_from_center(StateSE2(*map(float,observed[self.ego])),
                StateVector2D(float(speed),0.), StateVector2D(0.,0.), 0., time, self.vehicle)
        elif not np.allclose(self.state.center.serialize(), observed[self.ego], atol=2e-3):
            raise ValueError('Ego controller state mismatch')
        velocity = self.initial_speeds[:,None]*np.c_[np.cos(observed[:,2]),np.sin(observed[:,2])]
        if self.previous_observed is not None:
            available = valid & np.isfinite(self.previous_observed).all(-1)
            velocity[available] = (observed[available,:2]-self.previous_observed[available,:2])/dt
        agents = [make_agent(int(i),observed[i],velocity[i],time_s) for i in np.flatnonzero(valid) if i != self.ego]
        observation = DetectionsTracks(TrackedObjects(agents))
        self.previous_observed = np.where(valid[:,None], observed, np.nan).copy()
        history = SimulationHistoryBuffer.initialize_from_list(1, [self.state], [observation], dt)
        current = SimulationIteration(time, self.frame)
        next_iteration = SimulationIteration(TimePoint(time.time_us+int(round(dt*1e6))),self.frame+1)
        future = None
        if future_env is not None:
            future_env = np.asarray(future_env, dtype=float)
            now = np.where(valid[:,None], observed, np.nan)
            future = FixedFutureTrajectory(np.concatenate([now[None],future_env]), self.ego, dt, time_s)
        self.core._observation.set_future(future)
        was_gc_enabled = gc.isenabled()
        try:
            trajectory = self.core.compute_planner_trajectory(PlannerInput(current,history,[]))
        finally:
            if was_gc_enabled: gc.enable()
        command = self.tracker.track_trajectory(current,next_iteration,self.state,trajectory)
        self.state = self.motion.propagate_state(self.state,command,TimePoint(int(round(dt*1e6))))
        self.audits.append(dict(frame=self.frame,**self.core._observation.audit))
        self.frame += 1
        return np.array(self.state.center.serialize()), float(self.state.dynamic_car_state.speed)
