from typing import List, Tuple

import theseus as th
import torch

from objective.costs import (
    _TripleIntegrator,
)
from objective.objective import MotionPlannerObjective


class AccelerationBasedObjective(MotionPlannerObjective):
    def __init__(
        self,
        total_time: float,
        horizon: int,
        x_velocity_bounds: Tuple[float, float],
        y_velocity_bounds: Tuple[float, float],
        x_acceleration_bounds: Tuple[float, float],
        y_acceleration_bounds: Tuple[float, float],
        robot_radius: float,
        safety_distance: float,
        local_map_size: int,
        dtype: torch.dtype = torch.double,
        goal_cost: float = 50,
        quadratic_velocity_cost: float = 10,
        quadratic_acceleration_cost: float = 10,
        velocity_bounds_cost: float = 2000,
        acceleration_bounds_cost: float = 2000,
        current_state_cost: float = 2000,
        dynamic_cost: float = 2000,
        collision_cost: float = 2000,
    ):
        super().__init__(
            total_time=total_time,
            horizon=horizon,
            x_velocity_bounds=x_velocity_bounds,
            y_velocity_bounds=y_velocity_bounds,
            x_acceleration_bounds=x_acceleration_bounds,
            y_acceleration_bounds=y_acceleration_bounds,
            robot_radius=robot_radius,
            safety_distance=safety_distance,
            local_map_size=local_map_size,
            dtype=dtype,
            goal_cost=goal_cost,
            quadratic_velocity_cost=quadratic_velocity_cost,
            quadratic_acceleration_cost=quadratic_acceleration_cost,
            velocity_bounds_cost=velocity_bounds_cost,
            acceleration_bounds_cost=acceleration_bounds_cost,
            current_state_cost=current_state_cost,
            dynamic_cost=dynamic_cost,
            collision_cost=collision_cost,
        )

    def _get_variables(self) -> Tuple[List[th.Vector], List[th.Vector]]:
        accelerations: List[th.Vector] = []  # Length = N
        for timestep in range(self.horizon):
            accelerations.append(
                th.Vector(
                    dof=2,
                    name="acceleration_{}".format(timestep),
                    dtype=self.dtype,
                )
            )

        velocities: List[th.Vector] = [
            th.Vector(tensor=self.current_velocity, name="velocity_0", dtype=self.dtype)
        ]
        for timestep in range(1, self.horizon + 1):
            velocities.append(
                th.Vector(
                    tensor=velocities[timestep - 1].tensor
                    + self.dt.tensor * accelerations[timestep - 1].tensor,
                    name="velocity_{}".format(timestep),
                )
            )
        # Length = N + 1

        return velocities, accelerations

    def _get_dynamics_cost(self, timestep: int) -> th.CostFunction:
        return _TripleIntegrator(
            self.states[timestep],
            self.velocities[timestep].tensor,
            self.accelerations[timestep],
            self.states[timestep + 1],
            self.accelerations[timestep + 1],
            self.dt,
            self.dynamic_cost_weight,
            name="triple_integrator_cost_{}".format(timestep),
        )
