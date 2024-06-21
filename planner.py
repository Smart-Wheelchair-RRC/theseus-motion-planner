from typing import List, Tuple

import theseus as th
import torch
from theseus.embodied import Collision2D, HingeCost

from costs import (
    _BothXYDifference,
    _QuadraticAccelerationCost,
    _TripleIntegrator,
    _XYDifference,
)


class MotionPlannerObjective(th.Objective):
    def __init__(
        self,
        total_time: float,
        horizon: int,
        current_velocity: torch.Tensor,
        x_velocity_bounds: Tuple[float, float],
        y_velocity_bounds: Tuple[float, float],
        x_acceleration_bounds: Tuple[float, float],
        y_acceleration_bounds: Tuple[float, float],
        robot_radius: float,
        safety_distance: float,
        local_map_size: int,
        dtype: torch.dtype = torch.double,
    ):
        super().__init__(dtype=dtype)

        # -------- Auxiliary variables --------
        current_state = th.SE2(name="current_state", dtype=dtype)
        goal_position = th.Point2(name="goal_position", dtype=dtype)
        current_velocity: th.Vector = th.Vector(
            tensor=current_velocity, name="current_linear_velocity", dtype=dtype
        )
        dt: th.Variable = th.Variable(
            torch.tensor(total_time / horizon, dtype=dtype).view(1, 1), name="dt"
        )
        collision_cost_eps = th.Variable(
            torch.tensor(robot_radius + safety_distance, dtype=dtype).view(1, 1),
            name="cost_eps",
        )
        sdf_origin = th.Point2(name="sdf_origin", dtype=dtype)
        cell_size = th.Variable(torch.empty(1, 1, dtype=dtype), name="cell_size")
        sdf_data = th.Variable(
            torch.empty(1, local_map_size, local_map_size, dtype=dtype), name="sdf_data"
        )

        # -------- Cost Weights --------
        goal_cost_weight = th.DiagonalCostWeight(
            th.Variable(
                torch.tensor([[100, 100]], dtype=dtype),
                name="goal_cost_weight_variable",
            ),
            name="goal_cost_weight",
        )
        quadratic_acceleration_cost_weight = th.DiagonalCostWeight(
            th.Variable(
                torch.tensor([[50, 30]], dtype=dtype),
                name="quadratic_acceleration_cost_weight_variable",
            ),
            name="quadratic_acceleration_cost_weight",
        )

        # -------- Constraint Cost Weights --------
        control_cost_weight = th.ScaleCostWeight(
            th.Variable(
                torch.tensor(500, dtype=dtype), name="control_cost_weight_variable"
            ),
            name="control_cost_weight",
        )
        velocity_cost_weight = th.ScaleCostWeight(
            th.Variable(
                torch.tensor(500, dtype=dtype), name="velocity_cost_weight_variable"
            ),
            name="velocity_cost_weight",
        )
        current_state_cost_weight = th.DiagonalCostWeight(
            th.Variable(
                torch.tensor([[500.0, 500.0]], dtype=dtype),
                name="current_state_cost_weight_variable",
            ),
            name="current_state_cost_weight",
        )
        dynamic_cost_weight = th.ScaleCostWeight(
            th.Variable(
                torch.tensor(500, dtype=dtype), name="dynamic_cost_weight_variable"
            ),
            name="dynamic_cost_weight",
        )
        collision_cost_weight = th.ScaleCostWeight(
            th.Variable(
                torch.tensor(700, dtype=dtype), name="collision_cost_weight_variable"
            ),
            name="collision_cost_weight",
        )

        # -------- Optimization variables --------
        states: List[th.SE2] = []  # Length = N + 1
        accelerations: List[th.Vector] = []  # Length = N

        for timestep in range(horizon + 1):
            states.append(th.SE2(name="state_{}".format(timestep), dtype=dtype))

        for timestep in range(horizon):
            accelerations.append(
                th.Vector(
                    dof=2,
                    name="acceleration_{}".format(timestep),
                    dtype=dtype,
                )
            )

        # -------- Cost Functions --------
        ## Goal cost
        for state in states[1:]:
            self.add(
                _XYDifference(
                    state,
                    goal_position,
                    goal_cost_weight,
                    name="goal_cost_{}".format(state.name),
                )
            )

        ## Acceleration costs
        zero_acceleration = th.Vector(
            tensor=torch.tensor([0, 0], dtype=dtype),
            name="zero_acceleration",
            dtype=dtype,
        )
        for acceleration in accelerations[1:]:
            self.add(
                _QuadraticAccelerationCost(
                    acceleration,
                    zero_acceleration,
                    quadratic_acceleration_cost_weight,
                    name="quadratic_acceleration_cost_{}".format(acceleration.name),
                )
            )

        # -------- Soft Constraints --------
        ## Control bounds costs
        zero_threshold = th.Vector(
            tensor=torch.tensor([0, 0], dtype=dtype),
            name="zero_threshold",
            dtype=dtype,
        )
        lower_acceleration_bounds = th.Vector(
            tensor=torch.tensor(
                [
                    x_acceleration_bounds[0],
                    y_acceleration_bounds[0],
                ],
                dtype=dtype,
            ),
            name="lower_acceleration_bounds",
            dtype=dtype,
        )
        upper_acceleration_bounds = th.Vector(
            tensor=torch.tensor(
                [
                    x_acceleration_bounds[1],
                    y_acceleration_bounds[1],
                ],
                dtype=dtype,
            ),
            name="upper_acceleration_bounds",
            dtype=dtype,
        )
        for acceleration in accelerations[1:]:
            self.add(
                HingeCost(
                    acceleration,
                    down_limit=lower_acceleration_bounds,
                    up_limit=upper_acceleration_bounds,
                    threshold=zero_threshold,
                    cost_weight=control_cost_weight,
                    name="linear_acceleration_bounds_{}".format(acceleration.name),
                )
            )

        ## Velocity bound costs
        previous_velocity = current_velocity.tensor
        velocity_bounds = (
            torch.tensor([x_velocity_bounds[0], y_velocity_bounds[0]], dtype=dtype),
            torch.tensor([x_velocity_bounds[1], y_velocity_bounds[1]], dtype=dtype),
        )
        for acceleration in accelerations:
            # Bound the acceleration by velocity it produces
            self.add(
                HingeCost(
                    vector=acceleration,
                    down_limit=(velocity_bounds[0] - previous_velocity) / dt.tensor,
                    up_limit=(velocity_bounds[1] - previous_velocity) / dt.tensor,
                    threshold=zero_threshold,
                    cost_weight=velocity_cost_weight,
                    name="velocity_bounds_{}".format(acceleration.name),
                )
            )
            previous_velocity = previous_velocity + dt.tensor * acceleration.tensor

        ## Third Order Dynamics Cost
        velocity = current_velocity.tensor
        for i in range(horizon - 1):
            self.add(
                _TripleIntegrator(
                    states[i],
                    velocity,
                    accelerations[i],
                    states[i + 1],
                    accelerations[i + 1],
                    dt,
                    dynamic_cost_weight,
                    name="triple_integrator_cost_{}".format(i),
                )
            )
            velocity = velocity + dt.tensor * accelerations[i].tensor

        ## Current State Cost
        self.add(
            _BothXYDifference(
                states[0],
                current_state,
                current_state_cost_weight,
                name="current_state_cost",
            )
        )
        ## Collision cost
        for i in range(1, horizon + 1):
            self.add(
                Collision2D(
                    states[i],
                    sdf_origin=sdf_origin,
                    sdf_data=sdf_data,
                    sdf_cell_size=cell_size,
                    cost_eps=collision_cost_eps,
                    cost_weight=collision_cost_weight,
                    name="collision_cost_{}".format(i),
                )
            )
