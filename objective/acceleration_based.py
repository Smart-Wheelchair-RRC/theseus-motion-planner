from typing import List, Tuple

import theseus as th
import torch
from theseus.embodied import Collision2D, HingeCost

from objective.costs import (
    _BothXYDifference,
    _QuadraticVectorCost,
    _TripleIntegrator,
    _XYDifference,
)


class MotionPlannerObjective(th.Objective):
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
        super().__init__(dtype=dtype)

        # -------- Auxiliary variables --------
        current_state = th.SE2(name="current_state", dtype=dtype)
        current_velocity = th.Vector(dof=2, name="current_velocity", dtype=dtype)
        goal_position = th.Point2(name="goal_position", dtype=dtype)
        current_velocity: th.Vector = th.Vector(
            tensor=current_velocity, name="current_velocity", dtype=dtype
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
                torch.tensor([[goal_cost, goal_cost]], dtype=dtype),
                name="goal_cost_weight_variable",
            ),
            name="goal_cost_weight",
        )
        quadratic_velocity_cost_weight = th.DiagonalCostWeight(
            th.Variable(
                torch.tensor(
                    [[quadratic_velocity_cost, quadratic_velocity_cost]], dtype=dtype
                ),
                name="quadratic_velocity_cost_weight_variable",
            ),
            name="quadratic_velocity_cost_weight",
        )
        quadratic_acceleration_cost_weight = th.DiagonalCostWeight(
            th.Variable(
                torch.tensor(
                    [[quadratic_acceleration_cost, quadratic_acceleration_cost]],
                    dtype=dtype,
                ),
                name="quadratic_acceleration_cost_weight_variable",
            ),
            name="quadratic_acceleration_cost_weight",
        )

        # -------- Constraint Cost Weights --------
        velocity_bounds_cost_weight = th.ScaleCostWeight(
            th.Variable(
                torch.tensor(velocity_bounds_cost, dtype=dtype),
                name="velocity_bounds_cost_weight_variable",
            ),
            name="velocity_bounds_cost_weight",
        )
        acceleration_bounds_cost_weight = th.ScaleCostWeight(
            th.Variable(
                torch.tensor(acceleration_bounds_cost, dtype=dtype),
                name="acceleration_bounds_cost_weight_variable",
            ),
            name="acceleration_bounds_cost_weight",
        )
        current_state_cost_weight = th.DiagonalCostWeight(
            th.Variable(
                torch.tensor([[current_state_cost, current_state_cost]], dtype=dtype),
                name="current_state_cost_weight_variable",
            ),
            name="current_state_cost_weight",
        )
        dynamic_cost_weight = th.ScaleCostWeight(
            th.Variable(
                torch.tensor(dynamic_cost, dtype=dtype),
                name="dynamic_cost_weight_variable",
            ),
            name="dynamic_cost_weight",
        )
        collision_cost_weight = th.ScaleCostWeight(
            th.Variable(
                torch.tensor(collision_cost, dtype=dtype),
                name="collision_cost_weight_variable",
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

        # -------- Derived Variables --------
        velocities: List[th.Vector] = [
            th.Vector(tensor=current_velocity, name="velocity_0", dtype=dtype)
        ]

        for timestep in range(1, horizon + 1):
            velocities.append(
                th.Vector(
                    tensor=velocities[timestep - 1].tensor
                    + dt.tensor * accelerations[timestep - 1].tensor,
                    name="velocity_{}".format(timestep),
                )
            )
        # Length = N + 1

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

        zero_vector = th.Vector(
            tensor=torch.tensor([0, 0], dtype=dtype),
            name="zero_vector",
            dtype=dtype,
        )

        ## Velocity costs
        for velocity in velocities[1:]:
            self.add(
                _QuadraticVectorCost(
                    velocity,
                    zero_vector,
                    quadratic_velocity_cost_weight,
                    name="quadratic_cost_{}".format(velocity.name),
                )
            )

        ## Acceleration costs
        for acceleration in accelerations:
            self.add(
                _QuadraticVectorCost(
                    acceleration,
                    zero_vector,
                    quadratic_acceleration_cost_weight,
                    name="quadratic_cost_{}".format(acceleration.name),
                )
            )

        # -------- Soft Constraints --------
        zero_threshold = th.Vector(
            tensor=torch.tensor([0, 0], dtype=dtype),
            name="zero_threshold",
            dtype=dtype,
        )

        ## Velocity bounds costs
        lower_velocity_bounds = th.Vector(
            tensor=torch.tensor(
                [
                    x_velocity_bounds[0],
                    y_velocity_bounds[0],
                ],
                dtype=dtype,
            ),
            name="lower_velocity_bounds",
            dtype=dtype,
        )
        upper_velocity_bounds = th.Vector(
            tensor=torch.tensor(
                [
                    x_velocity_bounds[1],
                    y_velocity_bounds[1],
                ],
                dtype=dtype,
            ),
            name="upper_velocity_bounds",
            dtype=dtype,
        )
        for velocity in velocities[1:]:
            self.add(
                HingeCost(
                    velocity,
                    down_limit=lower_velocity_bounds,
                    up_limit=upper_velocity_bounds,
                    threshold=zero_threshold,
                    cost_weight=velocity_bounds_cost_weight,
                    name="bounds_{}".format(velocity.name),
                )
            )

        ## Acceleration bound costs
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
        for acceleration in accelerations:
            self.add(
                HingeCost(
                    vector=acceleration,
                    down_limit=lower_acceleration_bounds,
                    up_limit=upper_acceleration_bounds,
                    threshold=zero_threshold,
                    cost_weight=acceleration_bounds_cost_weight,
                    name="bounds_{}".format(acceleration.name),
                )
            )

        ## Third Order Dynamics Cost
        for timestep in range(horizon - 1):
            self.add(
                _TripleIntegrator(
                    states[timestep],
                    velocities[timestep].tensor,
                    accelerations[timestep],
                    states[timestep + 1],
                    accelerations[timestep + 1],
                    dt,
                    dynamic_cost_weight,
                    name="triple_integrator_cost_{}".format(timestep),
                )
            )

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
        for timestep in range(1, horizon + 1):
            self.add(
                Collision2D(
                    states[timestep],
                    sdf_origin=sdf_origin,
                    sdf_data=sdf_data,
                    sdf_cell_size=cell_size,
                    cost_eps=collision_cost_eps,
                    cost_weight=collision_cost_weight,
                    name="collision_cost_{}".format(timestep),
                )
            )
