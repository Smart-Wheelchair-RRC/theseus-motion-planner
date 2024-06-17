from typing import List, Optional, Tuple, Union

import theseus as th
import torch
from theseus.core import CostFunction, CostWeight, Variable, as_variable
from theseus.embodied import Collision2D, HingeCost
from theseus.geometry import LieGroup, Vector


class _TripleIntegrator(CostFunction):
    def __init__(
        self,
        pose1: LieGroup,
        vel1: Vector,
        acc1: Vector,
        pose2: LieGroup,
        vel2: Vector,
        acc2: Vector,
        dt: Union[float, torch.Tensor, Variable],
        cost_weight: CostWeight,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)
        dof = pose1.dof()
        if not (acc1.dof() == pose2.dof() == acc2.dof() == dof):
            raise ValueError(
                "All variables for a TripleIntegrator must have the same dimension."
            )

        self.dt = as_variable(dt)
        if self.dt.tensor.squeeze().ndim > 1:
            raise ValueError(
                "dt data must be a 0-D or 1-D tensor with numel in {1, batch_size}."
            )
        self.dt.tensor = self.dt.tensor.view(-1, 1)

        self.pose1 = pose1
        self.vel1 = vel1
        self.acc1 = acc1
        self.pose2 = pose2
        self.vel2 = vel2
        self.acc2 = acc2

        self.register_optim_vars(["pose1", "acc1", "pose2", "acc2"])
        self.register_aux_vars(["vel1", "vel2", "dt"])
        self.weight = cost_weight

    def dim(self):
        return 3 * self.pose1.dof()

    def _new_pose_diff(
        self, jacobians: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        return self.pose1.local(self.pose2, jacobians=jacobians)

    def _error_from_pose_diff(self, pose_diff: torch.Tensor) -> torch.Tensor:
        pose_diff_err = (
            pose_diff
            - self.dt.tensor.view(-1, 1) * self.vel1.tensor
            - 0.5 * self.dt.tensor.view(-1, 1) ** 2 * self.acc1.tensor
        )
        vel_diff = self.vel2.tensor - (
            self.vel1.tensor + self.dt.tensor.view(-1, 1) * self.acc1.tensor
        )
        acc_diff = self.acc2.tensor - self.acc1.tensor
        return torch.cat([pose_diff_err, vel_diff, acc_diff], dim=1)

    def error(self) -> torch.Tensor:
        return self._error_from_pose_diff(self._new_pose_diff())

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        batch_size = self.pose1.shape[0]
        dof = self.pose1.dof()
        dtype = self.pose1.dtype
        device = self.pose1.device

        Jerr_pose1 = torch.zeros(batch_size, 3 * dof, dof, dtype=dtype, device=device)
        Jerr_acc1 = torch.zeros_like(Jerr_pose1)
        Jerr_pose2 = torch.zeros_like(Jerr_pose1)
        Jerr_acc2 = torch.zeros_like(Jerr_pose1)

        Jlocal: List[torch.Tensor] = []
        error = self._error_from_pose_diff(self._new_pose_diff(Jlocal))

        Jerr_pose1[:, :dof, :] = Jlocal[0]
        identity = torch.eye(dof, dtype=dtype, device=device).repeat(batch_size, 1, 1)

        Jerr_acc1[:, :dof, :] = -0.5 * self.dt.tensor.view(-1, 1, 1) ** 2 * identity
        Jerr_acc1[:, dof : 2 * dof, :] = -self.dt.tensor.view(-1, 1, 1) * identity
        Jerr_acc1[:, 2 * dof :, :] = -identity

        Jerr_pose2[:, :dof, :] = Jlocal[1]
        Jerr_acc2[:, 2 * dof :, :] = identity

        return [Jerr_pose1, Jerr_acc1, Jerr_pose2, Jerr_acc2], error

    def _copy_impl(self, new_name: Optional[str] = None) -> "_TripleIntegrator":
        return _TripleIntegrator(
            self.pose1.copy(),
            self.vel1.copy(),
            self.acc1.copy(),
            self.pose2.copy(),
            self.vel2.copy(),
            self.acc2.copy(),
            self.dt.copy(),
            self.weight.copy(),
            name=new_name,
        )


class _QuadraticAccelerationCost(th.CostFunction):
    def __init__(
        self,
        var: th.Vector,
        target: th.Vector,
        cost_weight: th.CostWeight,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)

        if not isinstance(var, th.Vector) and not isinstance(target, th.Vector):
            raise ValueError(
                "QuadraticVelocityCost expects var and target of type Vector."
            )

        self.var = var
        self.target = target
        self.register_optim_vars(["var"])
        self.register_aux_vars(["target"])

    def _jacobians_and_error_impl(
        self, compute_jacobians: bool = False
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        Jlocal: List[torch.Tensor] = [] if compute_jacobians else None
        error = self.target.local(self.var, jacobians=Jlocal).pow(2)
        jac = [2 * Jlocal[1]] if compute_jacobians else None
        return jac, error

    def error(self) -> torch.Tensor:
        # Squared difference between var and target
        return self._jacobians_and_error_impl(compute_jacobians=False)[1]

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self._jacobians_and_error_impl(compute_jacobians=True)

    def dim(self) -> int:
        return self.var.dof

    def _copy_impl(
        self, new_name: Optional[str] = None
    ) -> "_QuadraticAccelerationCost":
        return _QuadraticAccelerationCost(  # type: ignore
            self.var.copy(), self.target.copy(), self.weight.copy(), name=new_name
        )


class _XYDifference(th.CostFunction):
    def __init__(
        self,
        var: th.SE2,
        target: th.Point2,
        cost_weight: th.CostWeight,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)
        if not isinstance(var, th.SE2) and not isinstance(target, th.Point2):
            raise ValueError(
                "XYDifference expects var of type SE2 and target of type Point2."
            )
        self.var = var
        self.target = target
        self.register_optim_vars(["var"])
        self.register_aux_vars(["target"])

    def _jacobians_and_error_impl(
        self, compute_jacobians: bool = False
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        Jlocal: List[torch.Tensor] = [] if compute_jacobians else None
        Jxy: List[torch.Tensor] = [] if compute_jacobians else None
        error = self.target.local(self.var.xy(jacobians=Jxy), jacobians=Jlocal)
        jac = [Jlocal[1].matmul(Jxy[0])] if compute_jacobians else None
        return jac, error

    def error(self) -> torch.Tensor:
        return self._jacobians_and_error_impl(compute_jacobians=False)[1]

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self._jacobians_and_error_impl(compute_jacobians=True)

    def dim(self) -> int:
        return 2

    def _copy_impl(self, new_name: Optional[str] = None) -> "_XYDifference":
        return _XYDifference(  # type: ignore
            self.var.copy(), self.target.copy(), self.weight.copy(), name=new_name
        )


class MotionPlannerObjective(th.Objective):
    def __init__(
        self,
        dt: float,
        current_velocity: torch.Tensor,
        linear_velocity_bounds: Tuple[float, float],
        angular_velocity_bounds: Tuple[float, float],
        horizon: int,
        map_size: float,
        epsilon_dist: float,
        linear_acceleration_bounds: Tuple[float, float],
        angular_acceleration_bounds: Tuple[float, float],
        dtype: torch.dtype = torch.double,
    ):
        super().__init__()

        # Auxiliary variables

        current_state = th.SE2(name="current_state", dtype=dtype)
        goal_position = th.Point2(name="goal_position", dtype=dtype)
        current_velocity = th.Vector(
            tensor=current_velocity, name="current_linear_velocity", dtype=dtype
        )
        dt = th.Variable(torch.tensor(dt).view(1, 1), name="dt", dtype=dtype)

        # Cost Weights
        goal_cost_weight = th.DiagonalCostWeight(
            th.Variable(torch.tensor([100, 100, 0]), name="goal_cost_weight_variable"),
            name="goal_cost_weight",
        )
        quadratic_acceleration_cost_weight = th.DiagonalCostWeight(
            th.Variable(
                torch.tensor([50, 30]),
                name="quadratic_acceleration_cost_weight_variable",
            ),
            name="quadratic_acceleration_cost_weight",
        )
        control_cost_weight = th.ScaleCostWeight(
            th.Variable(torch.tensor(500), name="control_cost_weight_variable"),
            name="control_cost_weight",
        )
        collision_cost_weight = th.ScaleCostWeight(
            th.Variable(torch.tensor(500), name="collision_cost_weight_variable"),
            name="collision_cost_weight",
        )

        # Optimization variables
        states: List[th.SE2] = []
        accelerations: List[th.Vector] = []

        for timestep in range(horizon):
            states.append(th.SE2(name="state_{}".format(timestep), dtype=dtype))
            accelerations.append(
                th.Vector(
                    name="acceleration_{}".format(timestep),
                    dtype=dtype,
                )
            )

        # Cost functions
        ## Goal cost
        self.add(
            _XYDifference(
                states[-1],
                goal_position,
                goal_cost_weight,
                name="goal_cost",
            )
        )
        ## Acceleration costs
        for acceleration in accelerations:
            self.add(
                _QuadraticAccelerationCost(
                    accelerations,
                    th.Vector(
                        tensor=torch.tensor([0, 0]),
                        name="zero_acceleration",
                        dtype=dtype,
                    ),
                    quadratic_acceleration_cost_weight,
                    name="quadratic_acceleration_cost",
                )
            )

        ## Control bounds costs
        for acceleration in accelerations:
            self.add(
                HingeCost(
                    acceleration,
                    down_limit=th.Vector(
                        tensor=torch.tensor(
                            [
                                linear_acceleration_bounds[0],
                                angular_acceleration_bounds[0],
                            ]
                        ),
                        name="lower_acceleration_bounds",
                        dtype=dtype,
                    ),
                    up_limit=th.Vector(
                        tensor=torch.tensor(
                            [
                                linear_acceleration_bounds[1],
                                angular_acceleration_bounds[1],
                            ]
                        ),
                        name="upper_acceleration_bounds",
                        dtype=dtype,
                    ),
                    threshold=0,
                    weight=control_cost_weight,
                    name="linear_acceleration_bounds",
                )
            )

        ## Third Order Dynamics Cost
        velocity = current_velocity
        for i in range(horizon - 1):
            self.add(
                _TripleIntegrator(
                    states[i],
                    velocity,
                    accelerations[i],
                    states[i + 1],
                    velocity,
                    accelerations[i + 1],
                    dt,
                    quadratic_acceleration_cost_weight,
                    name="triple_integrator_cost_{}".format(i),
                )
            )
            velocity = velocity + dt * accelerations[i].tensor

        # Velocity bound costs
        previous_velocity = current_velocity
        x_velocity_bounds = linear_velocity_bounds
        y_velocity_bounds = angular_velocity_bounds
        velocity_bounds = (
            torch.tensor(x_velocity_bounds[0], y_velocity_bounds[0]),
            torch.tensor(x_velocity_bounds[1], y_velocity_bounds[1]),
        )
        for acceleration in accelerations:
            # Bound the acceleration by velocity it produces
            self.add(
                HingeCost(
                    vector=acceleration,
                    down_limit=(velocity_bounds[0] - previous_velocity) / dt,
                    up_limit=(velocity_bounds[1] - previous_velocity) / dt,
                    threshold=0,
                    weight=control_cost_weight,
                    name="velocity_bounds",
                )
            )
            previous_velocity = previous_velocity + dt * acceleration.tensor

        # Collision cost
        for i in range(horizon):
            self.add(
                Collision2D(
                    states[i],
                    sdf_origin=th.Point2(name="sdf_origin", dtype=dtype),
                    sdf_data=th.Variable(
                        torch.empty(1, map_size, map_size, dtype=dtype), name="sdf_data"
                    ),
                    sdf_cell_size=th.Variable(
                        torch.empty(1, 1, dtype=dtype), name="cell_size"
                    ),
                    cost_eps=th.Variable(
                        torch.tensor(epsilon_dist, dtype=dtype).view(1, 1),
                        name="cost_eps",
                    ),
                    cost_weight=collision_cost_weight,
                    name="collision_cost_{}".format(i),
                )
            )
