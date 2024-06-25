from typing import List, Optional, Tuple, Union

import theseus as th
import torch
from theseus.core import CostFunction, CostWeight, Variable, as_variable
from theseus.geometry import SE2, Vector


class _DoubleIntegrator(CostFunction):
    def __init__(
        self,
        pose1: SE2,
        vel1: Vector,
        pose2: SE2,
        vel2: Vector,
        dt: Union[float, torch.Tensor, Variable],
        cost_weight: CostWeight,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)
        self.dt = as_variable(dt)
        if self.dt.tensor.squeeze().ndim > 1:
            raise ValueError(
                "dt data must be a 0-D or 1-D tensor with numel in {1, batch_size}."
            )
        self.dt.tensor = self.dt.tensor.view(-1, 1)
        self.pose1 = pose1
        self.vel1 = vel1
        self.pose2 = pose2
        self.vel2 = vel2
        self.register_optim_vars(["pose1", "vel1", "pose2", "vel2"])
        self.register_aux_vars(["dt"])
        self.weight = cost_weight

    def dim(self):
        return 2 * self.vel1.dof()

    def _new_pose_diff(
        self,
        jacobians_local: Optional[List[torch.Tensor]] = None,
        jacobians_xy_1: Optional[List[torch.Tensor]] = None,
        jacobians_xy_2: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        return self.pose1.xy(jacobians=jacobians_xy_1).local(
            self.pose2.xy(jacobians=jacobians_xy_2), jacobians=jacobians_local
        )

    def _error_from_pose_diff(self, pose_diff: torch.Tensor) -> torch.Tensor:
        pose_diff_err = pose_diff - self.dt.tensor.view(-1, 1) * self.vel1.tensor
        vel_diff = self.vel2.tensor - self.vel1.tensor
        return torch.cat([pose_diff_err, vel_diff], dim=1)

    def error(self) -> torch.Tensor:
        return self._error_from_pose_diff(self._new_pose_diff())

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # Pre-allocate jacobian tensors
        batch_size = self.pose1.shape[0]
        out_dof = self.vel1.dof()
        pose_dof = self.pose1.dof()
        dtype = self.pose1.dtype
        device = self.pose1.device
        Jerr_pose1 = torch.zeros(
            batch_size, 2 * out_dof, pose_dof, dtype=dtype, device=device
        )
        Jerr_vel1 = torch.zeros(
            batch_size, 2 * out_dof, out_dof, dtype=dtype, device=device
        )
        Jerr_pose2 = torch.zeros_like(Jerr_pose1)
        Jerr_vel2 = torch.zeros_like(Jerr_vel1)

        Jlocal: List[torch.Tensor] = []
        Jxy1: List[torch.Tensor] = []
        Jxy2: List[torch.Tensor] = []
        error = self._error_from_pose_diff(self._new_pose_diff(Jlocal, Jxy1, Jxy2))
        Jlocal[0] = Jlocal[0].matmul(Jxy1[0])
        Jlocal[1] = Jlocal[1].matmul(Jxy2[0])

        Jerr_pose1[:, :out_dof, :] = Jlocal[0]
        identity = torch.eye(out_dof, dtype=dtype, device=device).repeat(
            batch_size, 1, 1
        )
        Jerr_vel1[:, :out_dof, :] = -self.dt.tensor.view(-1, 1, 1) * identity
        Jerr_vel1[:, out_dof:, :] = -identity
        Jerr_pose2[:, :out_dof, :] = Jlocal[1]
        Jerr_vel2[:, out_dof:, :] = identity
        return [Jerr_pose1, Jerr_vel1, Jerr_pose2, Jerr_vel2], error

    def _copy_impl(self, new_name: Optional[str] = None) -> "_DoubleIntegrator":
        return _DoubleIntegrator(
            self.pose1.copy(),
            self.vel1.copy(),
            self.pose2.copy(),
            self.vel2.copy(),
            self.dt.copy(),
            self.weight.copy(),
            name=new_name,
        )


class _TripleIntegrator(CostFunction):
    def __init__(
        self,
        pose1: SE2,
        vel1: torch.Tensor,
        acc1: Vector,
        pose2: SE2,
        # vel2: Vector,
        acc2: Vector,
        dt: Union[float, torch.Tensor, Variable],
        cost_weight: CostWeight,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)
        # dof = pose1.dof()
        # if not (acc1.dof() == pose2.dof() == acc2.dof() == dof):
        #     print(pose1.dof(), acc1.dof(), acc2.dof())
        #     raise ValueError(
        #         "All variables for a TripleIntegrator must have the same dimension."
        #     )

        self.dt = as_variable(dt, dtype=pose1.dtype)
        self.vel1 = as_variable(vel1, dtype=pose1.dtype)
        if self.dt.tensor.squeeze().ndim > 1:
            raise ValueError(
                "dt data must be a 0-D or 1-D tensor with numel in {1, batch_size}."
            )
        if self.vel1.tensor.squeeze().ndim > 2:
            raise ValueError(
                "vel1 data must be a 2-D tensor with shape (batch_size, 2)."
            )

        self.dt.tensor = self.dt.tensor.view(-1, 1)
        self.vel1.tensor = self.vel1.tensor.view(-1, 2)

        self.pose1 = pose1
        self.acc1 = acc1
        self.pose2 = pose2
        # self.vel2 = vel2
        self.acc2 = acc2

        self.register_optim_vars(["pose1", "acc1", "pose2", "acc2"])
        self.register_aux_vars(["vel1", "dt"])
        self.weight = cost_weight

    def dim(self):
        return 2 * self.acc1.dof()

    def _new_pose_diff(
        self,
        jacobians_local: Optional[List[torch.Tensor]] = None,
        jacobians_xy_1: Optional[List[torch.Tensor]] = None,
        jacobians_xy_2: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        return self.pose1.xy(jacobians=jacobians_xy_1).local(
            self.pose2.xy(jacobians=jacobians_xy_2), jacobians=jacobians_local
        )

    def _error_from_pose_diff(self, pose_diff: torch.Tensor) -> torch.Tensor:
        pose_diff_err = (
            pose_diff
            - self.dt.tensor.view(-1, 1) * self.vel1.tensor
            - 0.5 * self.dt.tensor.view(-1, 1) ** 2 * self.acc1.tensor
        )
        # vel_diff = self.vel2.tensor - (
        #     self.vel1.tensor + self.dt.tensor.view(-1, 1) * self.acc1.tensor
        # )
        acc_diff = self.acc2.tensor - self.acc1.tensor
        return torch.cat([pose_diff_err, acc_diff], dim=1)

    def error(self) -> torch.Tensor:
        return self._error_from_pose_diff(self._new_pose_diff())

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        batch_size = self.pose1.shape[0]
        out_dof = self.acc1.dof()
        pose_dof = self.pose1.dof()
        dtype = self.pose1.dtype
        device = self.pose1.device

        Jerr_pose1 = torch.zeros(
            batch_size, 2 * out_dof, pose_dof, dtype=dtype, device=device
        )
        Jerr_acc1 = torch.zeros(
            batch_size, 2 * out_dof, out_dof, dtype=dtype, device=device
        )
        Jerr_pose2 = torch.zeros_like(Jerr_pose1)
        Jerr_acc2 = torch.zeros_like(Jerr_acc1)

        Jlocal: List[torch.Tensor] = []
        Jxy1: List[torch.Tensor] = []
        Jxy2: List[torch.Tensor] = []
        error = self._error_from_pose_diff(self._new_pose_diff(Jlocal, Jxy1, Jxy2))
        Jlocal[0] = Jlocal[0].matmul(Jxy1[0])
        Jlocal[1] = Jlocal[1].matmul(Jxy2[0])

        Jerr_pose1[:, :out_dof, :] = Jlocal[0]

        identity = torch.eye(out_dof, dtype=dtype, device=device).repeat(
            batch_size, 1, 1
        )

        Jerr_acc1[:, :out_dof, :] = -0.5 * self.dt.tensor.view(-1, 1, 1) ** 2 * identity
        Jerr_acc1[:, out_dof:, :] = -identity

        Jerr_pose2[:, :out_dof, :] = Jlocal[1]
        Jerr_acc2[:, out_dof:, :] = identity
        return [Jerr_pose1, Jerr_acc1, Jerr_pose2, Jerr_acc2], error

    def _copy_impl(self, new_name: Optional[str] = None) -> "_TripleIntegrator":
        return _TripleIntegrator(
            self.pose1.copy(),
            self.vel1.copy(),
            self.acc1.copy(),
            self.pose2.copy(),
            # self.vel2.copy(),
            self.acc2.copy(),
            self.dt.copy(),
            self.weight.copy(),
            name=new_name,
        )


class _QuadraticVectorCost(th.CostFunction):
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
                "_QuadraticVectorCost expects var and target of type Vector."
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
        return self.var.dof()

    def _copy_impl(self, new_name: Optional[str] = None) -> "_QuadraticVectorCost":
        return _QuadraticVectorCost(  # type: ignore
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


class _BothXYDifference(th.CostFunction):
    def __init__(
        self,
        var: SE2,
        target: SE2,
        cost_weight: CostWeight,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)
        if not isinstance(var, target.__class__):
            raise ValueError(
                "Variable for the Local inconsistent with the given target."
            )
        if not var.dof() == target.dof():
            raise ValueError(
                "Variable and target in the Local must have identical dof."
            )
        self.var = var
        self.target = target
        self.register_optim_vars(["var"])
        self.register_aux_vars(["target"])

    def error(self) -> torch.Tensor:
        error = self.target.xy(jacobians=None).local(
            self.var.xy(jacobians=None), jacobians=None
        )
        return error

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        Jlist: List[torch.Tensor] = []
        Jxy: List[torch.Tensor] = []
        error = self.target.xy(jacobians=None).local(
            self.var.xy(jacobians=Jxy), jacobians=Jlist
        )
        return [Jlist[1].matmul(Jxy[0])], error

    def dim(self) -> int:
        return 2

    def _copy_impl(self, new_name: Optional[str] = None) -> "_BothXYDifference":
        return _BothXYDifference(  # type: ignore
            self.var.copy(), self.target.copy(), self.weight.copy(), name=new_name
        )
