from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    """
    A simple ZeRO-1-style optimizer-state-sharding wrapper.

    Core idea
    ---------
    Every rank still owns a full copy of the model parameters, but each rank's
    *local optimizer* only owns a subset of those parameters.

    Therefore:
      - parameters: replicated on every rank
      - gradients: replicated on every rank
      - optimizer state: sharded across ranks

    After the local optimizer updates the parameters owned by this rank, the
    updated parameters are broadcast from their owner rank to every other rank
    so that all model replicas become identical again.

    Notes
    -----
    1. This is intentionally a learning implementation, not a production one.
       It broadcasts parameters one tensor at a time after every optimizer step.
       A production implementation would normally bucket/fuse communication.

    2. This optimizer assumes gradients are already correct/synchronized before
       ``step()``. In a normal DDP setup that synchronization is performed by DDP.

    3. Parameter ownership is assigned round-robin:
           owner(parameter_i) = i % world_size
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        optimizer_cls: type[torch.optim.Optimizer],
        **kwargs: Any,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "ShardedOptimizer requires an initialized torch.distributed "
                "process group."
            )

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls

        # Materialize the iterable exactly once. model.parameters() is usually
        # a generator, so iterating over it twice would otherwise lose data.
        raw_params = list(params)

        if len(raw_params) == 0:
            raise ValueError("ShardedOptimizer received an empty parameter list.")

        # Normalize the input into standard PyTorch optimizer parameter groups.
        #
        # Supported forms:
        #   model.parameters()
        #
        # or:
        #   [
        #       {"params": layer1.parameters(), "lr": 1e-3},
        #       {"params": layer2.parameters(), "lr": 1e-4},
        #   ]
        if isinstance(raw_params[0], dict):
            full_param_groups: list[dict[str, Any]] = []

            for group in raw_params:
                copied_group = dict(group)
                copied_group["params"] = list(copied_group["params"])
                full_param_groups.append(copied_group)
        else:
            full_param_groups = [{"params": raw_params}]

        # Flatten the parameters in a deterministic order.
        self._all_params: list[torch.Tensor] = []
        self._owners: list[int] = []

        # local_param_groups has the same hyperparameters as the original
        # groups, but each rank keeps only the parameters it owns.
        local_param_groups: list[dict[str, Any]] = []

        global_param_index = 0

        for group in full_param_groups:
            local_group = {
                key: value
                for key, value in group.items()
                if key != "params"
            }
            local_group_params: list[torch.Tensor] = []

            for param in group["params"]:
                owner = global_param_index % self.world_size

                self._all_params.append(param)
                self._owners.append(owner)

                if owner == self.rank:
                    local_group_params.append(param)

                global_param_index += 1

            # torch.optim optimizers do not like completely empty groups.
            if local_group_params:
                local_group["params"] = local_group_params
                local_param_groups.append(local_group)

        if not local_param_groups:
            raise RuntimeError(
                f"Rank {self.rank} owns no parameters. "
                "Use fewer ranks or a model with more parameter tensors."
            )

        # Initialize the Optimizer base class so this object behaves like a
        # normal torch.optim.Optimizer.
        #
        # We immediately replace param_groups/state below with those of the
        # local optimizer because only locally-owned parameters should have
        # optimizer state.
        super().__init__(self._all_params, defaults={})

        # This is the important ZeRO-1 step:
        # only parameters owned by this rank are passed to the real optimizer.
        self.local_optimizer = optimizer_cls(local_param_groups, **kwargs)

        # Expose the local optimizer's state through the usual Optimizer API.
        # Thus len(self.state), state_dict(), schedulers, etc. reflect only the
        # optimizer state actually stored on this rank.
        self.param_groups = self.local_optimizer.param_groups
        self.state = self.local_optimizer.state
        self.defaults = self.local_optimizer.defaults

    @torch.no_grad()
    def step(self, closure=None):
        """
        Update this rank's parameter shard, then synchronize updated parameters.

        Flow:
            1. Local optimizer updates only parameters owned by this rank.
            2. For every model parameter, its owner broadcasts the updated
               tensor to all other ranks.
            3. Every rank once again holds the same full model.

        Gradients are assumed to have already been synchronized before this call.
        """
        if closure is None:
            loss = self.local_optimizer.step()
        else:
            loss = self.local_optimizer.step(closure)

        # Every rank must execute broadcasts in the same order.
        #
        # Example with world_size=2:
        #   p0 owner=rank0 -> rank0 broadcasts p0
        #   p1 owner=rank1 -> rank1 broadcasts p1
        #   p2 owner=rank0 -> rank0 broadcasts p2
        #   ...
        for param, owner in zip(self._all_params, self._owners, strict=True):
            dist.broadcast(param.data, src=owner)

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Clear gradients for the *full replicated model*.

        We cannot simply call local_optimizer.zero_grad(), because that would
        clear gradients only for parameters owned by this rank, while every
        rank still has gradients for the full model.
        """
        for param in self._all_params:
            if param.grad is None:
                continue

            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()

    def local_parameter_count(self) -> int:
        """Number of scalar parameters whose optimizer state this rank owns."""
        return sum(
            param.numel()
            for group in self.local_optimizer.param_groups
            for param in group["params"]
        )

    def total_parameter_count(self) -> int:
        """Number of scalar parameters in the full replicated model."""
        return sum(param.numel() for param in self._all_params)


def get_sharded_optimizer(
    params,
    optimizer_cls: type[torch.optim.Optimizer],
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Adapter matching the CS336 Assignment 2 interface.
    """
    return ShardedOptimizer(
        params=params,
        optimizer_cls=optimizer_cls,
        **kwargs,
    )
