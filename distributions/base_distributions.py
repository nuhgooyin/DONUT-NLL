# Copyright (c) 2026, Markus Knoche. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch.distributions import Gamma

LOG_2PI = math.log(2.0 * math.pi)


def _positive(
    raw: torch.Tensor,
    *,
    cumsum_dim: int | None = None,
    elu_eps: float = 1e-4,
) -> torch.Tensor:
    out = (1.0 + F.elu(raw)).clamp(min=elu_eps)
    if cumsum_dim is not None:
        out = torch.cumsum(out, dim=int(cumsum_dim))
    return out


class Base1DDistribution(ABC):
    @abstractmethod
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate the element-wise log probability density."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, sample_shape: torch.Size | tuple[int, ...] = ()) -> torch.Tensor:
        """Draw samples from the distribution."""
        raise NotImplementedError

    def _normalize_index(
        self,
        index: int | slice | list[int] | tuple[int, ...] | torch.Tensor,
        N: int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(index, slice):
            return torch.arange(N, device=device)[index]

        if isinstance(index, (list, tuple)):
            idx = torch.tensor(index, dtype=torch.long, device=device)
            idx = torch.where(idx < 0, idx + N, idx)
            return idx

        if torch.is_tensor(index):
            if index.dtype == torch.bool:
                mask = index.to(device=device)
                if mask.dim() == 0:
                    mask = mask.view(1)
                else:
                    mask = mask.flatten()
                if mask.numel() != N:
                    raise IndexError(
                        f'Boolean mask must have length {N}, got {mask.numel()}'
                    )
                return mask.nonzero(as_tuple=False).squeeze(-1)
            else:
                idx = index.to(device=device, dtype=torch.long).flatten()
                idx = torch.where(idx < 0, idx + N, idx)
                return idx

        if isinstance(index, int):
            i = index if index >= 0 else N + index
            if not (0 <= i < N):
                raise IndexError(f'Index out of range: {index} for batch size {N}')
            return torch.tensor([i], dtype=torch.long, device=device)

        raise TypeError(f'Unsupported index type: {type(index)}')

    def _index_select_batch(self, idx: torch.Tensor) -> 'Base1DDistribution':
        new = object.__new__(self.__class__)
        batch_n = (
            self.loc.shape[0]
            if torch.is_tensor(self.loc) and self.loc.dim() >= 1
            else None
        )

        for name, val in self.__dict__.items():
            if (
                torch.is_tensor(val)
                and batch_n is not None
                and val.dim() >= 1
                and val.shape[0] == batch_n
            ):
                setattr(new, name, val.index_select(0, idx))
            else:
                setattr(new, name, val)
        return new

    def __getitem__(
        self,
        index: int | slice | list[int] | tuple[int, ...] | torch.Tensor,
    ) -> 'Base1DDistribution':
        N = self.loc.shape[0]
        device = self.loc.device
        idx = self._normalize_index(index, N, device)
        return self._index_select_batch(idx)


class Gaussian(Base1DDistribution):
    """A univariate Gaussian distribution.

    Args:
        loc:
            Mean of the distribution.
        raw_stddev:
            Raw unconstrained standard-deviation parameters.
        min_stddev:
            Minimum standard deviation for stability.
        cumsum_dim:
            Optional dimension along which standard deviations are accumulated.
    """

    def __init__(
        self,
        loc: torch.Tensor,
        raw_stddev: torch.Tensor,
        min_stddev: float = 0.01,
        cumsum_dim: int | None = None,
    ) -> None:
        self.loc = loc
        self.stddev = min_stddev + _positive(raw_stddev, cumsum_dim=cumsum_dim)
        self.scale = self.stddev

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate the Gaussian log probability density."""
        z = (value - self.loc) / self.stddev
        return -0.5 * (z * z + 2.0 * torch.log(self.stddev) + LOG_2PI)

    def sample(self, sample_shape: torch.Size | tuple[int, ...] = ()) -> torch.Tensor:
        """Draw samples from the Gaussian distribution."""
        sample_shape = torch.Size(sample_shape)
        eps = torch.randn(
            sample_shape + self.loc.shape,
            device=self.loc.device,
            dtype=self.loc.dtype,
        )
        return self.loc + eps * self.stddev


class Laplace(Base1DDistribution):
    """A univariate Laplace distribution.

    Args:
        loc:
            Location parameter of the distribution.
        raw_stddev:
            Raw unconstrained standard-deviation parameters.
        min_stddev:
            Minimum standard deviation for stability.
        cumsum_dim:
            Optional dimension along which standard deviations are accumulated.
    """

    def __init__(
        self,
        loc: torch.Tensor,
        raw_stddev: torch.Tensor,
        min_stddev: float = 0.01,
        cumsum_dim: int | None = None,
    ) -> None:
        self.loc = loc
        self.stddev = min_stddev + _positive(raw_stddev, cumsum_dim=cumsum_dim)
        self.scale = self.stddev / math.sqrt(2.0)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate the Laplace log probability density."""
        return -torch.abs(value - self.loc) / self.scale - (
            math.log(2.0) + torch.log(self.scale)
        )

    def sample(self, sample_shape: torch.Size | tuple[int, ...] = ()) -> torch.Tensor:
        """Draw samples from the Laplace distribution."""
        sample_shape = torch.Size(sample_shape)
        u = torch.rand(
            sample_shape + self.loc.shape,
            device=self.loc.device,
            dtype=self.loc.dtype,
        )
        sgn = torch.sign(u - 0.5)
        v = (1.0 - 2.0 * torch.abs(u - 0.5)).clamp_min(1e-6)
        return self.loc - self.scale * sgn * torch.log(v)


class GeneralizedGaussian(Base1DDistribution):
    """A univariate generalized Gaussian distribution.

    Args:
        loc:
            Location parameter of the distribution.
        raw_stddev:
            Raw unconstrained standard-deviation parameters.
        raw_shape:
            Raw unconstrained shape parameters.
        min_stddev:
            Minimum standard deviation for stability.
        min_shape:
            Lower bound for the shape parameter.
        max_shape:
            Upper bound for the shape parameter.
        cumsum_dim:
            Optional dimension along which standard deviations are accumulated.
    """

    def __init__(
        self,
        loc: torch.Tensor,
        raw_stddev: torch.Tensor,
        raw_shape: torch.Tensor,
        min_stddev: float = 0.01,
        min_shape: float = 0.5,
        max_shape: float = 4.0,
        cumsum_dim: int | None = None,
    ) -> None:
        if not (max_shape > min_shape):
            raise ValueError(
                f'max_shape must be > min_shape, got {min_shape}, {max_shape}'
            )
        self.loc = loc
        self.stddev = min_stddev + _positive(raw_stddev, cumsum_dim=cumsum_dim)
        self.shape = min_shape + (max_shape - min_shape) * torch.sigmoid(raw_shape)

        p = self.shape
        lg1 = torch.lgamma(1.0 / p)
        lg3 = torch.lgamma(3.0 / p)
        self.scale = self.stddev * torch.exp(0.5 * (lg1 - lg3))

    def log_prob(self, value: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """Evaluate the generalized Gaussian log probability density."""
        z = torch.abs(value - self.loc) / self.scale
        z = z.clamp_min(eps)
        p = self.shape
        log_norm = (
            torch.log(p) - math.log(2.0) - torch.log(self.scale) - torch.lgamma(1.0 / p)
        )
        return log_norm - torch.pow(z, p)

    def sample(self, sample_shape: torch.Size | tuple[int, ...] = ()) -> torch.Tensor:
        """Draw samples from the generalized Gaussian distribution."""
        sample_shape = torch.Size(sample_shape)
        p = self.shape
        k = (1.0 / p).to(dtype=self.loc.dtype, device=self.loc.device)
        gamma = Gamma(concentration=k, rate=torch.ones_like(k))
        T = gamma.sample(sample_shape=sample_shape)
        R = self.scale * torch.pow(T, 1.0 / p)

        sign = torch.randint(
            0, 2, size=R.shape, device=self.loc.device, dtype=torch.int8
        ).to(self.loc.dtype)
        sign = sign * 2.0 - 1.0
        return self.loc + sign * R


class VonMises(Base1DDistribution):
    """A univariate Von Mises distribution.

    Args:
        loc:
            Circular mean direction in radians.
        raw_inv_concentration:
            Raw unconstrained inverse-concentration parameters.
        max_concentration:
            Maximum concentration value for stability.
        cumsum_dim:
            Optional dimension along which inverse-concentration values are
            accumulated.
    """

    def __init__(
        self,
        loc: torch.Tensor,
        raw_inv_concentration: torch.Tensor,
        max_concentration: float = 50.0,
        cumsum_dim: int | None = None,
    ) -> None:
        self.loc = loc
        min_inv = 1.0 / float(max_concentration)
        inv_kappa = min_inv + _positive(raw_inv_concentration, cumsum_dim=cumsum_dim)
        self.concentration = 1.0 / inv_kappa

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate the Von Mises log probability density."""
        kappa = self.concentration
        log_i0 = torch.log(torch.special.i0e(kappa)) + torch.abs(kappa)
        return kappa * torch.cos(value - self.loc) - LOG_2PI - log_i0

    def sample(self, sample_shape: torch.Size | tuple[int, ...] = ()) -> torch.Tensor:
        """Draw samples from the Von Mises distribution."""
        sample_shape = torch.Size(sample_shape)
        vm = torch.distributions.von_mises.VonMises(
            loc=self.loc,
            concentration=self.concentration,
        )
        return vm.sample(sample_shape)


class GaussianScaleMixture(Base1DDistribution):
    """A univariate Gaussian scale mixture.

    The mixture shares a single location tensor and predicts one standard deviation
    and one mixture logit per component.

    Args:
        loc:
            Shared Gaussian location tensor.
        raw_stddev:
            Raw unconstrained standard-deviation parameters with one final component
            dimension.
        logits:
            Mixture logits with the same shape as ``raw_stddev``.
        min_stddev:
            Minimum standard deviation for stability.
        cumsum_dim:
            Optional dimension along which standard deviations are accumulated.
    """

    def __init__(
        self,
        loc: torch.Tensor,
        raw_stddev: torch.Tensor,
        logits: torch.Tensor,
        min_stddev: float = 0.01,
        cumsum_dim: int | None = None,
    ) -> None:
        if raw_stddev.shape != logits.shape:
            raise ValueError(
                f'raw_stddev and logits must have same shape, got '
                f'{raw_stddev.shape} vs {logits.shape}'
            )
        if raw_stddev.shape[:-1] != loc.shape:
            raise ValueError(
                f'Leading dims of raw_stddev must match loc.shape. '
                f'Got loc.shape={loc.shape}, raw_stddev.shape={raw_stddev.shape}'
            )

        self.loc = loc
        self.stddev = min_stddev + _positive(raw_stddev, cumsum_dim=cumsum_dim)
        self.scale = self.stddev

        self.logits = logits
        self.num_components = raw_stddev.shape[-1]

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate the Gaussian scale-mixture log probability density."""
        x = value - self.loc

        z = x.unsqueeze(-1) / self.stddev
        log_comp = -0.5 * (z * z + 2.0 * torch.log(self.stddev) + LOG_2PI)

        log_weights = F.log_softmax(self.logits, dim=-1)

        return torch.logsumexp(log_weights + log_comp, dim=-1)

    def sample(self, sample_shape: torch.Size | tuple[int, ...] = ()) -> torch.Tensor:
        """Draw samples from the Gaussian scale mixture."""
        sample_shape = torch.Size(sample_shape)
        device = self.loc.device
        dtype = self.loc.dtype

        batch_shape = self.loc.shape
        K = self.num_components

        probs = torch.softmax(self.logits, dim=-1)

        u = torch.rand(sample_shape + batch_shape, device=device, dtype=dtype)
        cdf = probs.cumsum(dim=-1)
        cdf_expanded = cdf.expand(sample_shape + batch_shape + (K,))
        u_expanded = u.unsqueeze(-1)

        comp_idx = (u_expanded > cdf_expanded).sum(dim=-1)
        comp_idx = comp_idx.clamp_max(K - 1)

        eps = torch.randn(
            sample_shape + batch_shape + (K,),
            device=device,
            dtype=dtype,
        )
        samples_all = self.loc.unsqueeze(-1) + eps * self.stddev

        comp_idx_expanded = comp_idx.unsqueeze(-1)
        samples = samples_all.gather(dim=-1, index=comp_idx_expanded).squeeze(-1)

        return samples
