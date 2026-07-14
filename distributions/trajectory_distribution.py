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

import torch
from torch.distributions import Categorical

from distributions import (
    Gaussian,
    Laplace,
    GeneralizedGaussian,
    GaussianScaleMixture,
    VonMises,
)


class HeadingAlignedPositionDistribution:
    """A heading-aligned position distribution.

    The distribution models longitudinal and lateral position errors in the local
    coordinate frame defined by the predicted heading. The two local axes are
    represented by independent one-dimensional distributions from the selected family.

    Args:
        loc:
            Predicted global position means with shape ``[N, K, T, 2]``.
        raw_params:
            Raw distribution parameters for the local longitudinal and lateral axes.
        family:
            Position distribution family. Supported values are ``gaussian``,
            ``laplace``, ``generalized_gaussian``, and ``scale_mixture_*`` variants.
        heading:
            Predicted headings in radians with shape ``[N, K, T]``.
        min_stddev:
            Minimum standard deviation added for numerical stability.
        cumsum_uncertainty:
            If ``True``, accumulate uncertainty parameters over time.
    """

    def __init__(
        self,
        loc: torch.Tensor,
        raw_params: torch.Tensor,
        family: str,
        heading: torch.Tensor,
        min_stddev: float,
        cumsum_uncertainty: bool = False,
    ) -> None:
        self.loc = loc
        self.heading = heading

        cumsum_dim = 2 if cumsum_uncertainty else None

        zero = torch.zeros_like(raw_params[..., 0])

        if family == 'gaussian':
            self.long_axis = Gaussian(
                loc=zero,
                raw_stddev=raw_params[..., 0],
                min_stddev=min_stddev,
                cumsum_dim=cumsum_dim,
            )
            self.lat_axis = Gaussian(
                loc=zero,
                raw_stddev=raw_params[..., 1],
                min_stddev=min_stddev,
                cumsum_dim=cumsum_dim,
            )
        elif family == 'laplace':
            self.long_axis = Laplace(
                loc=zero,
                raw_stddev=raw_params[..., 0],
                min_stddev=min_stddev,
                cumsum_dim=cumsum_dim,
            )
            self.lat_axis = Laplace(
                loc=zero,
                raw_stddev=raw_params[..., 1],
                min_stddev=min_stddev,
                cumsum_dim=cumsum_dim,
            )
        elif family == 'generalized_gaussian':
            self.long_axis = GeneralizedGaussian(
                loc=zero,
                raw_stddev=raw_params[..., 0],
                raw_shape=raw_params[..., 2],
                min_stddev=min_stddev,
                cumsum_dim=cumsum_dim,
            )
            self.lat_axis = GeneralizedGaussian(
                loc=zero,
                raw_stddev=raw_params[..., 1],
                raw_shape=raw_params[..., 3],
                min_stddev=min_stddev,
                cumsum_dim=cumsum_dim,
            )
        elif family.startswith('scale_mixture'):
            _, _, components = family.split('_')
            components = int(components)
            self.long_axis = GaussianScaleMixture(
                loc=zero,
                raw_stddev=raw_params[..., 0 * components : 1 * components],
                logits=raw_params[..., 1 * components : 2 * components],
                min_stddev=min_stddev,
                cumsum_dim=cumsum_dim,
            )
            self.lat_axis = GaussianScaleMixture(
                loc=zero,
                raw_stddev=raw_params[..., 2 * components : 3 * components],
                logits=raw_params[..., 3 * components : 4 * components],
                min_stddev=min_stddev,
                cumsum_dim=cumsum_dim,
            )
        else:
            raise ValueError(f'Unknown family: {family}')

    def log_prob(self, pos_gt: torch.Tensor) -> torch.Tensor:
        """Evaluate the log probability of global ground-truth positions.

        Args:
            pos_gt:
                Ground-truth positions with shape ``[N, T, 2]``.

        Returns:
            Log probability values with shape ``[N, K, T]``.
        """
        r = pos_gt.unsqueeze(1) - self.loc
        c, s = torch.cos(self.heading), torch.sin(self.heading)
        e_long = r[..., 0] * c + r[..., 1] * s
        e_lat = -r[..., 0] * s + r[..., 1] * c

        return self.long_axis.log_prob(e_long) + self.lat_axis.log_prob(e_lat)

    def sample_t(self, t: int = -1, num_samples: int = 1) -> torch.Tensor:
        """Draw position samples for a single timestep.

        Args:
            t:
                Timestep index to sample.
            num_samples:
                Number of samples to draw per agent and mode.

        Returns:
            Global position samples with shape ``[N, K, num_samples, 2]``.
        """
        S = int(num_samples)

        e_long = self.long_axis.sample((S,))
        e_lat = self.lat_axis.sample((S,))
        e_long = e_long[..., t]
        e_lat = e_lat[..., t]

        mu_last = self.loc[..., t, :]

        head_last = self.heading[..., t]
        c = torch.cos(head_last).unsqueeze(0)
        s = torch.sin(head_last).unsqueeze(0)
        dx = e_long * c - e_lat * s
        dy = e_long * s + e_lat * c
        r_global = torch.stack([dx, dy], dim=-1)

        samples = r_global + mu_last.unsqueeze(0)
        return samples.permute(1, 2, 0, 3).contiguous()

    def __getitem__(self, index):
        N = self.loc.shape[0]
        device = self.loc.device
        idx = self.long_axis._normalize_index(index, N, device)

        new = object.__new__(type(self))
        new.loc = self.loc.index_select(0, idx)
        new.heading = self.heading.index_select(0, idx)
        new.long_axis = self.long_axis[idx]
        new.lat_axis = self.lat_axis[idx]

        return new


class TrajectoryDistribution:
    """A multi-modal trajectory distribution.

    Args:
        position:
            Position distribution for all agents, modes, and timesteps.
        heading:
            Heading distribution for all agents, modes, and timesteps.
        logits:
            Raw mode logits. Shape ``[N, K]`` represents trajectory-level mixture
            weights, while shape ``[N, K, T]`` represents timestep-level mixture
            weights.
    """

    def __init__(
        self,
        position: HeadingAlignedPositionDistribution,
        heading: VonMises,
        logits: torch.Tensor,
    ) -> None:
        self.position = position
        self.heading = heading
        self.logits = logits

    @property
    def pi(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=1)

    @property
    def position_mean(self) -> torch.Tensor:
        return self.position.loc

    @property
    def heading_mean(self) -> torch.Tensor:
        return self.heading.loc

    def means(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pi, self.position_mean, self.heading_mean

    def sample_t(
        self,
        t: int = -1,
        num_samples: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw position and heading samples for a single timestep.

        If ``per_mode`` is ``True``, samples are returned separately for every
        mode. Otherwise, one mode is sampled per agent and sample according to
        the mode probabilities.

        Args:
            t:
                Timestep index to sample.
            num_samples:
                Number of samples to draw per agent, and per mode if ``per_mode`` is
                ``True``.

        Returns:
            A tuple ``(pos, head)``, where ``pos`` has shape ``[N, num_samples, 2]`` and
            ``head`` has shape ``[N, num_samples]``.

        Raises:
            ValueError:
                If the mode-probability tensor has an unsupported number of dimensions.
        """
        pos = self.position.sample_t(t, num_samples)
        head = self.heading.loc[:, :, t]
        head = head.unsqueeze(2).repeat(1, 1, num_samples)

        N, _, S, _ = pos.shape
        if self.pi.dim() == 2:
            pi_last = self.pi
        elif self.pi.dim() == 3:
            pi_last = self.pi[:, :, t]
        else:
            raise ValueError(f'Unsupported pi shape: {self.pi.shape}')

        cat = Categorical(probs=pi_last)
        mode_idx = cat.sample((S,))
        n_idx = torch.arange(N, device=pos.device)[:, None]
        s_idx = torch.arange(S, device=pos.device)[None, :]
        pos = pos[n_idx, mode_idx.T, s_idx]
        head = head[n_idx, mode_idx.T, s_idx]
        return pos, head

    def log_prob(
        self,
        pos: torch.Tensor,
        head: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the mean log likelihood of ground-truth trajectories.

        Args:
            pos:
                Ground-truth positions with shape ``[N, T, 2]``.
            head:
                Ground-truth headings with shape ``[N, T]``.
            mask:
                Validity mask with shape ``[N, T]``.

        Returns:
            Mean log likelihood over agents.
        """
        lp_pos = self.position.log_prob(pos)
        lp_head = self.heading.log_prob(head.unsqueeze(1))
        log_p = lp_pos + lp_head

        mask = mask.to(log_p.dtype)
        N, K, T = log_p.shape
        logits = self.logits

        if logits.dim() == 2:
            mask_bc = mask.unsqueeze(1)
            log_p = log_p * mask_bc

            log_w = torch.log_softmax(logits, dim=1)
            lp_mode = log_p.sum(dim=-1)
            log_mix = torch.logsumexp(log_w + lp_mode, dim=1)
            return log_mix.mean()

        elif logits.dim() == 3:
            log_w = torch.log_softmax(logits, dim=1)

            log_mix_t = torch.logsumexp(log_w + log_p, dim=1)

            log_mix_t = log_mix_t * mask

            log_mix = log_mix_t.sum(dim=-1)
            return log_mix.mean()

        else:
            raise ValueError(f'Unsupported logits shape: {logits.shape}')

    def nll(
        self,
        pos: torch.Tensor,
        head: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the negative log likelihood of ground-truth trajectories.

        Args:
            pos:
                Ground-truth positions with shape ``[N, T, 2]``.
            head:
                Ground-truth headings with shape ``[N, T]``.
            mask:
                Validity mask with shape ``[N, T]``.

        Returns:
            Mean negative log likelihood over agents.
        """
        log_p_pos = self.position.log_prob(pos)
        log_p_head = self.heading.log_prob(head.unsqueeze(1))
        log_p = log_p_pos + log_p_head

        mask = mask.to(log_p.dtype)
        N, T = mask.shape
        logits = self.logits

        if logits.dim() == 2:
            mask_bc = mask.unsqueeze(1)

            lp = log_p * mask_bc

            log_w = torch.log_softmax(logits, dim=1)

            lp_mode = lp.sum(dim=-1)
            log_mix = torch.logsumexp(log_w + lp_mode, dim=1)
            return -log_mix.mean()

        elif logits.dim() == 3:
            log_w = torch.log_softmax(logits, dim=1)

            log_mix_t = torch.logsumexp(log_w + log_p, dim=1)

            log_mix_t = log_mix_t * mask
            loglik = log_mix_t.sum(dim=-1)

            return -loglik.mean()

        else:
            raise ValueError(f'Unsupported logits shape: {logits.shape}')

    def wta_nll(
        self,
        pos: torch.Tensor,
        head: torch.Tensor,
        mask: torch.Tensor,
        best_mode: torch.Tensor | None = None,
        return_best_mode: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute winner-takes-all negative log likelihood.

        The best mode is selected by the smallest masked position error unless a
        precomputed ``best_mode`` tensor is provided. The returned loss combines
        the regression NLL of the selected mode with a mode-weight loss.

        Args:
            pos:
                Ground-truth positions with shape ``[N, T, 2]``.
            head:
                Ground-truth headings with shape ``[N, T]``.
            mask:
                Validity mask with shape ``[N, T]``.
            best_mode:
                Optional precomputed best mode index for each agent, with shape
                ``[N]``.
            return_best_mode:
                If ``True``, also return the selected best mode indices.

        Returns:
            The scalar WTA NLL loss. If ``return_best_mode`` is ``True``, returns
            ``(loss, best_mode)``.
        """
        if best_mode is None:
            mu = self.position.loc
            gt_pos = pos.unsqueeze(1)
            per_t = torch.norm(mu - gt_pos, p=2, dim=-1)
            per_t = per_t * mask.unsqueeze(1)
            d = per_t.sum(dim=-1)
            best_mode = d.argmin(dim=1)

        lp_pos = self.position.log_prob(pos)
        lp_head = self.heading.log_prob(head.unsqueeze(1))
        lp = lp_pos + lp_head

        s = lp.detach()[..., -1]

        agent_i = torch.arange(best_mode.shape[0], device=best_mode.device)
        reg_nll = -lp[agent_i, best_mode]

        valid_counts = mask.sum(dim=0).clamp_min(1.0)
        reg_nll = (reg_nll * mask).sum(dim=0) / valid_counts
        reg_nll = reg_nll.mean()

        log_w = torch.log_softmax(self.logits, dim=1)
        weights_nll = -torch.logsumexp(log_w + s, dim=1)

        weights_nll = (weights_nll * mask[:, -1]).sum(dim=0) / valid_counts[-1]

        total = reg_nll + weights_nll

        if return_best_mode:
            return total, best_mode
        else:
            return total

    def __getitem__(self, index):
        N = self.logits.shape[0]
        device = self.logits.device
        idx = self.heading._normalize_index(index, N, device)

        pos = self.position[index]
        head = self.heading[index]
        logits = self.logits.index_select(0, idx)

        return TrajectoryDistribution(position=pos, heading=head, logits=logits)


class DistributionFactory:
    """A trajectory distribution factory for creating trajectory distributions.

    Args:
        pos_family:
            Position distribution family used for heading-aligned position errors.
        max_concentration:
            Maximum concentration used by the Von Mises heading distribution.
        min_stddev:
            Minimum position standard deviation used for numerical stability.
    """

    def __init__(
        self,
        pos_family: str,
        max_concentration: int = 50,
        min_stddev: float = 0.1
        * math.sqrt(2.0),  # identical to min scale in DONUT / QCNet
    ) -> None:
        self.pos_family = pos_family
        self.max_concentration = max_concentration
        self.min_stddev = min_stddev

    def build(
        self,
        pos: torch.Tensor,
        pos_params: torch.Tensor,
        head: torch.Tensor,
        head_params: torch.Tensor,
        logits: torch.Tensor,
        *,
        cumsum_uncertainty: bool | None = None,
    ) -> TrajectoryDistribution:
        """Build a trajectory distribution from network outputs.

        Args:
            pos:
                Predicted position means with shape ``[N, K, T, 2]``.
            pos_params:
                Raw position distribution parameters.
            head:
                Predicted heading means with shape ``[N, K, T]``.
            head_params:
                Raw heading distribution parameters with shape ``[N, K, T]``.
            logits:
                Raw mode logits with shape ``[N, K]`` or ``[N, K, T]``.
            cumsum_uncertainty:
                If ``True``, accumulate uncertainty parameters over time. If ``None`` or
                ``False``, use per-timestep uncertainty parameters.

        Returns:
            A trajectory distribution combining heading-aligned position and Von Mises
            heading distributions.
        """

        head_dist = VonMises(
            loc=head,
            raw_inv_concentration=head_params,
            max_concentration=self.max_concentration,
            cumsum_dim=2 if cumsum_uncertainty else None,
        )

        pos_dist = HeadingAlignedPositionDistribution(
            loc=pos,
            raw_params=pos_params,
            family=self.pos_family,
            heading=head.detach(),
            min_stddev=self.min_stddev,
            cumsum_uncertainty=cumsum_uncertainty,
        )

        return TrajectoryDistribution(
            position=pos_dist, heading=head_dist, logits=logits
        )
