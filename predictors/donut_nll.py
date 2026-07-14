# Copyright (c) 2025-2026, Markus Knoche. All rights reserved.
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
from copy import deepcopy

import pytorch_lightning as pl
import torch
import torch.nn as nn

from distributions import DistributionFactory
from layers import MLPLayer
from modules import QCNetMapEncoder, DonutNet


class DonutNLL(pl.LightningModule):
    """
    DONUT-NLL extends DONUT by allowing various output distributions.

    Args:
        dataset:
            Dataset identifier passed to the map encoder.
        t_per_tok:
            Number of timesteps represented by one trajectory token.
        t_hist:
            Number of historical timesteps provided as input.
        t_pred:
            Number of future timesteps to predict.
        num_modes:
            Number of trajectory hypotheses generated per agent.
        refine:
            If ``True``, use an additional refinement decoder after the proposer.
        overpredict:
            If ``True``, also produce auxiliary over-prediction outputs.
        hidden_dim:
            Dimensionality of latent feature vectors.
        edge_limit:
            Relative threshold for pruning attention edges in large scenes.
        map_enc_radius:
            Local radius, in meters, considered by the map encoder.
        map_enc_layers:
            Number of map encoder layers.
        dec_attn_order:
            Decoder attention order. Supported attention types are temporal
            ``t``, social ``s``, road ``r``, and mode ``m`` attention.
        dec_attn_repetitions:
            Number of times the decoder attention order is repeated.
        dec_radius_r:
            Radius, in meters, used for road attention.
        dec_radius_s:
            Radius, in meters, used for social attention.
        position_distribution:
            Position distribution family used for trajectory likelihoods.
            Supported values include ``gaussian``, ``laplace``,
            ``generalized_gaussian``, and ``scale_mixture_*`` variants.
        loss_type:
            Training loss type. Must be one of ``wta``, ``traj_nll``, or
            ``step_nll``.
        target_loss_only:
            If ``True``, compute the loss only for target agents.
        lr:
            Initial learning rate for AdamW.
        weight_decay:
            Weight decay used for regularized parameters.
        decay_epochs:
            Number of epochs used by the learning-rate decay schedule.
    """

    def __init__(
        self,
        dataset,
        t_per_tok,
        t_hist,
        t_pred,
        num_modes,
        refine,
        overpredict,
        hidden_dim,
        edge_limit,
        map_enc_radius,
        map_enc_layers,
        dec_attn_order,
        dec_attn_repetitions,
        dec_radius_r,
        dec_radius_s,
        position_distribution,
        loss_type,
        target_loss_only,
        lr,
        weight_decay,
        decay_epochs,
        **_,
    ):
        super().__init__()
        self.save_hyperparameters()

        if loss_type not in {'wta', 'traj_nll', 'step_nll'}:
            raise ValueError(f'loss_type {loss_type} not supported.')

        level_config = {
            'order': dec_attn_order,
            'repetitions': dec_attn_repetitions,
            'radius_r': dec_radius_r,
            'radius_s': dec_radius_s,
            'edge_limit': edge_limit,
        }

        mode_config = {
            'num_modes': num_modes,
            'pred_steps': t_pred // t_per_tok,
        }

        self.t_per_tok = t_per_tok
        self.t_hist = t_hist
        self.t_pred = t_pred
        self.num_modes = num_modes

        self.refine = refine
        self.overpredict = overpredict

        self.map_encoder = QCNetMapEncoder(
            dataset=dataset,
            hidden_dim=hidden_dim,
            num_historical_steps=t_hist,
            pl2pl_radius=map_enc_radius,
            num_layers=map_enc_layers,
        )

        if position_distribution in {'gaussian', 'laplace'}:
            pos_params_per_dim = 1
        elif position_distribution == 'generalized_gaussian':
            pos_params_per_dim = 2
        elif position_distribution.startswith('scale_mixture'):
            _, _, components = position_distribution.split('_')
            components = int(components)
            pos_params_per_dim = 2 * components
        else:
            raise ValueError(f'position_distribution {position_distribution} not known')
        self.proposer = DonutNet(
            hidden_dim=hidden_dim,
            t_per_tok=t_per_tok,
            type_count=10,
            pos_params_per_dim=pos_params_per_dim,
            over_predict=self.overpredict,
            has_feature_input=False,
            has_feature_output=self.refine,
            **level_config,
            mode_config=mode_config,
        )
        logits_per_ts = t_per_tok if loss_type == 'step_nll' else 1
        self.to_logits_proposer = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=logits_per_ts
        )
        if self.refine:
            self.refiner = DonutNet(
                hidden_dim=hidden_dim,
                t_per_tok=t_per_tok,
                type_count=10,
                pos_params_per_dim=pos_params_per_dim,
                over_predict=self.overpredict,
                has_feature_input=True,
                has_feature_output=False,
                **level_config,
                mode_config=mode_config,
            )
            self.to_logits_refiner = MLPLayer(
                input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=logits_per_ts
            )

        self.distribution_factory = DistributionFactory(
            pos_family=position_distribution
        )
        self.loss_type = loss_type
        self.target_loss_only = target_loss_only

        self.lr = lr
        self.weight_decay = weight_decay
        self.decay_epochs = decay_epochs

    def forward(self, inp):
        """Run a forward pass and return trajectory distributions.

        The method encodes map context and observed agent history, then autoregressively
        predicts future trajectory tokens. The proposer decoder always produces
        trajectory distributions. If refinement is enabled, a second set of refined
        trajectory distributions is produced as well.

        Args:
            inp:
                Batched heterogeneous scene data from the dataloader. The input is
                expected to contain agent history fields such as position, heading,
                valid mask, batch index, and type, as well as map polygon fields used by
                the map encoder.

        Returns:
            A nested list of trajectory distribution objects. The outer list contains
            one entry for the proposer and, if ``refine`` is enabled, one additional
            entry for the refiner. Each inner list contains the main trajectory
            distribution and, if ``overpredict`` is enabled, an auxiliary
            over-prediction distribution.

            Conceptually, the structure is::

                [
                    [proposer_distribution, proposer_overprediction?],
                    [refiner_distribution, refiner_overprediction?],
                ]

            where the refiner entry is present only if ``refine`` is ``True``, and the
            over-prediction entries are present only if ``overpredict`` is ``True``.
        """
        map_enc = self.map_encoder(inp)

        start = self.t_hist % self.t_per_tok  # drop history to match t_per_tok
        x_pos = inp['agent']['position'][:, start : self.t_hist, :2].contiguous()
        x_head = inp['agent']['heading'][:, start : self.t_hist].contiguous()
        x_mask = inp['agent']['valid_mask'][:, start : self.t_hist].contiguous()
        x_batch = inp['agent']['batch']
        x_type = inp['agent']['type'].int()

        pl_x = map_enc['x_pl'][:, 0]
        pl_pos = inp['map_polygon']['position'][:, :2].contiguous()
        pl_head = inp['map_polygon']['orientation'].contiguous()
        pl_batch = inp['map_polygon']['batch']

        num_agents = x_pos.shape[0]
        num_hist = self.t_hist // self.t_per_tok
        num_pred = self.t_pred // self.t_per_tok

        x_pos = x_pos.reshape(num_agents, 1, num_hist, self.t_per_tok, 2)
        x_head = x_head.reshape(num_agents, 1, num_hist, self.t_per_tok)
        x_mask = x_mask.reshape(num_agents, 1, num_hist, self.t_per_tok)
        x_type = x_type.reshape(num_agents, 1, 1)

        # encode history
        xs, proposer_past = self.proposer(
            x_pos[:, :, :-1],
            x_head[:, :, :-1],
            x_type,
            x_mask[:, :, :-1],
            x_batch,
            0,
            pl_x,
            pl_pos,
            pl_head,
            pl_batch,
        )

        if self.refine:
            xs, refiner_past = self.refiner(
                x_pos[:, :, 1:],
                x_head[:, :, 1:],
                x_type,
                x_mask[:, :, 1:],
                x_batch,
                0,
                pl_x,
                pl_pos,
                pl_head,
                pl_batch,
                proposed=(xs['pos'], xs['head']),
                feature_input=xs['feats'],
            )

        # make multi-modal
        x_pos = x_pos[:, :, -1:].repeat_interleave(self.num_modes, 1)
        x_head = x_head[:, :, -1:].repeat_interleave(self.num_modes, 1)
        x_mask = x_mask[:, :, -1:].repeat_interleave(self.num_modes, 1)

        pred_proposer = {'pos': [], 'pos_params': [], 'head': [], 'head_params': []}
        logits_proposer = []
        if self.refine:
            pred_refiner = {'pos': [], 'pos_params': [], 'head': [], 'head_params': []}
            logits_refiner = []

        for pred_step in range(1, num_pred + 1):
            xs, proposer_past = self.proposer(
                x_pos,
                x_head,
                x_type,
                x_mask,
                x_batch,
                pred_step,
                pl_x,
                pl_pos,
                pl_head,
                pl_batch,
                past=proposer_past,
            )
            pred_proposer['pos'].append(xs['pos'])
            pred_proposer['pos_params'].append(xs['pos_params'])
            pred_proposer['head'].append(xs['head'])
            pred_proposer['head_params'].append(xs['head_params'])
            x_mask = torch.ones_like(x_mask)

            x_pos = xs['pos'][:, :, :, : self.t_per_tok].detach()
            x_pos = x_pos.reshape(num_agents, self.num_modes, 1, self.t_per_tok, 2)
            x_head = xs['head'][:, :, :, : self.t_per_tok].detach()
            x_head = x_head.reshape(num_agents, self.num_modes, 1, self.t_per_tok)

            logits_proposer.append(xs['logits'])

            if self.refine:
                xs, refiner_past = self.refiner(
                    x_pos,
                    x_head,
                    x_type,
                    x_mask,
                    x_batch,
                    pred_step,
                    pl_x,
                    pl_pos,
                    pl_head,
                    pl_batch,
                    proposed=(xs['pos'], xs['head']),
                    past=refiner_past,
                    feature_input=xs['feats'],
                )
                pred_refiner['pos'].append(xs['pos'])
                pred_refiner['pos_params'].append(xs['pos_params'])
                pred_refiner['head'].append(xs['head'])
                pred_refiner['head_params'].append(xs['head_params'])

                x_mask = torch.ones_like(x_mask)
                x_pos = xs['pos'][:, :, :, : self.t_per_tok].detach()
                x_pos = x_pos.reshape(num_agents, self.num_modes, 1, self.t_per_tok, 2)
                x_head = xs['head'][:, :, :, : self.t_per_tok].detach()
                x_head = x_head.reshape(num_agents, self.num_modes, 1, self.t_per_tok)
                logits_refiner.append(xs['logits'])

        if self.loss_type == 'step_nll':
            logits_proposer = torch.cat(logits_proposer, dim=2)
            logits_proposer = self.to_logits_proposer(logits_proposer)
            logits_proposer = logits_proposer.reshape(-1, self.num_modes, self.t_pred)
            if self.refine:
                logits_refiner = torch.cat(logits_refiner, dim=2)
                logits_refiner = self.to_logits_refiner(logits_refiner)
                logits_refiner = logits_refiner.reshape(-1, self.num_modes, self.t_pred)
        else:
            logits_proposer = self.to_logits_proposer(logits_proposer[-1])
            logits_proposer = logits_proposer.squeeze(-1).squeeze(-1)
            if self.refine:
                logits_refiner = self.to_logits_refiner(logits_refiner[-1])
                logits_refiner = logits_refiner.squeeze(-1).squeeze(-1)

        if self.overpredict:
            pred_proposer_overp = {}
        for key, val in pred_proposer.items():
            val = torch.cat(val, dim=2)
            if self.overpredict:
                val, val_overp = torch.split(val, self.t_per_tok, 3)
                pred_proposer_overp[key] = val_overp[:, :, :-1].flatten(2, 3)
            pred_proposer[key] = val.flatten(2, 3)
        if self.refine:
            if self.overpredict:
                pred_refiner_overp = {}
            for key, val in pred_refiner.items():
                val = torch.cat(val, dim=2)
                if self.overpredict:
                    val, val_overp = torch.split(val, self.t_per_tok, 3)
                    pred_refiner_overp[key] = val_overp[:, :, :-1].flatten(2, 3)
                pred_refiner[key] = val.flatten(2, 3)

        traj_distrs = []
        traj_distrs_proposer = []
        traj_distr_proposer = self.distribution_factory.build(
            **pred_proposer,
            logits=logits_proposer,
            cumsum_uncertainty=True,
        )
        traj_distrs_proposer.append(traj_distr_proposer)
        if self.overpredict:
            if self.loss_type == 'step_nll':
                tpt = self.t_per_tok
                logits_proposer = logits_proposer[
                    :, :, tpt - 1 : -1 : tpt
                ].repeat_interleave(tpt, dim=2)
            traj_distr_proposer_overp = self.distribution_factory.build(
                **pred_proposer_overp,
                logits=logits_proposer.detach(),
                cumsum_uncertainty=False,
            )
            traj_distrs_proposer.append(traj_distr_proposer_overp)
        traj_distrs.append(traj_distrs_proposer)
        if self.refine:
            traj_distrs_refiner = []
            traj_distr_refiner = self.distribution_factory.build(
                **pred_refiner,
                logits=logits_refiner,
                cumsum_uncertainty=True,
            )
            traj_distrs_refiner.append(traj_distr_refiner)
            if self.overpredict:
                if self.loss_type == 'step_nll':
                    tpt = self.t_per_tok
                    logits_refiner = logits_refiner[
                        :, :, tpt - 1 : -1 : tpt
                    ].repeat_interleave(tpt, dim=2)
                traj_distr_refiner_overp = self.distribution_factory.build(
                    **pred_refiner_overp,
                    logits=logits_refiner.detach(),
                    cumsum_uncertainty=False,
                )
                traj_distrs_refiner.append(traj_distr_refiner_overp)
            traj_distrs.append(traj_distrs_refiner)

        return traj_distrs

    def compute_loss(self, data, traj_distrs):

        gt_pos = data['agent']['position'][:, -self.t_pred :, :2]
        gt_head = data['agent']['heading'][:, -self.t_pred :]
        pred_mask = data['agent']['predict_mask'][:, -self.t_pred :]

        if self.target_loss_only:
            pred_mask = pred_mask & (data['agent']['category'][:, None] > 0)

        total_loss = 0
        if self.loss_type == 'wta':
            best_mode = None
            for traj_distrs_net in traj_distrs:
                for overp_i, traj_distr in enumerate(traj_distrs_net):
                    s = overp_i * self.t_per_tok
                    loss, best_mode = traj_distr.wta_nll(
                        gt_pos[:, s:],
                        gt_head[:, s:],
                        pred_mask[:, s:],
                        best_mode=best_mode,
                        return_best_mode=True,
                    )
                    total_loss += loss
        elif self.loss_type in {'traj_nll', 'step_nll'}:
            for traj_distrs_net in traj_distrs:
                for overp_i, traj_distr in enumerate(traj_distrs_net):
                    s = overp_i * self.t_per_tok
                    loss = traj_distr.nll(
                        gt_pos[:, s:], gt_head[:, s:], pred_mask[:, s:]
                    )
                    total_loss += loss
        else:
            raise ValueError(f'Unknown loss_type {self.loss_type}.')

        total_loss = total_loss.mean()
        if total_loss.isnan():
            raise RuntimeError('loss is nan')

        return {'loss': total_loss}

    def training_step(self, batch, batch_idx):
        batch = deepcopy(batch)
        traj_distrs = self(batch)
        losses = self.compute_loss(batch, traj_distrs)
        for loss_name, loss_val in losses.items():
            prog_bar = loss_name == 'loss'
            self.log(
                f'train_{loss_name}',
                loss_val,
                prog_bar=prog_bar,
                on_step=True,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

        return losses['loss']

    def validation_step(self, batch, batch_idx):
        traj_distrs = self(batch)
        losses = self.compute_loss(batch, traj_distrs)
        for loss_name, loss_val in losses.items():
            prog_bar = loss_name == 'loss'
            self.log(
                f'val_{loss_name}',
                loss_val,
                prog_bar=prog_bar,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

        return losses['loss']

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.LSTMCell,
            nn.GRU,
            nn.GRUCell,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.LayerNorm,
            nn.RMSNorm,
            nn.GroupNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    '%s.%s' % (module_name, param_name) if module_name else param_name
                )
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                'params': [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                'weight_decay': self.weight_decay,
            },
            {
                'params': [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                'weight_decay': 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.decay_epochs, eta_min=0.0
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('DONUT')
        parser.add_argument('--t_per_tok', type=int, default=10)
        parser.add_argument('--t_hist', type=int, default=11)
        parser.add_argument('--t_pred', type=int, default=80)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--refine', type=int, default=1)
        parser.add_argument('--overpredict', type=int, default=1)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--map_enc_layers', type=int, default=1)
        parser.add_argument('--map_enc_radius', type=float, default=50)
        parser.add_argument('--edge_limit', type=float, default=0.9999)
        parser.add_argument('--dec_attn_order', type=str, default='trsm')
        parser.add_argument('--dec_attn_repetitions', type=int, default=2)
        parser.add_argument('--dec_radius_r', type=float, default=50)
        parser.add_argument('--dec_radius_s', type=float, default=50)
        parser.add_argument(
            '--position_distribution', type=str, default='generalized_gaussian'
        )
        parser.add_argument('--loss_type', type=str, default='step_nll')
        parser.add_argument('--target_loss_only', action='store_true')
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--acc_batch_size', type=int, default=64)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--decay_epochs', type=int, default=32)
        return parent_parser
