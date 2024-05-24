import copy
import torch
from torch import nn
import numpy as np
from PIL import ImageColor, Image, ImageDraw, ImageFont

import networks as networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA(object):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.values = torch.zeros((2,)).to(device)
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset.detach(), scale.detach()


class RoboCentricWorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(RoboCentricWorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.TwoHeadEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_input_layers,
            config.dyn_output_layers,
            config.dyn_rec_depth,
            config.dyn_shared,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_temp_post,
            config.dyn_min_std,
            config.dyn_cell,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiMaskDecoder(
            feat_size, shapes, **config.decoder
        )
        if config.reward_head == "symlog_disc":
            self.heads["reward"] = networks.MLP(
                feat_size * 2,  # pytorch version
                (255,),
                config.reward_layers,
                config.units,
                config.act,
                config.norm,
                dist=config.reward_head,
                outscale=0.0,
                device=config.device,
            )
        else:
            self.heads["reward"] = networks.MLP(
                feat_size * 2,  # pytorch version
                [],
                config.reward_layers,
                config.units,
                config.act,
                config.norm,
                dist=config.reward_head,
                outscale=0.0,
                device=config.device,
            )
        self.heads["cont"] = networks.MLP(
            feat_size * 2,  # pytorch version
            [],
            config.cont_layers,
            config.units,
            config.act,
            config.norm,
            dist="binary",
            device=config.device,
        )

        for name in config.grad_heads:
            assert name in self.heads, name

        self._scales = dict(reward=config.reward_scale, cont=config.cont_scale)

        self.robot_predictor = networks.robotPosePredictor(self.embed_size)

        if config.use_diff_lr:
            print("use diff lr")
            self._model_opt = tools.Optimizer(
                "model",
                self._get_split_params(),
                config.model_lr,
                config.opt_eps,
                config.grad_clip,
                config.weight_decay,
                opt=config.opt,
                use_amp=self._use_amp,
            )
        else:
            self._model_opt = tools.Optimizer(
                "model",
                self.parameters(),
                config.model_lr,
                config.opt_eps,
                config.grad_clip,
                config.weight_decay,
                opt=config.opt,
                use_amp=self._use_amp,
            )

    def _get_split_params(self):

        def is_env_param(name):
            env_param_names = ['_env_obs_stat_layer', '_env_obs_out_layers', '_cnn2', '_env_decoder', '_env_cell',
                               '_env_inp_layers', '_env_img_out_layers', '_env_ims_stat_layer', 'env_W',
                               'robot_env_atten', 'reward', 'cont']
            for param_name in env_param_names:
                if param_name in name:
                    print(f"set {name} param lr")
                    return True

            return False

        params = []
        params.append({"params": self.dynamics._env_inp_layers.parameters(), "lr": self._config.model_env_lr})
        params.append({"params": self.dynamics._env_cell.parameters(), "lr": self._config.model_env_lr})
        params.append({"params": self.dynamics._env_img_out_layers.parameters(), "lr": self._config.model_env_lr})
        params.append({"params": self.dynamics._env_obs_out_layers.parameters(), "lr": self._config.model_env_lr})
        params.append({"params": self.dynamics._env_ims_stat_layer.parameters(), "lr": self._config.model_env_lr})
        params.append({"params": self.dynamics._env_obs_stat_layer.parameters(), "lr": self._config.model_env_lr})
        params.append({"params": self.dynamics.env_W, "lr": self._config.model_env_lr})
        params.append({"params": self.dynamics.robot_env_atten.parameters(), "lr": self._config.model_env_lr})
        params.append({"params": self.encoder._cnn2.parameters(), "lr": self._config.model_env_lr})
        params.append({"params": self.heads["decoder"]._env_decoder.parameters(), "lr": self._config.model_env_lr})

        # reward need fast learn
        params.append({"params": self.heads["reward"].parameters(), "lr": self._config.model_env_lr})
        params.append({"params": self.heads["cont"].parameters(), "lr": self._config.model_env_lr})

        params.append({"params": [p for n, p in self.named_parameters() if not is_env_param(n)],
                       "lr": self._config.model_lr})

        return params

    def reset(self, reset_mode=2):
        def reset_task_related_head():
            self.heads['reward'].layers.apply(tools.weight_init)
            self.heads['reward'].mean_layer.apply(tools.uniform_weight_init(0))
            self.heads['cont'].layers.apply(tools.weight_init)
            self.heads['cont'].mean_layer.apply(tools.uniform_weight_init(1))

        def reset_env_branch_dyn():
            self.dynamics._env_inp_layers.apply(tools.weight_init)
            self.dynamics._env_cell.apply(tools.weight_init)
            self.dynamics._env_img_out_layers.apply(tools.weight_init)
            self.dynamics._env_obs_out_layers.apply(tools.weight_init)
            self.dynamics._env_ims_stat_layer.apply(tools.weight_init)
            self.dynamics._env_obs_stat_layer.apply(tools.weight_init)
            self.dynamics.env_W *= 0

        def reset_atten():
            self.dynamics.robot_deter_layer.apply(tools.weight_init)
            self.dynamics.robot_env_atten.apply(tools.weight_init)

        reset_mode = int(reset_mode)
        if reset_mode == 0:
            print(f"using reset_mode {reset_mode} success! \n reset nothing")
        elif reset_mode == 1:
            print(f"using reset_mode {reset_mode} success! \n reset task related heads")
            reset_task_related_head()
        elif reset_mode == 2:
            print(f"using reset_mode {reset_mode} success! \n reset task related heads and env branch")
            reset_task_related_head()
            reset_env_branch_dyn()
        elif reset_mode == 3:
            print(f"using reset_mode {reset_mode} success! \n reset task related heads, env branch and attention model")
            reset_task_related_head()
            reset_env_branch_dyn()
            reset_atten()
        else:
            raise NotImplementedError()

    def _train(self, data, use_mask=False):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                robot_embed, env_embed = self.encoder(data)

                robot_post, env_post, robot_prior, env_prior = self.dynamics.observe(
                    robot_embed, env_embed, data["action"], data["is_first"]
                )

                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale

                robot_kl_loss, robot_kl_value, robot_dyn_loss, robot_rep_loss = self.dynamics.kl_loss(
                    robot_post, robot_prior, kl_free, dyn_scale, rep_scale
                )

                env_kl_loss, env_kl_value, env_dyn_loss, env_rep_loss = self.dynamics.kl_loss(
                    env_post, env_prior, kl_free, dyn_scale, rep_scale
                )

                preds = {}
                for name, head in self.heads.items():
                    if 'decoder' in name:
                        robot_feat = self.dynamics.get_feat(robot_post)
                        env_feat = self.dynamics.get_feat(env_post)
                        pred, robot_recon, env_recon, robot_mask, env_mask = head(robot_feat, env_feat)
                    else:
                        grad_head = name in self._config.grad_heads
                        feat = self.concat_merge_feat(robot_post, env_post)
                        feat = feat if grad_head else feat.detach()
                        pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    if 'robot_mask' in name:
                        # don't use mask all the time
                        if use_mask:
                            like = pred.log_prob(data[name])
                            losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
                        else:
                            pass
                    else:
                        like = pred.log_prob(data[name])
                        losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)

                model_loss = sum(losses.values()) + robot_kl_loss + env_kl_loss
            metrics = self._model_opt(model_loss, self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["robot_dyn_loss"] = to_np(robot_dyn_loss)
        metrics["robot_rep_loss"] = to_np(robot_rep_loss)
        metrics["robot_kl"] = to_np(torch.mean(robot_kl_value))
        metrics["env_dyn_loss"] = to_np(env_dyn_loss)
        metrics["env_rep_loss"] = to_np(env_rep_loss)
        metrics["env_kl"] = to_np(torch.mean(env_kl_value))

        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_robot_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(robot_prior).entropy())
            )
            metrics["prior_env_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(env_prior).entropy())
            )
            metrics["post_robot_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(robot_post).entropy())
            )
            metrics["post_env_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(env_post).entropy())
            )
            context = dict(
                embed=(robot_embed, env_embed),
                feat=self.concat_merge_feat(robot_post, env_post),
                robot_kl=robot_kl_value,
                env_kl=env_kl_value,
                postent=self.dynamics.get_dist(robot_post).entropy(),
            )
        robot_post = {k: v.detach() for k, v in robot_post.items()}
        env_post = {k: v.detach() for k, v in env_post.items()}

        return robot_post, env_post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = obs.copy()
        obs["image"] = torch.Tensor(obs["image"]) / 255.0 - 0.5
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data, pretrain=False):
        data = self.preprocess(data)
        robot_embed, env_embed = self.encoder(data)
        batch_size = robot_embed.shape[0]
        seleted_slice = []

        if pretrain:
            for i in range(int(batch_size)):
                if i % self._config.batch_size == 0:
                    seleted_slice.append(i)
                    seleted_slice.append(i + 1)

        else:
            seleted_slice = [0, 1, 2, 3, 4, 5]

        robot_states, env_states, _, _ = self.dynamics.observe(
            robot_embed[seleted_slice, :5], env_embed[seleted_slice, :5], data["action"][seleted_slice, :5],
            data["is_first"][seleted_slice, :5]
        )

        robot_feat = self.dynamics.get_feat(robot_states)
        env_feat = self.dynamics.get_feat(env_states)
        head_out, robot_recon, env_recon, recon_mask_robot, recon_mask_env = self.heads["decoder"](robot_feat,
                                                                                                   env_feat)
        recon = head_out["image"].mode()
        recon_mask_robot = recon_mask_robot
        recon_mask_env = recon_mask_env

        robot_init = {k: v[:, -1] for k, v in robot_states.items()}
        env_init = {k: v[:, -1] for k, v in env_states.items()}
        robot_prior, env_prior = self.dynamics.imagine(data["action"][seleted_slice, 5:], (robot_init, env_init))

        robot_feat = self.dynamics.get_feat(robot_prior)
        env_feat = self.dynamics.get_feat(env_prior)

        head_pred, robot_openl, env_openl, openl_mask_robot, openl_mask_env = self.heads["decoder"](robot_feat,
                                                                                                    env_feat)
        openl = head_pred["image"].mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        robot_model = torch.cat([robot_recon[:, :5], robot_openl], 1)
        env_model = torch.cat([env_recon[:, :5], env_openl], 1)

        mask_model_robot = torch.cat([recon_mask_robot[:, :5], openl_mask_robot], 1).repeat(1, 1, 1, 1, 3)
        mask_model_env = torch.cat([recon_mask_env[:, :5], openl_mask_env], 1).repeat(1, 1, 1, 1, 3)

        truth = data["image"][seleted_slice, ...] + 0.5
        model = model + 0.5
        robot_model = robot_model + 0.5
        env_model = env_model + 0.5
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error, robot_model, env_model, mask_model_robot, mask_model_env], 2)


    def concat_merge_feat(self, robot_state, env_state):
        robot_stoch = robot_state["stoch"]
        env_stoch = env_state["stoch"]
        if self.dynamics._discrete:
            shape = list(robot_stoch.shape[:-2]) + [self.dynamics._stoch * self.dynamics._discrete]
            robot_stoch = robot_stoch.reshape(shape)
            env_stoch = env_stoch.reshape(shape)

        robot_feat = torch.cat([robot_stoch, robot_state["deter"]], -1)
        env_feat = torch.cat([env_stoch, env_state["deter"]], -1)

        feat = torch.cat([robot_feat, env_feat], -1)

        return feat


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model, stop_grad_actor=True, reward=None):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        self._stop_grad_actor = stop_grad_actor
        self._reward = reward
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        feat_size *= 2
        self.actor = networks.ActionHead(
            feat_size,
            config.num_actions,
            config.actor_layers,
            config.units,
            config.act,
            config.norm,
            config.actor_dist,
            config.actor_init_std,
            config.actor_min_std,
            config.actor_max_std,
            config.actor_temp,
            outscale=1.0,
            unimix_ratio=config.action_unimix_ratio,
        )
        if config.value_head == "symlog_disc":
            self.value = networks.MLP(
                feat_size,
                (255,),
                config.value_layers,
                config.units,
                config.act,
                config.norm,
                config.value_head,
                outscale=0.0,
                device=config.device,
            )
        else:
            self.value = networks.MLP(
                feat_size,
                [],
                config.value_layers,
                config.units,
                config.act,
                config.norm,
                config.value_head,
                outscale=0.0,
                device=config.device,
            )
        if config.slow_value_target:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor_lr,
            config.ac_opt_eps,
            config.actor_grad_clip,
            **kw,
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.value_lr,
            config.ac_opt_eps,
            config.value_grad_clip,
            **kw,
        )
        if self._config.reward_EMA:
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
            self,
            robot_start,
            env_start,
            objective=None,
            action=None,
            reward=None,
            imagine=None,
            tape=None,
            repeats=None,
    ):
        objective = objective or self._reward
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_robot_state, imag_env_state, imag_action = self._imagine(
                    robot_start, env_start, self.actor, self._config.imag_horizon, repeats
                )
                reward = objective(imag_robot_state, imag_env_state)
                actor_ent = self.actor(imag_feat).entropy()
                # this target is not scaled
                # slow is flag to indicate whether slow_target is used for lambda-return
                target, weights, base = self._compute_target(
                    imag_feat, imag_robot_state, imag_env_state, imag_action, reward, actor_ent, None
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    None,
                    imag_action,
                    target,
                    actor_ent,
                    None,
                    weights,
                    base,
                )
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.slow_value_target:
                    value_loss = value_loss - value.log_prob(
                        slow_target.mode().detach()
                    )
                if self._config.value_decay:
                    value_loss += self._config.value_decay * value.mode()
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor_dist in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_robot_state, imag_env_state, imag_action, weights, metrics

    def _imagine(self, robot_start, env_start, policy, horizon, repeats=None):
        dynamics = self._world_model.dynamics
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        robot_start = {k: flatten(v) for k, v in robot_start.items()}
        env_start = {k: flatten(v) for k, v in env_start.items()}

        def step(prev, _):
            robot_state, env_state, _, _ = prev
            feat = self._world_model.concat_merge_feat(robot_state, env_state)
            inp = feat.detach() if self._stop_grad_actor else feat
            action = policy(inp).sample()
            robot_succ, env_succ = dynamics.img_step((robot_state, env_state), action,
                                                     sample=self._config.imag_sample)
            return robot_succ, env_succ, feat, action

        robot_succ, env_succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (robot_start, env_start, None, None)
        )
        robot_states = {k: torch.cat([robot_start[k][None], v[:-1]], 0) for k, v in robot_succ.items()}
        env_states = {k: torch.cat([env_start[k][None], v[:-1]], 0) for k, v in env_succ.items()}

        if repeats:
            raise NotImplemented("repeats is not implemented in this version")

        return feats, robot_states, env_states, actions

    def _compute_target(
            self, imag_feat, imag_robot_state, imag_env_state, imag_action, reward, actor_ent, state_ent
    ):
        if "cont" in self._world_model.heads:
            inp = self._world_model.concat_merge_feat(imag_robot_state, imag_env_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        if self._config.future_entropy and self._config.actor_entropy > 0:
            reward += self._config.actor_entropy * actor_ent
        if self._config.future_entropy and self._config.actor_state_entropy > 0:
            reward += self._config.actor_state_entropy * state_ent
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
            self,
            imag_feat,
            imag_state,
            imag_action,
            target,
            actor_ent,
            state_ent,
            weights,
            base,
    ):
        metrics = {}
        inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
        policy = self.actor(inp)
        actor_ent = policy.entropy()
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            values = self.reward_ema.values
            metrics["EMA_005"] = to_np(values[0])
            metrics["EMA_095"] = to_np(values[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                    policy.log_prob(imag_action)[:-1][:, :, None]
                    * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                    policy.log_prob(imag_action)[:-1][:, :, None]
                    * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        if not self._config.future_entropy and self._config.actor_entropy > 0:
            actor_entropy = self._config.actor_entropy * actor_ent[:-1][:, :, None]
            actor_target += actor_entropy
        if not self._config.future_entropy and (self._config.actor_state_entropy > 0):
            state_entropy = self._config.actor_state_entropy * state_ent[:-1]
            actor_target += state_entropy
            metrics["actor_state_entropy"] = to_np(torch.mean(state_entropy))
        actor_loss = -torch.mean(weights[:-1] * actor_target)
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.slow_value_target:
            if self._updates % self._config.slow_target_update == 0:
                mix = self._config.slow_target_fraction
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
