import math
import numpy as np
import re
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd
from torch.nn import init
import tools


class RSSM(nn.Module):
    def __init__(
            self,
            stoch=30,
            deter=200,
            hidden=200,
            layers_input=1,
            layers_output=1,
            rec_depth=1,
            shared=False,
            discrete=False,
            act="SiLU",
            norm="LayerNorm",
            mean_act="none",
            std_act="softplus",
            temp_post=True,
            min_std=0.1,
            cell="gru",
            unimix_ratio=0.01,
            initial="learned",
            num_actions=None,
            embed=None,
            device=None,
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._layers_input = layers_input
        self._layers_output = layers_output
        self._rec_depth = rec_depth
        self._shared = shared
        self._discrete = discrete
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._mean_act = mean_act
        self._std_act = std_act
        self._temp_post = temp_post
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._embed = embed
        self._device = device

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        if self._shared:
            inp_dim += self._embed
        for i in range(self._layers_input):
            inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            inp_layers.append(norm(self._hidden, eps=1e-03))
            inp_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._robot_inp_layers = nn.Sequential(*inp_layers)
        self._robot_inp_layers.apply(tools.weight_init)

        inp_layers = []
        if self._discrete:
            # inp_dim = self._stoch * self._discrete * 2 + self._deter
            inp_dim = self._stoch * self._discrete
        else:
            inp_dim = self._stoch + num_actions
        if self._shared:
            inp_dim += self._embed
        for i in range(self._layers_input):
            inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            inp_layers.append(norm(self._hidden, eps=1e-03))
            inp_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._env_inp_layers = nn.Sequential(*inp_layers)
        self._env_inp_layers.apply(tools.weight_init)

        if cell == "gru":
            self._robot_cell = GRUCell(self._hidden, self._deter)
            self._robot_cell.apply(tools.weight_init)
        elif cell == "gru_layer_norm":
            self._robot_cell = GRUCell(self._hidden, self._deter, norm=True)
            self._robot_cell.apply(tools.weight_init)
        else:
            raise NotImplementedError(cell)

        self._env_cell = copy.deepcopy(self._robot_cell)

        img_out_layers = []
        inp_dim = self._deter
        for i in range(self._layers_output):
            img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            img_out_layers.append(norm(self._hidden, eps=1e-03))
            img_out_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._robot_img_out_layers = nn.Sequential(*img_out_layers)
        self._robot_img_out_layers.apply(tools.weight_init)
        self._env_img_out_layers = copy.deepcopy(self._robot_img_out_layers)

        obs_out_layers = []
        if self._temp_post:
            inp_dim = self._deter + self._embed
        else:
            inp_dim = self._embed
        for i in range(self._layers_output):
            obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            obs_out_layers.append(norm(self._hidden, eps=1e-03))
            obs_out_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._robot_obs_out_layers = nn.Sequential(*obs_out_layers)
        self._robot_obs_out_layers.apply(tools.weight_init)

        obs_out_layers = []
        if self._temp_post:
            inp_dim = self._deter + self._embed
        else:
            inp_dim = self._embed
        for i in range(self._layers_output):
            obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            obs_out_layers.append(norm(self._hidden, eps=1e-03))
            obs_out_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._env_obs_out_layers = nn.Sequential(*obs_out_layers)
        self._env_obs_out_layers.apply(tools.weight_init)

        if self._discrete:
            self._robot_ims_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._robot_ims_stat_layer.apply(tools.weight_init)
            self._robot_obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._robot_obs_stat_layer.apply(tools.weight_init)

            self._env_ims_stat_layer = copy.deepcopy(self._robot_ims_stat_layer)
            self._env_obs_stat_layer = copy.deepcopy(self._env_ims_stat_layer)
        else:
            self._robot_ims_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._robot_ims_stat_layer.apply(tools.weight_init)
            self._robot_obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._robot_obs_stat_layer.apply(tools.weight_init)

            self._env_ims_stat_layer = copy.deepcopy(self._robot_ims_stat_layer)
            self._env_obs_stat_layer = copy.deepcopy(self._env_ims_stat_layer)

        if self._initial == "learned":
            self.robot_W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

            self.env_W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

        self.robot_deter_layer = nn.Sequential(
            nn.Linear(self._deter, self._deter),
        )

        self.robot_env_atten = ScaledDotProductAttention(self._discrete, self._discrete, self._discrete, h=1)

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter).to(self._device)
        if self._discrete:
            robot_state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                deter=deter,
            )

            env_state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                deter=deter,
            )
        else:
            robot_state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                std=torch.zeros([batch_size, self._stoch]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
            )

            env_state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                std=torch.zeros([batch_size, self._stoch]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return robot_state, env_state
        elif self._initial == "learned":
            robot_state["deter"] = torch.tanh(self.robot_W).repeat(batch_size, 1)
            robot_state["stoch"] = self.get_robot_stoch(robot_state["deter"])

            env_state["deter"] = torch.tanh(self.env_W).repeat(batch_size, 1)
            env_state["stoch"] = self.get_env_stoch(env_state["deter"])

            return robot_state, env_state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, robot_embed, env_embed, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])

        robot_state, env_state = state
        # (batch, time, ch) -> (time, batch, ch)
        robot_embed, env_embed, action, is_first = swap(robot_embed), swap(env_embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        robot_post, env_post, robot_prior, env_prior = tools.static_scan(
            lambda prev_state, prev_act, robot_embed, env_embed, is_first: self.obs_step(
                prev_state[:2], prev_act, robot_embed, env_embed, is_first
            ),
            (action, robot_embed, env_embed, is_first),
            (robot_state, env_state, robot_state, env_state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        robot_post = {k: swap(v) for k, v in robot_post.items()}
        env_post = {k: swap(v) for k, v in env_post.items()}
        robot_prior = {k: swap(v) for k, v in robot_prior.items()}
        env_prior = {k: swap(v) for k, v in env_prior.items()}
        return robot_post, env_post, robot_prior, env_prior

    def imagine(self, action, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        robot_state, env_state = state
        action = action
        action = swap(action)
        robot_prior, env_prior = tools.static_scan(self.img_step, [action], state)
        robot_prior = {k: swap(v) for k, v in robot_prior.items()}
        env_prior = {k: swap(v) for k, v in env_prior.items()}
        return robot_prior, env_prior

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, robot_embed, env_embed, is_first, sample=True):
        # if shared is True, prior and post both use same networks(inp_layers, _img_out_layers, _ims_stat_layer)
        # otherwise, post use different network(_obs_out_layers) with prior[deter] and embed as inputs
        prev_action *= (1.0 / torch.clip(torch.abs(prev_action), min=1.0)).detach()
        prev_robot_state, prev_env_state = prev_state
        if torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            robot_init_state, env_init_state = self.initial(len(is_first))
            for key, val in prev_robot_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_robot_state[key] = (
                        val * (1.0 - is_first_r) + robot_init_state[key] * is_first_r
                )
            for key, val in prev_env_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_env_state[key] = (
                        val * (1.0 - is_first_r) + env_init_state[key] * is_first_r
                )

        robot_prior, env_prior = self.img_step(prev_state, prev_action, None, sample)
        if self._shared:
            post = self.img_step(prev_state, prev_action, robot_embed, sample)
            raise NotImplementedError(self._shared)
        else:
            if self._temp_post:
                robot_x = torch.cat([robot_prior["deter"], robot_embed], -1)
                env_x = torch.cat([env_prior["deter"], env_embed], -1)
            else:
                robot_x = robot_embed
                env_x = env_embed
            # (batch_size, prior_deter + embed) -> (batch_size, hidden)
            robot_x = self._robot_obs_out_layers(robot_x)
            env_x = self._env_obs_out_layers(env_x)
            # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
            robot_stats = self._robot_suff_stats_layer("obs", robot_x)
            env_stats = self._env_suff_stats_layer("obs", env_x)
            if sample:
                robot_stoch = self.get_dist(robot_stats).sample()
                env_stoch = self.get_dist(env_stats).sample()
            else:
                robot_stoch = self.get_dist(robot_stats).mode()
                env_stoch = self.get_dist(env_stats).mode()
            robot_post = {"stoch": robot_stoch, "deter": robot_prior["deter"], **robot_stats}
            env_post = {"stoch": env_stoch, "deter": env_prior["deter"], **env_stats}
        return robot_post, env_post, robot_prior, env_prior

    # this is used for making future image
    def img_step(self, prev_state, prev_action, embed=None, sample=True):
        # (batch, stoch, discrete_num)
        prev_robot_state, prev_env_state = prev_state
        prev_action *= (1.0 / torch.clip(torch.abs(prev_action), min=1.0)).detach()
        prev_robot_stoch = prev_robot_state["stoch"]
        prev_env_stoch = prev_env_state["stoch"]
        if self._discrete:
            shape = list(prev_robot_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_robot_stoch = prev_robot_stoch.reshape(shape)

            shape = list(prev_env_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_env_stoch = prev_env_stoch.reshape(shape)
        if self._shared:
            if embed is None:
                shape = list(prev_action.shape[:-1]) + [self._embed]
                embed = torch.zeros(shape)
            # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action, embed)
            x = torch.cat([prev_robot_stoch, prev_action, embed], -1)
            raise NotImplementedError
        else:
            robot_x = torch.cat([prev_robot_stoch, prev_action], -1)
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        robot_x = self._robot_inp_layers(robot_x)

        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            robot_deter = prev_robot_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            robot_x, robot_deter = self._robot_cell(robot_x, [robot_deter])
            robot_deter = robot_deter[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        robot_x = self._robot_img_out_layers(robot_x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        robot_stats = self._robot_suff_stats_layer("ims", robot_x)
        if sample:
            robot_stoch = self.get_dist(robot_stats).sample()
        else:
            robot_stoch = self.get_dist(robot_stats).mode()

        robot_prior = {"stoch": robot_stoch, "deter": robot_deter, **robot_stats}

        ### then calculate env branch ###

        # use stoch and deter
        robot_deter_projection = self.robot_deter_layer(robot_deter).reshape(robot_deter.shape[0], -1,
                                                                                   self._discrete)
        atten_kv_input = torch.concat([robot_stoch, robot_deter_projection], dim=-2)

        atten = self.robot_env_atten(prev_env_state["stoch"], atten_kv_input).reshape(-1, self._stoch * self._discrete)

        env_x = self._env_inp_layers(atten)

        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            env_deter = prev_env_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            env_x, env_deter = self._env_cell(env_x, [env_deter])
            env_deter = env_deter[0]  # Keras wraps the state in a list.
        env_x = self._env_img_out_layers(env_x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        env_stats = self._env_suff_stats_layer("ims", env_x)

        if sample:
            env_stoch = self.get_dist(env_stats).sample()
        else:
            env_stoch = self.get_dist(env_stats).mode()

        env_prior = {"stoch": env_stoch, "deter": env_deter, **env_stats}
        return robot_prior, env_prior

    def get_robot_stoch(self, deter):
        x = self._robot_img_out_layers(deter)
        stats = self._robot_suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def get_env_stoch(self, deter):
        x = self._env_img_out_layers(deter)
        stats = self._env_suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _robot_suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._robot_ims_stat_layer(x)
            elif name == "obs":
                x = self._robot_obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._robot_ims_stat_layer(x)
            elif name == "obs":
                x = self._robot_obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def _env_suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._env_ims_stat_layer(x)
            elif name == "obs":
                x = self._env_obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._env_ims_stat_layer(x)
            elif name == "obs":
                x = self._env_obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        rep_loss = torch.mean(torch.clip(rep_loss, min=free))
        dyn_loss = torch.mean(torch.clip(dyn_loss, min=free))
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


class MultiEncoder(nn.Module):
    def __init__(
            self,
            shapes,
            mlp_keys,
            cnn_keys,
            act,
            norm,
            cnn_depth,
            kernel_size,
            minres,
            mlp_layers,
            mlp_units,
            symlog_inputs,
    ):
        super(MultiEncoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(
                input_shape, cnn_depth, act, norm, kernel_size, minres
            )
            self.outdim += self._cnn.outdim
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
            )
            self.outdim += mlp_units

    def forward(self, obs):
        outputs = []
        if self.cnn_shapes:
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            outputs.append(self._cnn(inputs))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, -1)
        return outputs


class MultiDecoder(nn.Module):
    def __init__(
            self,
            feat_size,
            shapes,
            mlp_keys,
            cnn_keys,
            act,
            norm,
            cnn_depth,
            kernel_size,
            minres,
            mlp_layers,
            mlp_units,
            cnn_sigmoid,
            image_dist,
            vector_dist,
    ):
        super(MultiDecoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                cnn_sigmoid=cnn_sigmoid,
            )
        if self.mlp_shapes:
            self._mlp = MLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
            )
        self._image_dist = image_dist

    def forward(self, features):
        dists = {}
        if self.cnn_shapes:
            feat = features
            outputs = self._cnn(feat)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class ConvEncoder(nn.Module):
    def __init__(
            self,
            input_shape,
            depth=32,
            act="SiLU",
            norm="LayerNorm",
            kernel_size=4,
            minres=4,
    ):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        h, w, input_ch = input_shape
        layers = []
        for i in range(int(np.log2(h) - np.log2(minres))):
            if i == 0:
                in_dim = input_ch
            else:
                in_dim = 2 ** (i - 1) * depth
            out_dim = 2 ** i * depth
            layers.append(
                Conv2dSame(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            layers.append(ChLayerNorm(out_dim))
            layers.append(act())
            h, w = h // 2, w // 2

        self.outdim = out_dim * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class ConvDecoder(nn.Module):
    def __init__(
            self,
            feat_size,
            shape=(3, 64, 64),
            depth=32,
            act=nn.ELU,
            norm=nn.LayerNorm,
            kernel_size=4,
            minres=4,
            outscale=1.0,
            cnn_sigmoid=False,
    ):
        super(ConvDecoder, self).__init__()
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres
        self._embed_size = minres ** 2 * depth * 2 ** (layer_num - 1)

        self._linear_layer = nn.Linear(feat_size, self._embed_size)
        self._linear_layer.apply(tools.weight_init)
        in_dim = self._embed_size // (minres ** 2)

        layers = []
        h, w = minres, minres
        for i in range(layer_num):
            out_dim = self._embed_size // (minres ** 2) // (2 ** (i + 1))
            bias = False
            initializer = tools.weight_init
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False
                initializer = tools.uniform_weight_init(outscale)

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ChLayerNorm(out_dim))
            if act:
                layers.append(act())
            [m.apply(initializer) for m in layers[-3:]]
            h, w = h * 2, w * 2

        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self._minres, self._minres, self._embed_size // self._minres ** 2]
        )
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch, time, -1) -> (batch * time, ch, h, w) necessary???
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch * time, ch, h, w) -> (batch * time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean) - 0.5
        return mean


class MLP(nn.Module):
    def __init__(
            self,
            inp_dim,
            shape,
            layers,
            units,
            act="SiLU",
            norm="LayerNorm",
            dist="normal",
            std=1.0,
            outscale=1.0,
            symlog_inputs=False,
            device="cuda",
    ):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        self._layers = layers
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._dist = dist
        self._std = std
        self._symlog_inputs = symlog_inputs
        self._device = device

        layers = []
        for index in range(self._layers):
            layers.append(nn.Linear(inp_dim, units, bias=False))
            layers.append(norm(units, eps=1e-03))
            layers.append(act())
            if index == 0:
                inp_dim = units
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = tools.symlog(x)
        out = self.layers(x)
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(
                    torchd.normal.Normal(mean, std), len(shape)
                )
            )
        if dist == "huber":
            return tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0), len(shape)
                )
            )
        if dist == "binary":
            return tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        if dist == "symlog_disc":
            return tools.DiscDist(logits=mean, device=self._device)
        if dist == "symlog_mse":
            return tools.SymlogDist(mean)
        raise NotImplementedError(dist)


class ActionHead(nn.Module):
    def __init__(
            self,
            inp_dim,
            size,
            layers,
            units,
            act=nn.ELU,
            norm=nn.LayerNorm,
            dist="trunc_normal",
            init_std=0.0,
            min_std=0.1,
            max_std=1.0,
            temp=0.1,
            outscale=1.0,
            unimix_ratio=0.01,
    ):
        super(ActionHead, self).__init__()
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._min_std = min_std
        self._max_std = max_std
        self._init_std = init_std
        self._unimix_ratio = unimix_ratio
        self._temp = temp() if callable(temp) else temp

        pre_layers = []
        for index in range(self._layers):
            pre_layers.append(nn.Linear(inp_dim, self._units, bias=False))
            pre_layers.append(norm(self._units, eps=1e-03))
            pre_layers.append(act())
            if index == 0:
                inp_dim = self._units
        self._pre_layers = nn.Sequential(*pre_layers)
        self._pre_layers.apply(tools.weight_init)

        if self._dist in ["tanh_normal", "tanh_normal_5", "normal", "trunc_normal"]:
            self._dist_layer = nn.Linear(self._units, 2 * self._size)
            self._dist_layer.apply(tools.uniform_weight_init(outscale))

        elif self._dist in ["normal_1", "onehot", "onehot_gumbel"]:
            self._dist_layer = nn.Linear(self._units, self._size)
            self._dist_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        x = self._pre_layers(x)
        if self._dist == "tanh_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = torch.tanh(mean)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "tanh_normal_5":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = 5 * torch.tanh(mean / 5)
            std = F.softplus(std + 5) + 5
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "normal_1":
            mean = self._dist_layer(x)
            dist = torchd.normal.Normal(mean, 1)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "trunc_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "onehot":
            x = self._dist_layer(x)
            dist = tools.OneHotDist(x, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            x = self._dist_layer(x)
            temp = self._temp
            dist = tools.ContDist(torchd.gumbel.Gumbel(x, 1 / temp))
        else:
            raise NotImplementedError(self._dist)
        return dist


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=False, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(inp_size + size, 3 * size, bias=False)
        if norm:
            self._norm = nn.LayerNorm(3 * size, eps=1e-03)

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSame(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class robotPosePredictor(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(feat_size, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.pos_head = nn.Sequential(
            nn.Linear(256, 9),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.base(x)
        pose = self.pos_head(x)

        return pose


class MaskDecoder(nn.Module):
    def __init__(
            self,
            feat_size,
            shape=(3, 64, 64),
            depth=32,
            act=nn.ELU,
            norm=nn.LayerNorm,
            kernel_size=4,
            minres=4,
            outscale=1.0,
            cnn_sigmoid=False,
    ):
        super(MaskDecoder, self).__init__()
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._shape = list(shape)
        self._shape[0] += 1  # add mask
        self._shape = tuple(self._shape)
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres
        self._embed_size = minres ** 2 * depth * 2 ** (layer_num - 1)

        self._linear_layer = nn.Linear(feat_size, self._embed_size)
        self._linear_layer.apply(tools.weight_init)
        in_dim = self._embed_size // (minres ** 2)

        layers = []
        h, w = minres, minres
        for i in range(layer_num):
            out_dim = self._embed_size // (minres ** 2) // (2 ** (i + 1))
            bias = False
            initializer = tools.weight_init
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False
                initializer = tools.uniform_weight_init(outscale)

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ChLayerNorm(out_dim))
            if act:
                layers.append(act())
            [m.apply(initializer) for m in layers[-3:]]
            h, w = h * 2, w * 2

        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self._minres, self._minres, self._embed_size // self._minres ** 2]
        )
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch, time, -1) -> (batch * time, ch, h, w) necessary???
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch * time, ch, h, w) -> (batch * time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean) - 0.5

        mask = mean[..., -1:]
        recon = mean[..., :-1]
        if not self._cnn_sigmoid:
            mask = F.sigmoid(mask)
        return recon, mask


class MultiMaskDecoder(nn.Module):
    def __init__(
            self,
            feat_size,
            shapes,
            mlp_keys,
            cnn_keys,
            act,
            norm,
            cnn_depth,
            kernel_size,
            minres,
            mlp_layers,
            mlp_units,
            cnn_sigmoid,
            image_dist,
            vector_dist,
    ):
        super(MultiMaskDecoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._robot_decoder = MaskDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                cnn_sigmoid=cnn_sigmoid,
            )
            self._env_decoder = MaskDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                cnn_sigmoid=cnn_sigmoid,
            )
        if self.mlp_shapes:
            raise NotImplementedError
        self._image_dist = image_dist

    def forward(self, robot_features, env_features):
        dists = {}
        if self.cnn_shapes:
            robot_recon, robot_mask = self._robot_decoder(robot_features)
            env_recon, env_mask = self._env_decoder(env_features)

            robot_out = robot_recon * robot_mask
            env_out = env_mask * env_recon

            recon = robot_out + env_out
            dists['image'] = self._make_image_dist(recon)
            dists['robot_mask'] = self._make_image_dist(robot_mask)

        if self.mlp_shapes:
            raise NotImplementedError
        return dists, robot_out, env_out, robot_mask, env_mask

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class TwoHeadEncoder(nn.Module):
    def __init__(
            self,
            shapes,
            mlp_keys,
            cnn_keys,
            act,
            norm,
            cnn_depth,
            kernel_size,
            minres,
            mlp_layers,
            mlp_units,
            symlog_inputs,
    ):
        super(TwoHeadEncoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)

            self._cnn_share = ConvEncoderFeat(
                input_shape, cnn_depth, act, norm, kernel_size, minres * 4
            )
            feat_shape = self._cnn_share.out_shape

            self._cnn1 = ConvHalfEncoder(
                feat_shape, cnn_depth * 4, act, norm, kernel_size, minres
            )

            self._cnn2 = ConvHalfEncoder(
                feat_shape, cnn_depth * 4, act, norm, kernel_size, minres
            )

            self.outdim += self._cnn1.outdim

    def forward(self, obs):
        if self.cnn_shapes:
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            feat = self._cnn_share(inputs)
            out1 = self._cnn1(feat)
            out2 = self._cnn2(feat)
        else:
            raise NotImplementedError
        return out1, out2


class ConvEncoderFeat(nn.Module):
    def __init__(
            self,
            input_shape,
            depth=32,
            act="SiLU",
            norm="LayerNorm",
            kernel_size=4,
            minres=4,
    ):
        super(ConvEncoderFeat, self).__init__()
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        h, w, input_ch = input_shape
        layers = []
        for i in range(int(np.log2(h) - np.log2(minres))):
            if i == 0:
                in_dim = input_ch
            else:
                in_dim = 2 ** (i - 1) * depth
            out_dim = 2 ** i * depth
            layers.append(
                Conv2dSame(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            layers.append(ChLayerNorm(out_dim))
            layers.append(act())
            h, w = h // 2, w // 2

        self.outdim = out_dim * h * w
        self.out_shape = (h, w, out_dim)
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch * time, ch, h, w) -> (batch * time, h, w, ch)
        x = x.permute(0, 2, 3, 1)
        # (batch * time, ch, h, w) -> (batch, time, h, w, ch)
        x = x.reshape(list(obs.shape[:-3]) + list(x.shape[1:]))
        return x


class ConvHalfEncoder(nn.Module):
    def __init__(
            self,
            input_shape,
            depth=32,
            act="SiLU",
            norm="LayerNorm",
            kernel_size=4,
            minres=4,
    ):
        super(ConvHalfEncoder, self).__init__()
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        h, w, input_ch = input_shape
        layers = []
        num_layers = int(np.log2(h) - np.log2(minres))
        for i in range(num_layers):
            if i == 0:
                in_dim = input_ch
            else:
                in_dim = 2 ** (i - 1) * depth
            if i == num_layers - 1:
                out_dim = in_dim
            else:
                out_dim = 2 ** i * depth
            layers.append(
                Conv2dSame(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            layers.append(ChLayerNorm(out_dim))
            layers.append(act())
            h, w = h // 2, w // 2

        self.outdim = out_dim * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys_values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys_values.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys_values).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(keys_values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)

        atten_out = self.fc_o(out)  # (b_s, nq, d_model)

        # add residual
        out = self.layernorm(queries + atten_out)

        return out
