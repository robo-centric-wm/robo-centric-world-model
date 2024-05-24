import argparse
import functools
import os
import pathlib
import sys
import collections

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import ruamel.yaml as yaml
import time
import shutil

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd

to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.RoboCentricWorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(
            config, self._wm, config.behavior_stop_grad
        )
        if (
                config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if self._should_reset(step):
            state = None
        if state is not None and reset.any():
            mask = 1 - reset
            for key in state[0].keys():
                for i in range(state[0][key].shape[0]):
                    state[0][key][i] *= mask[i]
            for key in state[1].keys():
                for i in range(state[1][key].shape[0]):
                    state[1][key][i] *= mask[i]
            for i in range(len(state[2])):
                state[2][i] *= mask[i]
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )

            if self._step < self._config.use_mask_steps:
                use_mask = True
            else:
                use_mask = False

            for _ in range(steps):
                self._train(next(self._dataset), use_mask)
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            batch_size = len(obs["image"])
            robot_latent, env_latent = self._wm.dynamics.initial(len(obs["image"]))
            action = torch.zeros((batch_size, self._config.num_actions)).to(
                self._config.device
            )
        else:
            robot_latent, env_latent, action = state
        obs = self._wm.preprocess(obs)
        robot_embed, env_embed = self._wm.encoder(obs)
        robot_latent, env_latent, _, _ = self._wm.dynamics.obs_step(
            (robot_latent, env_latent), action, robot_embed, env_embed, obs["is_first"],
            self._config.collect_dyn_sample
        )

        feat = self._wm.concat_merge_feat(robot_latent, env_latent)

        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        action = action.detach()
        if self._config.actor_dist == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        action = self._exploration(action, training)
        policy_output = {"action": action, "logprob": logprob}
        state = (robot_latent, env_latent, action)
        return policy_output, state

    def _exploration(self, action, training):
        amount = self._config.expl_amount if training else self._config.eval_noise
        if amount == 0:
            return action
        if "onehot" in self._config.actor_dist:
            probs = amount / self._config.num_actions + (1 - amount) * action
            return tools.OneHotDist(probs=probs).sample()
        else:
            return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)

    def _train(self, data, use_mask=False):
        metrics = {}
        robot_post, env_post, context, mets = self._wm._train(data, use_mask)
        metrics.update(mets)
        reward = lambda robot_state, env_state: self._wm.heads["reward"](
            self._wm.concat_merge_feat(robot_state, env_state)
        ).mode()
        metrics.update(self._task_behavior._train(robot_post, env_post, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(robot_post, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    def _train_wm(self, data, use_mask=False):
        metrics = {}
        robot_post, env_post, context, mets = self._wm._train(data, use_mask)
        metrics.update(mets)
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    def pretrain_wm(self, train_step):
        for i in range(int(train_step)):
            if self._update_count < self._config.use_mask_steps:
                use_mask = True
            else:
                use_mask = False
            data = self.get_data()
            self._train_wm(data, use_mask)
            self._update_count += 1
            self._metrics["update_count"] = self._update_count

        self._logger.step = self._update_count
        for name, values in self._metrics.items():
            self._logger.scalar(name, float(np.mean(values)))
            self._metrics[name] = []

        data = self.get_data()
        openl = self._wm.video_pred(data, pretrain=True)
        self._logger.video("train_openl", to_np(openl))
        self._logger.write(fps=True)

    def get_data(self):
        datas = {}
        for dataset in self._dataset:
            data = next(dataset)
            # datas.append(data)
            for k, v in data.items():
                if k in datas.keys():
                    datas[k] = np.concatenate([datas[k], v], axis=0)
                else:
                    datas[k] = v
        return datas

    def reset(self, reset_mode=2):
        self._wm.reset(reset_mode)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_multi_dataset(episodes_list, config):
    datasets = []
    for episodes in episodes_list:
        generator = tools.sample_episodes(episodes, config.batch_length)
        dataset = tools.from_generator(generator, config.batch_size)
        datasets.append(dataset)
    return datasets


def make_env(config, mode):
    suite, task = config.task.split("_", 1)
    if suite == "metaworld":
        import envs.meta_world as meta_world
        env = meta_world.MetaWorld(
            task,
            config.action_repeat,
            config.size,
            config.camera,
            config.seed,
            config.use_mask,
        )
        env = wrappers.NormalizeActions(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


def training(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    ######## code save ########
    base_dir = os.path.dirname(__file__)
    code_dir = logdir / 'code'
    code_dir.mkdir(parents=True, exist_ok=True)

    save_files = ["dreamer.py", "models.py", "networks.py", "configs.yaml"]
    for file in save_files:
        if 'config' in file:
            file_path = os.path.join(os.path.dirname(base_dir), file)
        else:
            file_path = os.path.join(base_dir, file)
        save_path = code_dir / file
        shutil.copyfile(file_path, save_path)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode: make_env(config, mode)
    train_envs = [make("train") for _ in range(config.envs)]
    eval_envs = [make("eval") for _ in range(config.eval_envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low).repeat(config.envs, 1),
                    torch.Tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


def pretraining(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    ######## code save ########
    base_dir = os.path.dirname(__file__)
    code_dir = logdir / 'code'
    code_dir.mkdir(parents=True, exist_ok=True)

    save_files = ["dreamer.py", "models.py", "networks.py", "configs.yaml"]
    for file in save_files:
        if 'config' in file:
            file_path = os.path.join(os.path.dirname(base_dir), file)
        else:
            file_path = os.path.join(base_dir, file)
        save_path = code_dir / file
        shutil.copyfile(file_path, save_path)

    print("Create envs.")
    make = lambda mode: make_env(config, mode)
    train_envs = make("train")

    acts = train_envs.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    observation_space = train_envs.observation_space
    action_space = train_envs.action_space
    del train_envs

    train_eps_list = []
    for dir in config.offline_traindir:
        train_eps = tools.load_episodes_random(dir, limit=config.dataset_size)
        train_eps_list.append(train_eps)
    train_datasets = make_multi_dataset(train_eps_list, config)

    print("Simulate agent.")
    agent = Dreamer(
        observation_space,
        action_space,
        config,
        logger,
        train_datasets,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    while agent._update_count < config.pretrain_steps:
        logger.write()
        print(f"Start training {agent._update_count}/{config.pretrain_steps}.")
        agent.pretrain_wm(config.log_every)
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / f"pretrain_{agent._update_count}.pt")


def finetuning(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    ######## code save ########
    base_dir = os.path.dirname(__file__)
    code_dir = logdir / 'code'
    code_dir.mkdir(parents=True, exist_ok=True)

    save_files = ["dreamer.py", "models.py", "networks.py", "configs.yaml"]
    for file in save_files:
        if 'config' in file:
            file_path = os.path.join(os.path.dirname(base_dir), file)
        else:
            file_path = os.path.join(base_dir, file)
        save_path = code_dir / file
        shutil.copyfile(file_path, save_path)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode: make_env(config, mode)
    train_envs = [make("train") for _ in range(config.envs)]
    eval_envs = [make("eval") for _ in range(config.eval_envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low).repeat(config.envs, 1),
                    torch.Tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)

    # load pretrained world model
    checkpoint = torch.load(config.pretrain_model_dir)['agent_state_dict']
    wm_state_dict = {key: value for key, value in checkpoint.items() if
                     'wm' in key and 'reward' not in key and 'cont' not in key}

    agent.load_state_dict(wm_state_dict, strict=False)

    # reset
    agent.reset(config.reset_mode)
    print(f'reset mode {config.reset_mode}')

    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "finetune_latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-mode", type=str, default='training')
    parser.add_argument("--configs", nargs="+", type=str, default='metaworld_vision')
    parser.add_argument('--logtime', type=str, default=time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    args, remaining = parser.parse_known_args()
    run_mode = args.run_mode
    logtime = args.logtime

    config_name = args.configs if isinstance(args.configs, str) else args.configs[0]
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent.parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    args = parser.parse_args(remaining)

    if run_mode == 'training':
        args.logdir = f"./logdir/{run_mode}/" + config_name + '/' + str(logtime) + '-seed' + str(
            args.seed) + '/' + args.task
        training(args)

    elif run_mode == 'pretraining':
        args.batch_size = args.batch_size // len(args.offline_traindir)

        args.logdir = "./logdir/pretraining/" + str(logtime)
        pretraining(args)

    elif run_mode == 'finetuning':
        args.logdir = f"./logdir/finetuning/{config_name}/reset-mode-{str(args.reset_mode)}/{str(logtime)}-seed{str(args.seed)}/{args.task}"

        finetuning(args)


if __name__ == "__main__":
    main()
