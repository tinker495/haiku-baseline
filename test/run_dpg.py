import argparse
import os

import gymnasium as gym

from jax_baselines.DDPG.ddpg import DDPG
from jax_baselines.SAC.sac import SAC
from jax_baselines.TD3.td3 import TD3
from jax_baselines.TD7.td7 import TD7
from jax_baselines.TQC.tqc import TQC

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.0000625, help="learning rate")
    parser.add_argument("--model_lib", type=str, default="flax", help="model lib")
    parser.add_argument("--env", type=str, default="Pendulum-v0", help="environment")
    parser.add_argument("--worker_id", type=int, default=0, help="unlty ml agent's worker id")
    parser.add_argument("--worker", type=int, default=1, help="gym_worker_size")
    parser.add_argument("--algo", type=str, default="DDPG", help="algo ID")
    parser.add_argument("--gamma", type=float, default=0.995, help="gamma")
    parser.add_argument(
        "--target_update_tau", type=float, default=2e-3, help="target update intervals"
    )
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--buffer_size", type=float, default=100000, help="buffer_size")
    parser.add_argument("--per", action="store_true")
    parser.add_argument(
        "--n_step",
        type=int,
        default=1,
        help="n step setting when n > 1 is n step td method",
    )
    parser.add_argument("--steps", type=float, default=1e6, help="step size")
    parser.add_argument("--verbose", type=int, default=0, help="verbose")
    parser.add_argument("--logdir", type=str, default="log/dpg/", help="log file dir")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--n_support", type=int, default=25, help="n_support for QRDQN,IQN,FQF")
    parser.add_argument("--mixture", type=str, default="truncated", help="mixture type")
    parser.add_argument("--quantile_drop", type=float, default=0.1, help="quantile_drop ratio")
    parser.add_argument("--node", type=int, default=256, help="network node number")
    parser.add_argument("--hidden_n", type=int, default=2, help="hidden layer number")
    parser.add_argument("--action_noise", type=float, default=0.1, help="action_noise")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimaizer")
    parser.add_argument("--gradient_steps", type=int, default=1, help="gradient_steps")
    parser.add_argument("--train_freq", type=int, default=1, help="train_frequancy")
    parser.add_argument("--critic_num", type=int, default=2, help="tqc critic number")
    parser.add_argument("--ent_coef", type=str, default="auto", help="sac entropy coefficient")
    parser.add_argument("--learning_starts", type=int, default=5000, help="learning start")
    parser.add_argument("--cvar", type=float, default=1.0, help="cvar")
    parser.add_argument("--time_scale", type=float, default=20.0, help="unity time scale")
    parser.add_argument(
        "--capture_frame_rate", type=int, default=1, help="unity capture frame rate"
    )
    args = parser.parse_args()
    env_name = args.env
    embedding_mode = "normal"
    if os.path.exists(env_name):
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.side_channel.engine_configuration_channel import (
            EngineConfigurationChannel,
        )
        from mlagents_envs.side_channel.environment_parameters_channel import (
            EnvironmentParametersChannel,
        )

        engine_configuration_channel = EngineConfigurationChannel()
        channel = EnvironmentParametersChannel()
        engine_configuration_channel.set_configuration_parameters(
            time_scale=args.time_scale, capture_frame_rate=args.capture_frame_rate
        )
        env = UnityEnvironment(
            file_name=env_name,
            worker_id=args.worker_id,
            no_graphics=False,
            side_channels=[engine_configuration_channel, channel],
        )
        env_name = env_name.split("/")[-1].split(".")[0]
        env_type = "unity"
    else:
        # import mujoco_py
        if args.worker > 1:
            from jax_baselines.common.worker import gymnasium as gymMultiworker

            env = gymMultiworker(env_name, worker_num=args.worker)
        else:
            env = gym.make(env_name)
        env_type = "gym"

    policy_kwargs = {"node": args.node, "hidden_n": args.hidden_n, "embedding_mode": embedding_mode}

    if args.algo == "DDPG":
        if args.model_lib == "flax":
            from model_builder.flax.dpg.ddpg_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.dpg.ddpg_builder import model_builder_maker
        agent = DDPG(
            env,
            model_builder_maker,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_tau=args.target_update_tau,
            learning_starts=args.learning_starts,
            prioritized_replay=args.per,
            n_step=args.n_step,
            train_freq=args.train_freq,
            seed=args.seed,
            gradient_steps=args.gradient_steps,
            tensorboard_log=args.logdir + env_type + "/" + env_name,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
        )
    if args.algo == "TD3":
        if args.model_lib == "flax":
            from model_builder.flax.dpg.td3_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.dpg.td3_builder import model_builder_maker
        agent = TD3(
            env,
            model_builder_maker,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_tau=args.target_update_tau,
            learning_starts=args.learning_starts,
            prioritized_replay=args.per,
            action_noise=args.action_noise,
            n_step=args.n_step,
            train_freq=args.train_freq,
            seed=args.seed,
            gradient_steps=args.gradient_steps,
            tensorboard_log=args.logdir + env_type + "/" + env_name,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
        )
    if args.algo == "SAC":
        if args.model_lib == "flax":
            from model_builder.flax.dpg.sac_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.dpg.sac_builder import model_builder_maker
        agent = SAC(
            env,
            model_builder_maker,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_tau=args.target_update_tau,
            learning_starts=args.learning_starts,
            prioritized_replay=args.per,
            n_step=args.n_step,
            train_freq=args.train_freq,
            ent_coef=args.ent_coef,
            seed=args.seed,
            gradient_steps=args.gradient_steps,
            tensorboard_log=args.logdir + env_type + "/" + env_name,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
        )
    if args.algo == "TQC":
        if args.model_lib == "flax":
            from model_builder.flax.dpg.tqc_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.dpg.tqc_builder import model_builder_maker
        agent = TQC(
            env,
            model_builder_maker,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_tau=args.target_update_tau,
            learning_starts=args.learning_starts,
            quantile_drop=args.quantile_drop,
            prioritized_replay=args.per,
            n_step=args.n_step,
            train_freq=args.train_freq,
            ent_coef=args.ent_coef,
            seed=args.seed,
            gradient_steps=args.gradient_steps,
            n_support=args.n_support,
            critic_num=args.critic_num,
            mixture_type=args.mixture,
            tensorboard_log=args.logdir + env_type + "/" + env_name,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
        )
    if args.algo == "TD7":
        if args.model_lib == "flax":
            from model_builder.flax.dpg.td7_builder import model_builder_maker
        elif args.model_lib == "haiku":
            from model_builder.haiku.dpg.td7_builder import model_builder_maker
        eval_env = gym.make(env_name)
        agent = TD7(
            env,
            eval_env,
            model_builder_maker,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            batch_size=args.batch,
            buffer_size=int(args.buffer_size),
            target_network_update_freq=250,
            learning_starts=args.learning_starts,
            action_noise=args.action_noise,
            train_freq=args.train_freq,
            seed=args.seed,
            gradient_steps=args.gradient_steps,
            tensorboard_log=args.logdir + env_type + "/" + env_name,
            policy_kwargs=policy_kwargs,
            optimizer=args.optimizer,
        )

    agent.learn(int(args.steps))

    agent.test()

    env.close()
