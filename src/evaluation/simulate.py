import os
import random

import imageio
import numpy as np
import torch
from tqdm import tqdm

from envs import make_env
from ltl import FixedSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence.search import ExhaustiveSearch
from utils.model_store import ModelStore
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str,
                        choices=['PointLtl2-v0', 'LetterEnv-v0', 'FlatWorld-v0', 'SafetyBallNav-v0', 'RepoMan-v0'],
                        default='PointLtl2-v0')
    parser.add_argument('--exp', type=str, default='deepset')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--formula', type=str, default='(F blue) & (!blue U (green & F yellow))')
    parser.add_argument('--finite', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--render', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--randomize_agent', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--randomize_objects', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save_gifs', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--gif_dir', type=str, default='gifs')
    args = parser.parse_args()
    print(f"randomize_agent: {args.randomize_agent}, randomize_objects: {args.randomize_objects}")
    if args.env == 'LetterEnv-v0':
        gamma = 0.94
    elif args.env in ('PointLtl2-v0', 'SafetyBallNav-v0'):
        gamma = 0.998
    elif args.env == 'RepoMan-v0':
        gamma = 0.98
    else:
        gamma = 0.98
    return simulate(args.env, gamma, args.exp, args.seed, args.num_episodes, args.formula, args.finite, args.render,
                    args.deterministic, args.randomize_agent, args.randomize_objects,
                    save_gifs=args.save_gifs, gif_dir=args.gif_dir)


def simulate(env, gamma, exp, seed, num_episodes, formula, finite, render, deterministic,
             randomize_agent=True, randomize_objects=True, save_gifs=False, gif_dir='gifs'):
    env_name = env
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    if save_gifs:
        render_mode = 'rgb_array'
    elif render:
        render_mode = 'human'
    else:
        render_mode = None

    sampler = FixedSampler.partial(formula)
    env = make_env(env_name, sampler, render_mode=render_mode,
                   randomize_agent=randomize_agent, randomize_objects=randomize_objects)
    config = model_configs[env_name]
    model_store = ModelStore(env_name, exp, seed)
    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()
    model = build_model(env, training_status, config)

    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=render)

    if save_gifs:
        gif_out = os.path.join(gif_dir, env_name, exp, f'seed{seed}')
        os.makedirs(gif_out, exist_ok=True)

    num_successes = 0
    num_violations = 0
    num_accepting_visits = 0
    steps = []
    rets = []

    env.reset(seed=seed)

    pbar = range(num_episodes)
    if not render:
        pbar = tqdm(pbar)
    for i in pbar:
        obs, info = env.reset(), {}
        if render:
            print(obs['goal'])
        agent.reset()
        done = False
        num_steps = 0
        frames = []

        if save_gifs:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        while not done:
            action = agent.get_action(obs, info, deterministic=deterministic)
            action = action.flatten()
            if action.shape == (1,):
                action = action[0]
            obs, reward, done, info = env.step(action)
            num_steps += 1

            if save_gifs:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            if done:
                if finite:
                    final_reward = int('success' in info)
                    if 'success' in info:
                        num_successes += 1
                        steps.append(num_steps)
                    elif 'violation' in info:
                        num_violations += 1
                    rets.append(final_reward * gamma ** (num_steps - 1))
                    if not render:
                        pbar.set_postfix({
                            'S': num_successes / (i + 1),
                            'V': num_violations / (i + 1),
                            'ADR': np.mean(rets),
                            'AS': np.mean(steps),
                        })
                else:
                    num_accepting_visits += info['num_accepting_visits']
                    if not render:
                        pbar.set_postfix({
                            'A': num_accepting_visits / (i + 1),
                        })

        if save_gifs and frames:
            outcome = 'success' if 'success' in info else ('violation' if 'violation' in info else 'timeout')
            gif_path = os.path.join(gif_out, f'ep{i:04d}_{outcome}.gif')
            imageio.mimsave(gif_path, frames, fps=5)

    env.close()
    if finite:
        success_rate = num_successes / num_episodes
        violation_rate = num_violations / num_episodes
        average_steps = np.mean(steps)
        adr = np.mean(rets)
        print(f'{seed}: {success_rate:.3f},{violation_rate:.3f},{adr:.3f},{average_steps:.3f}')
        return success_rate, average_steps
    else:
        average_visits = num_accepting_visits / num_episodes
        print(f'{seed}: {average_visits:.3f}')
        return average_visits


if __name__ == '__main__':
    main()
