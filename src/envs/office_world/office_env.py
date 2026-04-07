# Code adapted from src/envs/letter_world/letter_env.py

import pickle
import random
from typing import Any, Literal

import numpy as np
import gymnasium as gym
import pygame
from gymnasium import spaces
from gymnasium.core import ObsType, ActType, RenderFrame

from ltl.logic import Assignment


MAP_HEIGHT = 13
MAP_WIDTH = 17

# Binary wall map: 1 = wall, 0 = free space
# Letters in the labeled map below are placed on free (0) cells
#
#   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#   1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1
#   1 0 B 0 0 0 n 0 0 0 n 0 0 0 C 0 1
#   1 0 0 0 1 F 0 0 1 0 0 0 1 0 0 0 1
#   1 1 0 1 1 1 0 1 1 1 0 1 1 1 0 1 1
#   1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1
#   1 0 n 0 1 0 G 0 1 0 E 0 1 0 n 0 1
#   1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1
#   1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1
#   1 0 0 0 1 0 0 0 1 0 0 F 1 0 0 0 1
#   1 0 A 0 0 0 n 0 0 0 n 0 0 0 D 0 1
#   1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1
#   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
MAP = (
    "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n"
    "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n"
    "1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1\n"
    "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n"
    "1 1 0 1 1 1 0 1 1 1 0 1 1 1 0 1 1\n"
    "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n"
    "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n"
    "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n"
    "1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1\n"
    "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n"
    "1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1\n"
    "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n"
    "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
)

# Fixed label positions used for evaluation (use_fixed_map=True).
# Keys are (row, col), values are lowercase proposition letters.
FIXED_LABEL_MAP: dict[tuple[int, int], str] = {
    (10, 2): 'a',
    (2, 2): 'b',
    (2, 14): 'c',
    (10, 14): 'd',
    (6, 10): 'e',
    (3, 5): 'f',
    (9, 11): 'f',
    (6, 6): 'g',
    (2, 6): 'n',
    (2, 10): 'n',
    (6, 2): 'n',
    (6, 14): 'n',
    (10, 6): 'n',
    (10, 10): 'n',
}

# Ordered list of label positions; index i matches LETTERS[i].
LABEL_POSITIONS: list[tuple[int, int]] = list(FIXED_LABEL_MAP.keys())

# One character per label position — used to assign letters when shuffling.
# "abcdeffgnnnnnn" (14 characters for 14 positions)
LETTERS: str = ''.join(FIXED_LABEL_MAP[pos] for pos in LABEL_POSITIONS)

# Agent starts to the right of 'a' at (10, 2) in the fixed map.
FIXED_AGENT_START: tuple[int, int] = (10, 3)


def _parse_grid(map_str: str) -> list[list[int]]:
    return [[int(x) for x in row.split()] for row in map_str.strip().split('\n')]


class OfficeWorldEnv(gym.Env):
    """
    Office world grid environment with a fixed wall structure and labeled regions.

    Each letter represents a proposition. The agent moves in 4 directions and is
    blocked by walls (no wrapping). The active proposition is the letter at the
    agent's current cell, if any.

    When use_fixed_map=False (training), letter positions are randomly shuffled
    among the 14 label locations and the agent is placed on a random free cell.
    When use_fixed_map=True (evaluation), positions are fixed as in FIXED_LABEL_MAP
    and the agent starts to the right of 'a'.
    """
    metadata = {'render_modes': ['human', 'rgb_array', 'path'], 'render_fps': 10}

    def __init__(
            self,
            use_fixed_map: bool,
            render_mode: str | None = None,
            map: dict[tuple[int, int], str] | None = None,
    ):
        self.render_mode = render_mode
        self.use_fixed_map = use_fixed_map
        self.letters = LETTERS
        self.letter_types = sorted(set(self.letters))  # ['a','b','c','d','e','f','g','n']

        self.grid = _parse_grid(MAP)

        self.free_cells: list[tuple[int, int]] = [
            (i, j)
            for i in range(MAP_HEIGHT)
            for j in range(MAP_WIDTH)
            if self.grid[i][j] == 0
        ]

        # Precompute wall channel for the observation
        self._wall_obs = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.uint8)
        for i in range(MAP_HEIGHT):
            for j in range(MAP_WIDTH):
                if self.grid[i][j] == 1:
                    self._wall_obs[i, j] = 1

        # Actions: up, down, left, right
        self.action_space = spaces.Discrete(4)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Observation: (H, W, num_letter_types + 1 agent + 1 walls)
        num_channels = len(self.letter_types) + 2
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(MAP_HEIGHT, MAP_WIDTH, num_channels),
            dtype=np.uint8,
        )

        self.map: dict[tuple[int, int], str] = map if map is not None else {}
        self.agent: tuple[int, int] = FIXED_AGENT_START

        self.rng = random.Random()

        if render_mode is not None:
            self.renderer = OfficeWorldRenderer(
                render_mode=render_mode,
                render_fps=self.metadata['render_fps'],
            )

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        di, dj = self.actions[action]
        new_i = self.agent[0] + di
        new_j = self.agent[1] + dj

        if (0 <= new_i < MAP_HEIGHT and 0 <= new_j < MAP_WIDTH
                and self.grid[new_i][new_j] == 0):
            self.agent = (new_i, new_j)

        if self.render_mode == 'human':
            self._render_frame()

        return self._get_observation(), 0.0, False, False, {'propositions': self.get_active_propositions()}

    def _get_observation(self) -> ObsType:
        obs = np.zeros((MAP_HEIGHT, MAP_WIDTH, len(self.letter_types) + 2), dtype=np.uint8)

        for loc, letter in self.map.items():
            obs[loc[0], loc[1], self.letter_types.index(letter)] = 1

        obs[self.agent[0], self.agent[1], len(self.letter_types)] = 1
        obs[:, :, len(self.letter_types) + 1] = self._wall_obs

        return obs

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        if seed is not None:
            self.rng = random.Random(seed)

        if self.use_fixed_map:
            self.map = dict(FIXED_LABEL_MAP)
            self.agent = FIXED_AGENT_START
        else:
            positions = list(LABEL_POSITIONS)
            self.rng.shuffle(positions)
            self.map = {positions[i]: self.letters[i] for i in range(len(self.letters))}

            available = [p for p in self.free_cells if p not in self.map]
            self.agent = self.rng.choice(available)

        return self._get_observation(), {'propositions': set()}

    def get_active_propositions(self) -> set[str]:
        if self.agent in self.map:
            return {self.map[self.agent]}
        return set()

    def get_propositions(self) -> list[str]:
        return self.letter_types

    def get_possible_assignments(self) -> list[Assignment]:
        return Assignment.zero_or_one_propositions(set(self.get_propositions()))

    def save_world_info(self, path: str):
        with open(path, 'wb+') as f:
            pickle.dump(self.map, f)

    def load_world_info(self, path: str):
        with open(path, 'rb') as f:
            self.map = pickle.load(f)
        self.use_fixed_map = True

    def print(self):
        print('+' + '-' * MAP_WIDTH + '+')
        for i in range(MAP_HEIGHT):
            line = '|'
            for j in range(MAP_WIDTH):
                if (i, j) == self.agent:
                    line += 'A'
                elif self.grid[i][j] == 1:
                    line += '#'
                elif (i, j) in self.map:
                    line += self.map[(i, j)]
                else:
                    line += ' '
            print(line + '|')
        print('+' + '-' * MAP_WIDTH + '+')
        print('Active:', self.get_active_propositions())

    def print_features(self):
        obs = self._get_observation()
        print('+' + '-' * MAP_WIDTH + '+')
        for i in range(MAP_HEIGHT):
            line = '|'
            for j in range(MAP_WIDTH):
                if np.max(obs[i, j, :-1]) > 0:  # exclude wall channel
                    line += str(np.argmax(obs[i, j, :-1]))
                else:
                    line += ' '
            print(line + '|')
        print('+' + '-' * MAP_WIDTH + '+')

    def wait_for_input(self) -> bool:
        if self.render_mode not in ('human', 'path'):
            return False
        return self.renderer.wait_for_input()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == 'rgb_array':
            return self._render_frame()

    def render_path(self, actions: list[int]):
        return self.renderer.render_path(self, [self.actions[a] for a in actions])

    def _render_frame(self):
        return self.renderer.render(self)

    def close(self):
        if self.render_mode is not None:
            self.renderer.close()


class OfficeWorldRenderer:
    def __init__(
            self,
            render_mode: Literal['human', 'path', 'rgb_array'] = 'human',
            render_fps: int = 10,
            cell_size: int = 60,
    ):
        self.cell_size = cell_size
        self.screen_width = MAP_WIDTH * cell_size
        self.screen_height = MAP_HEIGHT * cell_size
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.grid = _parse_grid(MAP)

        if render_mode in ('human', 'path'):
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('OfficeWorld')
            self.clock = pygame.time.Clock()

        self.font = pygame.font.Font(None, int(cell_size * 0.75))
        self.bg_color = (240, 240, 240)
        self.wall_color = (50, 50, 50)
        self.cell_color = (200, 200, 200)
        self.agent_color = (0, 128, 0)
        self.border_color = (100, 100, 100)
        self.arrow_color = (47, 79, 79, 160)

    def render(self, env: OfficeWorldEnv):
        canvas = self.draw_canvas(env)
        return self.update(canvas)

    def draw_canvas(self, env: OfficeWorldEnv):
        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill(self.bg_color)

        for i in range(MAP_HEIGHT):
            for j in range(MAP_WIDTH):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                if self.grid[i][j] == 1:
                    pygame.draw.rect(canvas, self.wall_color, rect)
                elif (i, j) == env.agent:
                    pygame.draw.rect(canvas, self.agent_color, rect)
                    pygame.draw.rect(canvas, self.border_color, rect, 1)
                    if (i, j) in env.map:
                        text = self.font.render(env.map[(i, j)].upper(), True, (255, 255, 0))
                        canvas.blit(text, text.get_rect(center=rect.center))
                elif (i, j) in env.map:
                    pygame.draw.rect(canvas, self.cell_color, rect)
                    pygame.draw.rect(canvas, self.border_color, rect, 1)
                    text = self.font.render(env.map[(i, j)].upper(), True, (0, 0, 0))
                    canvas.blit(text, text.get_rect(center=rect.center))
                else:
                    pygame.draw.rect(canvas, self.bg_color, rect)
                    pygame.draw.rect(canvas, self.border_color, rect, 1)

        return canvas

    def update(self, canvas):
        if self.render_mode in ('human', 'path'):
            self.screen.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.render_fps)
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def render_path(self, env: OfficeWorldEnv, actions: list[tuple[int, int]]):
        canvas = self.draw_canvas(env)
        path_canvas = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        path_canvas.fill((0, 0, 0, 0))

        current = env.agent
        for di, dj in actions:
            ni = current[0] + di
            nj = current[1] + dj
            if (0 <= ni < MAP_HEIGHT and 0 <= nj < MAP_WIDTH
                    and self.grid[ni][nj] == 0):
                next_pos = (ni, nj)
            else:
                next_pos = current
            self._draw_arrow(path_canvas, current, next_pos)
            current = next_pos

        canvas.blit(path_canvas, (0, 0))
        return self.update(canvas)

    def _draw_arrow(self, canvas, start: tuple[int, int], end: tuple[int, int]):
        if start == end:
            return
        sx = start[1] * self.cell_size + self.cell_size // 2
        sy = start[0] * self.cell_size + self.cell_size // 2
        ex = end[1] * self.cell_size + self.cell_size // 2
        ey = end[0] * self.cell_size + self.cell_size // 2

        pygame.draw.line(canvas, self.arrow_color, (sx, sy), (ex, ey), 5)
        angle = np.arctan2(ey - sy, ex - sx)
        arrowhead_size = 10
        arrowhead_angle = np.pi / 6
        arrowhead_points = [
            (ex, ey),
            (ex - arrowhead_size * np.cos(angle - arrowhead_angle),
             ey - arrowhead_size * np.sin(angle - arrowhead_angle)),
            (ex - arrowhead_size * np.cos(angle + arrowhead_angle),
             ey - arrowhead_size * np.sin(angle + arrowhead_angle)),
        ]
        pygame.draw.polygon(canvas, self.arrow_color, arrowhead_points)

    def wait_for_input(self) -> bool:
        pygame.event.clear()
        while True:
            event = pygame.event.wait()
            if event.type == pygame.KEYDOWN:
                return event.key == pygame.K_q

    def close(self):
        if self.render_mode in ('human', 'path'):
            pygame.display.quit()
            pygame.quit()


def main():
    str_to_action = {'w': 0, 's': 1, 'a': 2, 'd': 3}

    env = OfficeWorldEnv(use_fixed_map=True)
    env.reset()
    while True:
        env.print()
        print('\nAction? (wasd, q to quit): ', end='')
        a = input().strip()
        if a == 'q':
            break
        if a in str_to_action:
            obs, reward, term, trunc, info = env.step(str_to_action[a])
            print('Active propositions:', info['propositions'])
        else:
            print('Unknown action')


if __name__ == '__main__':
    main()
