import os

import gymnasium
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete

main_dir = os.path.split(os.path.abspath(__file__))[0]
assets_dir = os.path.join(main_dir, 'assets')


def _load_image(name):
    fullname = os.path.join(assets_dir, name)
    try:
        image = pygame.image.load(fullname)
    except pygame.error:
        print('Cannot load image:', fullname)
        raise SystemExit(str(pygame.geterror()))
    image = image.convert_alpha()
    return image


def _calculate_topleft_position(position, sprite_size):
    return sprite_size * position[1], sprite_size * position[0]


class _Collectible(pygame.sprite.Sprite):
    _COLLECTIBLE_IMAGES = {
        ('square', 'purple'): 'purple_square.png',
        ('circle', 'purple'): 'purple_circle.png',
        ('square', 'beige'):  'beige_square.png',
        ('circle', 'beige'):  'beige_circle.png',
        ('square', 'blue'):   'blue_square.png',
        ('circle', 'blue'):   'blue_circle.png',
    }

    def __init__(self, sprite_size, shape, colour):
        pygame.sprite.Sprite.__init__(self)
        self.shape = shape
        self.colour = colour
        self._sprite_size = sprite_size
        image = _load_image(self._COLLECTIBLE_IMAGES[(shape, colour)])
        self.image = pygame.transform.scale(image, (sprite_size, sprite_size))
        self.rect = self.image.get_rect()
        self.position = None

    def reset(self, position):
        self.position = position
        self.rect.topleft = _calculate_topleft_position(position, self._sprite_size)

    @property
    def symbols(self):
        return {self.shape, self.colour}


class _Player(pygame.sprite.Sprite):
    def __init__(self, sprite_size):
        pygame.sprite.Sprite.__init__(self)
        self._sprite_size = sprite_size
        image = _load_image('character.png')
        self.image = pygame.transform.scale(image, (sprite_size, sprite_size))
        self.rect = self.image.get_rect()
        self.position = None

    def reset(self, position):
        self.position = position
        self.rect.topleft = _calculate_topleft_position(position, self._sprite_size)

    def move(self, direction):
        self.position = (self.position[0] + direction[0], self.position[1] + direction[1])
        self.rect.topleft = _calculate_topleft_position(self.position, self._sprite_size)


class CollectEnv(gymnasium.Env):
    """
    Grid-world environment where an agent navigates to collectible objects.

    Propositions are the individual color and shape names:
      ['beige', 'blue', 'purple', 'circle', 'square']

    Active propositions at any step are the shape + color of the collectible
    at the agent's current grid position (empty set if no collectible is there).

    Compatible with the deep-LTL training pipeline (SequenceWrapper, PPO).
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    _BOARDS = {
        'original': [
            '##########',
            '#        #',
            '#        #',
            '#    #   #',
            '#   ##   #',
            '#  ##    #',
            '#   #    #',
            '#        #',
            '#        #',
            '##########',
        ],
    }

    _AVAILABLE_COLLECTIBLES = [
        ('square', 'purple'),
        ('circle', 'purple'),
        ('square', 'beige'),
        ('circle', 'beige'),
        ('square', 'blue'),
        ('circle', 'blue'),
    ]

    _ACTIONS = {
        0: (-1, 0),  # North
        1: (0,  1),  # East
        2: ( 1, 0),  # South
        3: (0, -1),  # West
    }

    _SCREEN_SIZE = (400, 400)
    _SPRITE_SIZE = 40
    _GRID_SIZE = 10
    _DISPLAY_SCALE = 2  # multiply screen size for the human render window

    # Default fixed positions for each collectible (in _AVAILABLE_COLLECTIBLES order).
    # Any collectible without a listed position falls back to the next free space.
    _DEFAULT_OBJECT_POSITIONS = [
        (2, 7),  # square_purple
        (2, 5),  # circle_purple
        (3, 6),  # square_beige
        (2, 6),  # circle_beige
        (1, 6),  # square_blue
        # (8, 1) circle_blue — omitted, falls back to first unclaimed free space
    ]

    def __init__(self, board='original', render_mode=None,
                 randomize_agent=True, randomize_objects=True,
                 object_start_positions=None):
        print(f"randomize_agent: {randomize_agent}")
        print(f"randomize_objects: {randomize_objects}")
        super().__init__()
        self.render_mode = render_mode
        self._randomize_agent = randomize_agent
        self._randomize_objects = randomize_objects
        # object_start_positions overrides _DEFAULT_OBJECT_POSITIONS when randomize_objects=False
        self._object_start_positions = (
            object_start_positions if object_start_positions is not None
            else self._DEFAULT_OBJECT_POSITIONS
        )
        self.action_space = Discrete(4)

        # Flat observation: normalized (row, col) for player + each collectible
        # Shape: (2 + 2*num_collectibles,) = (14,)
        n_sprites = 1 + len(self._AVAILABLE_COLLECTIBLES)
        self.observation_space = Box(0.0, 1.0, shape=(2 * n_sprites,), dtype=np.float32)

        self.board = np.array([list(row) for row in self._BOARDS[board]])
        self.free_spaces = list(map(tuple, np.argwhere(self.board != '#')))

        pygame.init()
        pygame.display.init()
        pygame.display.set_mode((1, 1))
        self._display_window = None
        self._bestdepth = pygame.display.mode_ok(self._SCREEN_SIZE, 0, 32)
        self._surface = pygame.Surface(self._SCREEN_SIZE, 0, self._bestdepth)
        self._background = pygame.Surface(self._SCREEN_SIZE)
        self._build_board()

        self.player = _Player(self._SPRITE_SIZE)
        self.collectibles = [
            _Collectible(self._SPRITE_SIZE, shape, colour)
            for shape, colour in self._AVAILABLE_COLLECTIBLES
        ]
        self._label_map: dict[tuple, set] = {}

    def _build_board(self):
        for col in range(self.board.shape[1]):
            for row in range(self.board.shape[0]):
                position = _calculate_topleft_position((row, col), self._SPRITE_SIZE)
                image_name = 'wall.png' if self.board[row, col] == '#' else 'ground.png'
                image = _load_image(image_name)
                image = pygame.transform.scale(image, (self._SPRITE_SIZE, self._SPRITE_SIZE))
                self._background.blit(image, position)

    def _get_obs(self) -> np.ndarray:
        max_idx = float(self._GRID_SIZE - 1)
        coords = [
            self.player.position[0] / max_idx,
            self.player.position[1] / max_idx,
        ]
        for c in self.collectibles:
            coords.extend([c.position[0] / max_idx, c.position[1] / max_idx])
        return np.array(coords, dtype=np.float32)

    def _active_propositions(self) -> set:
        return self._label_map.get(self.player.position, set())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        n_obj = len(self.collectibles)

        if self._randomize_agent and self._randomize_objects:
            # All positions sampled jointly (no collisions guaranteed)
            indices = self.np_random.choice(len(self.free_spaces), size=1 + n_obj, replace=False)
            positions = [self.free_spaces[int(i)] for i in indices]
            player_pos = positions[0]
            obj_positions = positions[1:]
        elif self._randomize_objects:
            # Player fixed at free_spaces[0]; objects fill remaining spaces randomly
            player_pos = self.free_spaces[0]
            remaining = self.free_spaces[1:]
            indices = self.np_random.choice(len(remaining), size=n_obj, replace=False)
            obj_positions = [remaining[int(i)] for i in indices]
        elif self._randomize_agent:
            # Objects fixed at free_spaces[1..n_obj]; player fills a remaining space randomly
            obj_positions = self.free_spaces[1:n_obj + 1]
            remaining = [s for s in self.free_spaces if s not in obj_positions]
            idx = int(self.np_random.integers(len(remaining)))
            player_pos = remaining[idx]
        else:
            # Fully deterministic: use _object_start_positions, filling any gaps
            # with the first unclaimed free spaces
            obj_positions = list(self._object_start_positions[:n_obj])
            if len(obj_positions) < n_obj:
                claimed = set(obj_positions)
                fallbacks = [s for s in self.free_spaces if s not in claimed]
                obj_positions += fallbacks[:n_obj - len(obj_positions)]
            player_pos = self.free_spaces[0]

        self.player.reset(player_pos)
        self._label_map = {}
        for collectible, pos in zip(self.collectibles, obj_positions):
            collectible.reset(pos)
            self._label_map[pos] = collectible.symbols

        obs = self._get_obs()
        info = {'propositions': self._active_propositions()}
        return obs, info

    def step(self, action):
        direction = self._ACTIONS[int(action)]
        next_pos = (self.player.position[0] + direction[0],
                    self.player.position[1] + direction[1])
        if self.board[next_pos] != '#':
            self.player.move(direction)

        obs = self._get_obs()
        info = {'propositions': self._active_propositions()}
        return obs, 0.0, False, False, info

    # ------------------------------------------------------------------
    # Methods required by the deep-LTL training pipeline
    # ------------------------------------------------------------------

    def get_propositions(self) -> list[str]:
        """All atomic proposition names used in LTL tasks."""
        return ['beige', 'blue', 'purple', 'circle', 'square']

    def get_possible_assignments(self):
        """
        All valid proposition assignments that can occur in this environment:
          - empty set  (agent not on any collectible)
          - one pair per collectible type  (e.g. {blue, square})
        """
        from ltl.logic import Assignment
        props = set(self.get_propositions())
        assignments = [Assignment({p: False for p in props})]
        for shape, colour in self._AVAILABLE_COLLECTIBLES:
            true_props = {shape, colour}
            assignments.append(Assignment({p: (p in true_props) for p in props}))
        return assignments

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        self._surface.blit(self._background, (0, 0))
        render_group = pygame.sprite.RenderPlain()
        render_group.add(self.collectibles)
        render_group.add(self.player)
        render_group.draw(self._surface)
        if self.render_mode == 'human':
            display_w = self._SCREEN_SIZE[0] * self._DISPLAY_SCALE
            display_h = self._SCREEN_SIZE[1] * self._DISPLAY_SCALE
            if self._display_window is None:
                self._display_window = pygame.display.set_mode((display_w, display_h))
                pygame.display.set_caption('RepoMan')
            scaled = pygame.transform.scale(self._surface, (display_w, display_h))
            self._display_window.blit(scaled, (0, 0))
            pygame.display.flip()
        elif self.render_mode == 'rgb_array':
            return np.copy(pygame.surfarray.array3d(self._surface)).swapaxes(0, 1)
