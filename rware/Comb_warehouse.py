from collections import OrderedDict
from enum import Enum
from typing import List, Tuple, Optional, Dict

import gymnasium as gym
from gymnasium.utils import seeding
import networkx as nx
import numpy as np


_COLLISION_LAYERS = 2

_LAYER_AGENTS = 0
_LAYER_SHELFS = 1


class _VectorWriter:
    def __init__(self, size: int):
        self.vector = np.zeros(size, dtype=np.float32)
        self.idx = 0

    def write(self, data):
        data_size = len(data)
        self.vector[self.idx : self.idx + data_size] = data
        self.idx += data_size

    def skip(self, bits):
        self.idx += bits


class Action(Enum):
    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class RewardType(Enum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2


class ObservationType(Enum):
    DICT = 0
    FLATTENED = 1
    IMAGE = 2
    IMAGE_DICT = 3


class ImageLayer(Enum):
    """
    Input layers of image-style observations
    """

    SHELVES = 0  # binary layer indicating shelves (also indicates carried shelves)
    REQUESTS = 1  # binary layer indicating requested shelves
    AGENTS = 2  # binary layer indicating agents in the environment (no way to distinguish agents)
    AGENT_DIRECTION = 3  # layer indicating agent directions as int (see Direction enum + 1 for values)
    AGENT_LOAD = 4  # binary layer indicating agents with load
    GOALS = 5  # binary layer indicating goal/ delivery locations
    ACCESSIBLE = 6  # binary layer indicating accessible cells (all but occupied cells/ out of map)


class Entity:
    def __init__(self, id_: int, x: int, y: int):
        self.id = id_
        self.prev_x = None
        self.prev_y = None
        self.x = x
        self.y = y


class Agent(Entity):
    counter = 0

    def __init__(self, x: int, y: int, dir_: Direction, msg_bits: int, battery_max: int = 10):
        Agent.counter += 1
        super().__init__(Agent.counter, x, y)
        self.dir = dir_
        self.message = np.zeros(msg_bits)
        self.req_action: Optional[Action] = None
        self.carrying_shelf: Optional[Shelf] = None
        self.canceled_action = None
        self.has_delivered = False
        self.step_count = 0  # Track number of steps taken
        self.battery_max = battery_max  # Add battery_max attribute
        self.battery = self.battery_max  # Initialize battery to battery_max

    @property
    def collision_layers(self):
        if self.loaded:
            return (_LAYER_AGENTS, _LAYER_SHELFS)
        else:
            return (_LAYER_AGENTS,)

    def req_location(self, grid_size) -> Tuple[int, int]:
        if self.req_action != Action.FORWARD:
            return self.x, self.y
        elif self.dir == Direction.UP:
            return self.x, max(0, self.y - 1)
        elif self.dir == Direction.DOWN:
            return self.x, min(grid_size[0] - 1, self.y + 1)
        elif self.dir == Direction.LEFT:
            return max(0, self.x - 1), self.y
        elif self.dir == Direction.RIGHT:
            return min(grid_size[1] - 1, self.x + 1), self.y

        raise ValueError(
            f"Direction is {self.dir}. Should be one of {[v for v in Direction]}"
        )

    def req_direction(self) -> Direction:
        wraplist = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        if self.req_action == Action.RIGHT:
            return wraplist[(wraplist.index(self.dir) + 1) % len(wraplist)]
        elif self.req_action == Action.LEFT:
            return wraplist[(wraplist.index(self.dir) - 1) % len(wraplist)]
        else:
            return self.dir

    def update_battery(self, amount=1.0):
        if self.battery > 0:
            self.battery = max(0, self.battery - amount)
            
    def recharge_battery(self):
        self.battery = self.battery_max  # Reset battery to full (battery_max)


class Shelf(Entity):
    counter = 0

    def __init__(self, x, y):
        Shelf.counter += 1
        super().__init__(Shelf.counter, x, y)

    @property
    def collision_layers(self):
        return (_LAYER_SHELFS,)


class ChargingStation(Entity):
    counter = 0
    def __init__(self, x, y):
        ChargingStation.counter += 1
        super().__init__(ChargingStation.counter, x, y)


class Warehouse(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        shelf_columns: int,
        column_height: int,
        shelf_rows: int,
        n_agents: int,
        msg_bits: int,
        sensor_range: int,
        request_queue_size: int,
        max_inactivity_steps: Optional[int],
        max_steps: Optional[int],
        reward_type: RewardType,
        layout: Optional[str] = None,
        observation_type: ObservationType = ObservationType.FLATTENED,
        image_observation_layers: List[ImageLayer] = [
            ImageLayer.SHELVES,
            ImageLayer.REQUESTS,
            ImageLayer.AGENTS,
            ImageLayer.GOALS,
            ImageLayer.ACCESSIBLE,
        ],
        image_observation_directional: bool = True,
        normalised_coordinates: bool = False,
        render_mode: str = "human",
        battery_max: int = 10,  # Add battery_max argument
    ):
        """The robotic warehouse environment

        Creates a grid world where multiple agents (robots)
        are supposed to collect shelfs, bring them to a goal
        and then return them.
        .. note:
            The grid looks like this:

            shelf
            columns
                vv
            ----------
            -XX-XX-XX-        ^
            -XX-XX-XX-  Column Height
            -XX-XX-XX-        v
            ----------
            -XX----XX-   <\
            -XX----XX-   <- Shelf Rows
            -XX----XX-   </
            ----------
            ----GG----

            G: is the goal positions where agents are rewarded if
            they bring the correct shelfs.

            The final grid size will be
            height: (column_height + 1) * shelf_rows + 2
            width: (2 + 1) * shelf_columns + 1

            The bottom-middle column will be removed to allow for
            robot queuing next to the goal locations

        :param shelf_columns: Number of columns in the warehouse
        :type shelf_columns: int
        :param column_height: Column height in the warehouse
        :type column_height: int
        :param shelf_rows: Number of columns in the warehouse
        :type shelf_rows: int
        :param n_agents: Number of spawned and controlled agents
        :type n_agents: int
        :param msg_bits: Number of communication bits for each agent
        :type msg_bits: int
        :param sensor_range: Range of each agents observation
        :type sensor_range: int
        :param request_queue_size: How many shelfs are simultaneously requested
        :type request_queue_size: int
        :param max_inactivity: Number of steps without a delivered shelf until environment finishes
        :type max_inactivity: Optional[int]
        :param reward_type: Specifies if agents are rewarded individually or globally
        :type reward_type: RewardType
        :param layout: A string for a custom warehouse layout. X are shelve locations, dots are corridors, and g are the goal locations. Ignores shelf_columns, shelf_height and shelf_rows when used.
        :type layout: str
        :param observation_type: Specifies type of observations
        :param image_observation_layers: Specifies types of layers observed if image-observations
            are used
        :type image_observation_layers: List[ImageLayer]
        :param image_observation_directional: Specifies whether image observations should be
            rotated to be directional (agent perspective) if image-observations are used
        :type image_observation_directional: bool
        :param normalised_coordinates: Specifies whether absolute coordinates should be normalised
            with respect to total warehouse size
        :type normalised_coordinates: bool
        """

        self.goals: List[Tuple[int, int]] = []

        self.n_agents = n_agents
        if not layout:
            self._make_layout_from_params(shelf_columns, shelf_rows, column_height)
        else:
            self._make_layout_from_str(layout)

        self.msg_bits = msg_bits
        self.sensor_range = sensor_range
        self.max_inactivity_steps: Optional[int] = max_inactivity_steps
        self.reward_type = reward_type
        self.reward_range = (0, 1)

        self._cur_inactive_steps = None
        self._cur_steps = 0
        self.max_steps = max_steps

        self.normalised_coordinates = normalised_coordinates

        sa_action_space = [len(Action), *msg_bits * (2,)]
        if len(sa_action_space) == 1:
            sa_action_space = gym.spaces.Discrete(sa_action_space[0])
        else:
            sa_action_space = gym.spaces.MultiDiscrete(sa_action_space)
        self.action_space = gym.spaces.Tuple(tuple(n_agents * [sa_action_space]))

        self.request_queue_size = request_queue_size
        self.request_queue = []

        self.battery_max = battery_max  # Store battery_max in environment
        self.agents: List[Agent] = []

        # default values:
        self.fast_obs = None
        self.image_obs = None
        self.image_dict_obs = None
        if observation_type == ObservationType.IMAGE:
            self.observation_space = self._use_image_obs(
                image_observation_layers, image_observation_directional
            )
        elif observation_type == ObservationType.IMAGE_DICT:
            self.observation_space = self._use_image_dict_obs(
                image_observation_layers, image_observation_directional
            )

        else:
            # used for DICT observation type and needed as preceeding stype to generate
            # FLATTENED observations as well
            self.observation_space = self._use_slow_obs()

            # for performance reasons we
            # can flatten the obs vector
            if observation_type == ObservationType.FLATTENED:
                self.observation_space = self._use_fast_obs()

        self.global_image = None
        self.renderer = None
        self.render_mode = render_mode

    def _make_layout_from_params(self, shelf_columns, shelf_rows, column_height):
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"

        self.grid_size = (
            (column_height + 1) * shelf_rows + 2,
            (2 + 1) * shelf_columns + 1,
        )
        self.column_height = column_height
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        self.goals = [
            (self.grid_size[1] // 2 - 1, self.grid_size[0] - 1),
            (self.grid_size[1] // 2, self.grid_size[0] - 1),
        ]

        self.highways = np.zeros(self.grid_size, dtype=np.uint8)

        def highway_func(x, y):
            is_on_vertical_highway = x % 3 == 0
            is_on_horizontal_highway = y % (column_height + 1) == 0
            is_on_delivery_row = y == self.grid_size[0] - 1
            is_on_queue = (y > self.grid_size[0] - (column_height + 3)) and (
                x == self.grid_size[1] // 2 - 1 or x == self.grid_size[1] // 2
            )
            return (
                is_on_vertical_highway
                or is_on_horizontal_highway
                or is_on_delivery_row
                or is_on_queue
            )

        for x in range(self.grid_size[1]):
            for y in range(self.grid_size[0]):
                self.highways[y, x] = int(highway_func(x, y))

        # Add charging stations at (4,4) and (5,4)
        if not hasattr(self, 'charging_stations'):
            self.charging_stations = []
        self.charging_stations.append(ChargingStation(4, 4))
        self.charging_stations.append(ChargingStation(5, 4))

    def _make_layout_from_str(self, layout):
        layout = layout.strip()
        layout = layout.replace(" ", "")
        grid_height = layout.count("\n") + 1
        lines = layout.split("\n")
        grid_width = len(lines[0])
        for line in lines:
            assert len(line) == grid_width, "Layout must be rectangular"

        self.grid_size = (grid_height, grid_width)
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        self.highways = np.zeros(self.grid_size, dtype=np.uint8)
        self.charging_stations = []  # List of ChargingStation objects

        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                assert char.lower() in "gx.c", "Unknown character in layout: {}".format(char)
                if char.lower() == "g":
                    self.goals.append((x, y))
                    self.highways[y, x] = 1
                elif char.lower() == ".":
                    self.highways[y, x] = 1
                elif char == "C":
                    self.charging_stations.append(ChargingStation(x, y))
                    self.highways[y, x] = 1  # Charging stations are accessible

        assert len(self.goals) >= 1, "At least one goal is required"
        # New: Check that number of goals matches number of agents
        if hasattr(self, 'n_agents') and len(self.goals) != self.n_agents:
            raise ValueError(f"Number of goals in layout ({len(self.goals)}) does not match n_agents ({self.n_agents}). Please update the layout or n_agents.")

    def _use_image_obs(self, image_observation_layers, directional=True):
        """
        Set image observation space
        :param image_observation_layers (List[ImageLayer]): list of layers to use as image channels
        :param directional (bool): flag whether observations should be directional (pointing in
            direction of agent or north-wise)
        """
        self.image_obs = True
        self.fast_obs = False
        self.image_dict_obs = True
        self.image_observation_directional = directional
        self.image_observation_layers = image_observation_layers

        observation_shape = (1 + 2 * self.sensor_range, 1 + 2 * self.sensor_range)

        layers_min = []
        layers_max = []
        for layer in image_observation_layers:
            if layer == ImageLayer.AGENT_DIRECTION:
                # directions as int
                layer_min = np.zeros(observation_shape, dtype=np.float32)
                layer_max = np.ones(observation_shape, dtype=np.float32) * max(
                    [d.value + 1 for d in Direction]
                )
            else:
                # binary layer
                layer_min = np.zeros(observation_shape, dtype=np.float32)
                layer_max = np.ones(observation_shape, dtype=np.float32)
            layers_min.append(layer_min)
            layers_max.append(layer_max)

        # total observation
        min_obs = np.stack(layers_min)
        max_obs = np.stack(layers_max)
        return gym.spaces.Tuple(
            tuple([gym.spaces.Box(min_obs, max_obs, dtype=np.float32)] * self.n_agents)
        )

    def _use_image_dict_obs(self, image_observation_layers, directional=True):
        """
        Get image dictionary observation with image and flattened feature vector
        :param image_observation_layers (List[ImageLayer]): list of layers to use as image channels
        :param directional (bool): flag whether observations should be directional (pointing in
            direction of agent or north-wise)
        """
        image_obs_space = self._use_image_obs(image_observation_layers, directional)[0]
        self.image_obs = False
        self.image_dict_obs = True
        feature_space = gym.spaces.Dict(
            OrderedDict(
                {
                    "direction": gym.spaces.Discrete(4),
                    "on_highway": gym.spaces.MultiBinary(1),
                    "carrying_shelf": gym.spaces.MultiBinary(1),
                }
            )
        )

        feature_flat_dim = gym.spaces.flatdim(feature_space)
        feature_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(feature_flat_dim,),
            dtype=np.float32,
        )

        return gym.spaces.Tuple(
            tuple(
                [
                    gym.spaces.Dict(
                        {"image": image_obs_space, "features": feature_space}
                    )
                    for _ in range(self.n_agents)
                ]
            )
        )

    def _use_slow_obs(self):
        self.fast_obs = False

        self._obs_bits_for_self = 4 + len(Direction)
        self._obs_bits_per_agent = 1 + len(Direction) + self.msg_bits
        self._obs_bits_per_shelf = 2
        self._obs_bits_for_requests = 2

        self._obs_sensor_locations = (1 + 2 * self.sensor_range) ** 2

        self._obs_length = (
            self._obs_bits_for_self
            + self._obs_sensor_locations * self._obs_bits_per_agent
            + self._obs_sensor_locations * self._obs_bits_per_shelf
        )

        max_grid_val = max(self.grid_size)
        low = np.zeros(2)
        if self.normalised_coordinates:
            high = np.ones(2)
            dtype = np.float32
        else:
            high = np.ones(2) * max_grid_val
            dtype = np.int32
        location_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(2,),
            dtype=dtype,
        )

        self_observation_dict_space = gym.spaces.Dict(
            OrderedDict(
                {
                    "location": location_space,
                    "carrying_shelf": gym.spaces.MultiBinary(1),
                    "direction": gym.spaces.Discrete(4),
                    "on_highway": gym.spaces.MultiBinary(1),
                }
            )
        )
        sensor_per_location_dict = OrderedDict(
            {
                "has_agent": gym.spaces.MultiBinary(1),
                "direction": gym.spaces.Discrete(4),
            }
        )
        if self.msg_bits > 0:
            sensor_per_location_dict["local_message"] = gym.spaces.MultiBinary(
                self.msg_bits
            )
        sensor_per_location_dict.update(
            {
                "has_shelf": gym.spaces.MultiBinary(1),
                "shelf_requested": gym.spaces.MultiBinary(1),
            }
        )
        return gym.spaces.Tuple(
            tuple(
                [
                    gym.spaces.Dict(
                        OrderedDict(
                            {
                                "self": self_observation_dict_space,
                                "sensors": gym.spaces.Tuple(
                                    self._obs_sensor_locations
                                    * (gym.spaces.Dict(sensor_per_location_dict),)
                                ),
                            }
                        )
                    )
                    for _ in range(self.n_agents)
                ]
            )
        )

    def _use_fast_obs(self):
        if self.fast_obs:
            return self.observation_space

        self.fast_obs = True
        ma_spaces = []
        for sa_obs in self.observation_space:
            flatdim = gym.spaces.flatdim(sa_obs)
            ma_spaces += [
                gym.spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        return gym.spaces.Tuple(tuple(ma_spaces))

    def _is_highway(self, x: int, y: int) -> bool:
        return self.highways[y, x]

    def _make_img_obs(self, agent):
        # write image observations
        if agent.id == 1:
            layers = []
            # first agent's observation --> update global observation layers
            for layer_type in self.image_observation_layers:
                if layer_type == ImageLayer.SHELVES:
                    layer = self.grid[_LAYER_SHELFS].copy().astype(np.float32)
                    layer[layer > 0.0] = 1.0
                elif layer_type == ImageLayer.REQUESTS:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for requested_shelf in self.request_queue:
                        layer[requested_shelf.y, requested_shelf.x] = 1.0
                elif layer_type == ImageLayer.AGENTS:
                    layer = self.grid[_LAYER_AGENTS].copy().astype(np.float32)
                    layer[layer > 0.0] = 1.0
                elif layer_type == ImageLayer.AGENT_DIRECTION:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        agent_direction = ag.dir.value + 1
                        layer[ag.x, ag.y] = float(agent_direction)
                elif layer_type == ImageLayer.AGENT_LOAD:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        if ag.carrying_shelf is not None:
                            layer[ag.x, ag.y] = 1.0
                elif layer_type == ImageLayer.GOALS:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for goal_y, goal_x in self.goals:
                        layer[goal_y, goal_x] = 1.0
                elif layer_type == ImageLayer.ACCESSIBLE:
                    layer = np.ones(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        layer[ag.y, ag.x] = 0.0
                else:
                    raise ValueError(f"Unknown image layer type: {layer_type}")
                layer = np.pad(layer, self.sensor_range, mode="constant")
                layers.append(layer)
            self.global_layers = np.stack(layers)
        # ... rest of function unchanged ...
        start_x = agent.y
        end_x = agent.y + 2 * self.sensor_range + 1
        start_y = agent.x
        end_y = agent.x + 2 * self.sensor_range + 1
        obs = self.global_layers[:, start_x:end_x, start_y:end_y]
        if self.image_observation_directional:
            if agent.dir == Direction.DOWN:
                obs = np.rot90(obs, k=2, axes=(1, 2))
            elif agent.dir == Direction.LEFT:
                obs = np.rot90(obs, k=3, axes=(1, 2))
            elif agent.dir == Direction.RIGHT:
                obs = np.rot90(obs, k=1, axes=(1, 2))
        return obs

    def _get_default_obs(self, agent):
        min_x = agent.x - self.sensor_range
        max_x = agent.x + self.sensor_range + 1

        min_y = agent.y - self.sensor_range
        max_y = agent.y + self.sensor_range + 1

        # sensors
        if (
            (min_x < 0)
            or (min_y < 0)
            or (max_x > self.grid_size[1])
            or (max_y > self.grid_size[0])
        ):
            padded_agents = np.pad(
                self.grid[_LAYER_AGENTS], self.sensor_range, mode="constant"
            )
            padded_shelfs = np.pad(
                self.grid[_LAYER_SHELFS], self.sensor_range, mode="constant"
            )
            # + self.sensor_range due to padding
            min_x += self.sensor_range
            max_x += self.sensor_range
            min_y += self.sensor_range
            max_y += self.sensor_range

        else:
            padded_agents = self.grid[_LAYER_AGENTS]
            padded_shelfs = self.grid[_LAYER_SHELFS]

        agents = padded_agents[min_y:max_y, min_x:max_x].reshape(-1)
        shelfs = padded_shelfs[min_y:max_y, min_x:max_x].reshape(-1)

        if self.fast_obs:
            # write flattened observations
            flatdim = gym.spaces.flatdim(self.observation_space[agent.id - 1])
            obs = _VectorWriter(flatdim)

            if self.normalised_coordinates:
                agent_x = agent.x / (self.grid_size[1] - 1)
                agent_y = agent.y / (self.grid_size[0] - 1)
            else:
                agent_x = agent.x
                agent_y = agent.y

            obs.write([agent_x, agent_y, int(agent.carrying_shelf is not None)])
            direction = np.zeros(4)
            direction[agent.dir.value] = 1.0
            obs.write(direction)
            obs.write([int(self._is_highway(agent.x, agent.y))])

            # 'has_agent': MultiBinary(1),
            # 'direction': Discrete(4),
            # 'local_message': MultiBinary(2)
            # 'has_shelf': MultiBinary(1),
            # 'shelf_requested': MultiBinary(1),

            for i, (id_agent, id_shelf) in enumerate(zip(agents, shelfs)):
                if id_agent == 0:
                    # no agent, direction, or message
                    obs.write([0.0])  # no agent present
                    obs.write([1.0, 0.0, 0.0, 0.0])  # agent direction
                    obs.skip(self.msg_bits)  # agent message
                else:
                    obs.write([1.0])  # agent present
                    direction = np.zeros(4)
                    direction[self.agents[id_agent - 1].dir.value] = 1.0
                    obs.write(direction)  # agent direction as onehot
                    if self.msg_bits > 0:
                        obs.write(self.agents[id_agent - 1].message)  # agent message
                if id_shelf == 0:
                    obs.write([0.0, 0.0])  # no shelf or requested shelf
                else:
                    obs.write(
                        [1.0, int(self.shelfs[id_shelf - 1] in self.request_queue)]
                    )  # shelf presence and request status
            return obs.vector

        # write dictionary observations
        obs = {}
        if self.normalised_coordinates:
            agent_x = agent.x / (self.grid_size[1] - 1)
            agent_y = agent.y / (self.grid_size[0] - 1)
        else:
            agent_x = agent.x
            agent_y = agent.y
        # --- self data
        obs["self"] = {
            "location": np.array([agent_x, agent_y], dtype=np.int32),
            "carrying_shelf": [int(agent.carrying_shelf is not None)],
            "direction": agent.dir.value,
            "on_highway": [int(self._is_highway(agent.x, agent.y))],
        }
        # --- sensor data
        obs["sensors"] = tuple({} for _ in range(self._obs_sensor_locations))

        # find neighboring agents
        for i, id_ in enumerate(agents):
            if id_ == 0:
                obs["sensors"][i]["has_agent"] = [0]
                obs["sensors"][i]["direction"] = 0
                obs["sensors"][i]["local_message"] = (
                    self.msg_bits * [0] if self.msg_bits > 0 else None
                )
            else:
                obs["sensors"][i]["has_agent"] = [1]
                obs["sensors"][i]["direction"] = self.agents[id_ - 1].dir.value
                obs["sensors"][i]["local_message"] = (
                    self.agents[id_ - 1].message if self.msg_bits > 0 else None
                )

        # find neighboring shelfs:
        for i, id_ in enumerate(shelfs):
            if id_ == 0:
                obs["sensors"][i]["has_shelf"] = [0]
                obs["sensors"][i]["shelf_requested"] = [0]
            else:
                obs["sensors"][i]["has_shelf"] = [1]
                obs["sensors"][i]["shelf_requested"] = [
                    int(self.shelfs[id_ - 1] in self.request_queue)
                ]

        return obs

    def _make_obs(self, agent):
        if self.image_obs:
            return self._make_img_obs(agent)
        elif self.image_dict_obs:
            image_obs = self._make_img_obs(agent)
            feature_obs = _VectorWriter(
                self.observation_space[agent.id - 1]["features"].shape[0]
            )
            direction = np.zeros(4)
            direction[agent.dir.value] = 1.0
            feature_obs.write(direction)
            feature_obs.write(
                [
                    int(self._is_highway(agent.x, agent.y)),
                    int(agent.carrying_shelf is not None),
                ]
            )
            return {
                "image": image_obs,
                "features": feature_obs.vector,
            }
        else:
            return self._get_default_obs(agent)

    def _get_info(self):
        return {}

    def _recalc_grid(self):
        self.grid[:] = 0
        for s in self.shelfs:
            self.grid[_LAYER_SHELFS, s.y, s.x] = s.id

        for a in self.agents:
            self.grid[_LAYER_AGENTS, a.y, a.x] = a.id

    def reset(self, seed=None, options=None):
        if seed is not None:
            # setting seed
            super().reset(seed=seed, options=options)

        Shelf.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0

        # --- Persist agent battery across episodes ---
        prev_batteries = [agent.battery for agent in self.agents] if hasattr(self, 'agents') and self.agents else None

        # make the shelfs
        self.shelfs = [
            Shelf(x, y)
            for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            )
            if not self._is_highway(x, y)
        ]

        # spawn agents at random locations
        agent_locs = self.np_random.choice(
            np.arange(self.grid_size[0] * self.grid_size[1]),
            size=self.n_agents,
            replace=False,
        )
        agent_locs = np.unravel_index(agent_locs, self.grid_size)
        # and direction
        agent_dirs = self.np_random.choice([d for d in Direction], size=self.n_agents)
        self.agents = [
            Agent(x, y, dir_, self.msg_bits, battery_max=self.battery_max)
            for y, x, dir_ in zip(*agent_locs, agent_dirs)
        ]
        # Restore previous battery values if available
        if prev_batteries is not None:
            for agent, prev_battery in zip(self.agents, prev_batteries):
                agent.battery = prev_battery
        # Reset step counts
        for agent in self.agents:
            agent.step_count = 0

        self._recalc_grid()

        self.request_queue = list(
            self.np_random.choice(
                self.shelfs, size=self.request_queue_size, replace=False
            )
        )

        return tuple([self._make_obs(agent) for agent in self.agents]), self._get_info()

    def step(self, actions):
        # --- Custom movement/collision logic: block moves into agents/obstacles ---
        DIRECTIONS = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        current_positions = {(agent.x, agent.y) for agent in self.agents}
        obstacle_cells = set()
        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                if not self._is_highway(x, y):
                    obstacle_cells.add((x, y))
        intended_positions = []
        failed_move = [False] * len(self.agents)
        for i, agent in enumerate(self.agents):
            act = Action(actions[i])
            # If agent has completed, force NOOP and prevent any movement/turn
            if i in self.completed_agents:
                act = Action.NOOP
            agent.req_action = act
            # Battery depletion for FORWARD moves
            if act == Action.FORWARD:
                agent.update_battery(1.0)
            if act == Action.FORWARD:
                next_x, next_y = agent.req_location(self.grid_size)
            else:
                next_x, next_y = agent.x, agent.y
            intended_positions.append((next_x, next_y))
        from collections import Counter
        pos_counts = Counter(intended_positions)
        for i, agent in enumerate(self.agents):
            next_pos = intended_positions[i]
            if (next_pos in obstacle_cells or
                (next_pos in current_positions and next_pos != (agent.x, agent.y)) or
                pos_counts[next_pos] > 1):
                intended_positions[i] = (agent.x, agent.y)
                failed_move[i] = True
        for i, agent in enumerate(self.agents):
            act = Action(actions[i])
            # If agent has completed, force NOOP and prevent any movement/turn
            if i in self.completed_agents:
                act = Action.NOOP
            agent.prev_x, agent.prev_y = agent.x, agent.y
            agent.req_action = act
            if act == Action.FORWARD:
                agent.x, agent.y = intended_positions[i]
            elif act == Action.LEFT:
                idx = DIRECTIONS.index(agent.dir)
                agent.dir = DIRECTIONS[(idx - 1) % 4]
            elif act == Action.RIGHT:
                idx = DIRECTIONS.index(agent.dir)
                agent.dir = DIRECTIONS[(idx + 1) % 4]
            # Battery logic: recharge if on any charging station (after position update)
            if hasattr(self, 'charging_stations'):
                for cs in self.charging_stations:
                    if isinstance(cs, tuple):
                        cs_x, cs_y = cs[0], cs[1]
                    else:
                        cs_x, cs_y = cs.x, cs.y
                    if (agent.x, agent.y) == (cs_x, cs_y):
                        agent.recharge_battery()
                        break
        # Sync battery display for renderer
        self._agent_batteries = [agent.battery for agent in self.agents]  # Ensure renderer uses live battery values
        # --- Reward assignment ---
        rewards = np.zeros(self.n_agents)
        for i, agent in enumerate(self.agents):
            if (agent.x, agent.y) == self.goals[i] and i not in self.completed_agents:
                rewards[i] = 10.0
                self.completed_agents.add(i)
            elif i in self.completed_agents:
                rewards[i] = 0.0
            elif failed_move[i]:
                rewards[i] = -1.0
            else:
                rewards[i] = -0.1
        # --- Team reward logic ---
        if not self.team_completed:
            self.team_reward += -0.1  # Per-step penalty
            if len(self.completed_agents) == self.n_agents:
                self.team_reward += 50.0
                self.team_completed = True
        # --- Episode termination ---
        self._cur_steps += 1
        done = self.team_completed or (self._cur_steps >= self.max_steps)
        info = {}
        info['team_reward'] = self.team_reward
        info['team_completed'] = self.team_completed
        info['completed_agents'] = list(self.completed_agents)
        obs = tuple([self._make_obs(agent) for agent in self.agents])
        return obs, list(rewards), done, info

    def render(self):
        self._agent_batteries = [agent.battery for agent in self.agents]  # Always sync before rendering
        if self.render_mode != 'human':
            return  # Support headless mode
        if not self.renderer:
            from rware.rendering import Viewer
            self.renderer = Viewer(self.grid_size)
        return self.renderer.render(self, return_rgb_array=False)

    def close(self):
        if self.renderer:
            self.renderer.close()

    def seed(self, seed=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    def get_global_image(
        self,
        image_layers=[
            ImageLayer.SHELVES,
            ImageLayer.GOALS,
        ],
        recompute=False,
        pad_to_shape=None,
    ):
        """
        Get global image observation
        :param image_layers: image layers to include in global image
        :param recompute: bool whether image should be recomputed or taken from last computation
            (for default params, image will be constant for environment so no recomputation needed
             but if agent or request information is included, then should be recomputed)
         :param pad_to_shape: if given than pad environment global image shape into this
             shape (if doesn't fit throw exception)
        """
        if recompute or self.global_image is None:
            layers = []
            for layer_type in image_layers:
                if layer_type == ImageLayer.SHELVES:
                    layer = self.grid[_LAYER_SHELFS].copy().astype(np.float32)
                    # set all occupied shelf cells to 1.0 (instead of shelf ID)
                    layer[layer > 0.0] = 1.0
                elif layer_type == ImageLayer.REQUESTS:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for requested_shelf in self.request_queue:
                        layer[requested_shelf.y, requested_shelf.x] = 1.0
                elif layer_type == ImageLayer.AGENTS:
                    layer = self.grid[_LAYER_AGENTS].copy().astype(np.float32)
                    # set all occupied agent cells to 1.0 (instead of agent ID)
                    layer[layer > 0.0] = 1.0
                elif layer_type == ImageLayer.AGENT_DIRECTION:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        agent_direction = ag.dir.value + 1
                        layer[ag.x, ag.y] = float(agent_direction)
                elif layer_type == ImageLayer.AGENT_LOAD:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        if ag.carrying_shelf is not None:
                            layer[ag.x, ag.y] = 1.0
                elif layer_type == ImageLayer.GOALS:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for goal_y, goal_x in self.goals:
                        layer[goal_y, goal_x] = 1.0
                elif layer_type == ImageLayer.ACCESSIBLE:
                    layer = np.ones(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        layer[ag.y, ag.x] = 0.0
                else:
                    raise ValueError(f"Unknown image layer type: {layer_type}")
                layer = np.pad(layer, self.sensor_range, mode="constant")
                layers.append(layer)
            self.global_image = np.stack(layers)
            if pad_to_shape is not None:
                padding_dims = [
                    pad_dim - global_dim
                    for pad_dim, global_dim in zip(
                        pad_to_shape, self.global_image.shape
                    )
                ]
                assert all([dim >= 0 for dim in padding_dims])
                pad_before = [pad_dim // 2 for pad_dim in padding_dims]
                pad_after = [
                    pad_dim // 2 if pad_dim % 2 == 0 else pad_dim // 2 + 1
                    for pad_dim in padding_dims
                ]
                self.global_image = np.pad(
                    self.global_image,
                    pad_width=tuple(zip(pad_before, pad_after)),
                    mode="constant",
                    constant_values=0,
                )
        return self.global_image


# --- HRPF Team Warehouse ---
class HRPFTeamWarehouse(Warehouse):
    def __init__(self, *args, **kwargs):
        battery_max = kwargs.pop('battery_max', 10)  # Remove battery_max from kwargs if present
        super().__init__(*args, **kwargs, battery_max=battery_max)
        self.completed_agents = set()
        self.team_completed = False
        self.team_reward = 0.0
        self._initial_goals = None

        # --- Infer observation space from a sample observation ---
        import gymnasium as gym
        import numpy as np

        # Get a sample observation by resetting the environment
        sample = self.reset()
        if isinstance(sample, tuple):
            sample_obs = sample[0]
        else:
            sample_obs = sample
        if isinstance(sample_obs, (list, tuple)) and len(sample_obs) > 0:
            obs_shape = sample_obs[0].shape
            self.observation_space = gym.spaces.Tuple(
                tuple([
                    gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=obs_shape,
                        dtype=np.float32
                    ) for _ in range(self.n_agents)
                ])
            )
        else:
            raise RuntimeError("Could not infer observation space: sample_obs is empty or not a list.")

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.completed_agents = set()
        self.team_completed = False
        self.team_reward = 0.0
        self._initial_goals = list(self.goals)
        return obs, info

    def step(self, actions):
        # --- Custom movement/collision logic: block moves into agents/obstacles ---
        DIRECTIONS = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        current_positions = {(agent.x, agent.y) for agent in self.agents}
        obstacle_cells = set()
        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                if not self._is_highway(x, y):
                    obstacle_cells.add((x, y))
        intended_positions = []
        failed_move = [False] * len(self.agents)
        for i, agent in enumerate(self.agents):
            act = Action(actions[i])
            # If agent has completed, force NOOP and prevent any movement/turn
            if i in self.completed_agents:
                act = Action.NOOP
            agent.req_action = act
            # Battery depletion for FORWARD moves
            if act == Action.FORWARD:
                agent.update_battery(1.0)
            if act == Action.FORWARD:
                next_x, next_y = agent.req_location(self.grid_size)
            else:
                next_x, next_y = agent.x, agent.y
            intended_positions.append((next_x, next_y))
        from collections import Counter
        pos_counts = Counter(intended_positions)
        for i, agent in enumerate(self.agents):
            next_pos = intended_positions[i]
            if (next_pos in obstacle_cells or
                (next_pos in current_positions and next_pos != (agent.x, agent.y)) or
                pos_counts[next_pos] > 1):
                intended_positions[i] = (agent.x, agent.y)
                failed_move[i] = True
        for i, agent in enumerate(self.agents):
            act = Action(actions[i])
            # If agent has completed, force NOOP and prevent any movement/turn
            if i in self.completed_agents:
                act = Action.NOOP
            agent.prev_x, agent.prev_y = agent.x, agent.y
            agent.req_action = act
            if act == Action.FORWARD:
                agent.x, agent.y = intended_positions[i]
            elif act == Action.LEFT:
                idx = DIRECTIONS.index(agent.dir)
                agent.dir = DIRECTIONS[(idx - 1) % 4]
            elif act == Action.RIGHT:
                idx = DIRECTIONS.index(agent.dir)
                agent.dir = DIRECTIONS[(idx + 1) % 4]
            # Battery logic: recharge if on any charging station (after position update)
            if hasattr(self, 'charging_stations'):
                for cs in self.charging_stations:
                    if isinstance(cs, tuple):
                        cs_x, cs_y = cs[0], cs[1]
                    else:
                        cs_x, cs_y = cs.x, cs.y
                    if (agent.x, agent.y) == (cs_x, cs_y):
                        agent.recharge_battery()
                        break
        # --- Reward assignment ---
        rewards = np.zeros(self.n_agents)
        for i, agent in enumerate(self.agents):
            if (agent.x, agent.y) == self._initial_goals[i] and i not in self.completed_agents:
                rewards[i] = 10.0
                self.completed_agents.add(i)
            elif i in self.completed_agents:
                rewards[i] = 0.0
            elif failed_move[i]:
                rewards[i] = -1.0
            else:
                rewards[i] = -0.1
        # --- Team reward logic ---
        if not self.team_completed:
            self.team_reward += -0.1  # Per-step penalty
            if len(self.completed_agents) == self.n_agents:
                self.team_reward += 50.0
                self.team_completed = True
        # --- Episode termination ---
        self._cur_steps += 1
        done = self.team_completed or (self._cur_steps >= self.max_steps)
        info = {}
        info['team_reward'] = self.team_reward
        info['team_completed'] = self.team_completed
        info['completed_agents'] = list(self.completed_agents)
        obs = tuple([self._make_obs(agent) for agent in self.agents])
        return obs, list(rewards), done, info

    def _make_img_obs(self, agent):
        # write image observations
        if agent.id == 1:
            layers = []
            # first agent's observation --> update global observation layers
            for layer_type in self.image_observation_layers:
                if layer_type == ImageLayer.SHELVES:
                    layer = self.grid[_LAYER_SHELFS].copy().astype(np.float32)
                    layer[layer > 0.0] = 1.0
                elif layer_type == ImageLayer.REQUESTS:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for requested_shelf in self.request_queue:
                        layer[requested_shelf.y, requested_shelf.x] = 1.0
                elif layer_type == ImageLayer.AGENTS:
                    layer = self.grid[_LAYER_AGENTS].copy().astype(np.float32)
                    layer[layer > 0.0] = 1.0
                elif layer_type == ImageLayer.AGENT_DIRECTION:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        agent_direction = ag.dir.value + 1
                        layer[ag.x, ag.y] = float(agent_direction)
                elif layer_type == ImageLayer.AGENT_LOAD:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        if ag.carrying_shelf is not None:
                            layer[ag.x, ag.y] = 1.0
                elif layer_type == ImageLayer.GOALS:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for goal_y, goal_x in self._initial_goals:
                        layer[goal_y, goal_x] = 1.0
                elif layer_type == ImageLayer.ACCESSIBLE:
                    layer = np.ones(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        layer[ag.y, ag.x] = 0.0
                else:
                    raise ValueError(f"Unknown image layer type: {layer_type}")
                layer = np.pad(layer, self.sensor_range, mode="constant")
                layers.append(layer)
            self.global_layers = np.stack(layers)
        # ... rest of function unchanged ...
        start_x = agent.y
        end_x = agent.y + 2 * self.sensor_range + 1
        start_y = agent.x
        end_y = agent.x + 2 * self.sensor_range + 1
        obs = self.global_layers[:, start_x:end_x, start_y:end_y]
        if self.image_observation_directional:
            if agent.dir == Direction.DOWN:
                obs = np.rot90(obs, k=2, axes=(1, 2))
            elif agent.dir == Direction.LEFT:
                obs = np.rot90(obs, k=3, axes=(1, 2))
            elif agent.dir == Direction.RIGHT:
                obs = np.rot90(obs, k=1, axes=(1, 2))
        return obs

    def render(self):
        self._agent_batteries = [agent.battery for agent in self.agents]  # Always sync before rendering
        if self.render_mode != 'human':
            return  # Support headless mode
        if not self.renderer:
            from rware.rendering import Viewer
            self.renderer = Viewer(self.grid_size)
        return self.renderer.render(self, return_rgb_array=False)


if __name__ == "__main__":
    env = Warehouse(9, 8, 3, 10, 3, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()
    from tqdm import tqdm

    # env.render()

    for _ in tqdm(range(1000000)):
        # time.sleep(0.05)
        # env.render()
        actions = env.action_space.sample()
        env.step(actions)
