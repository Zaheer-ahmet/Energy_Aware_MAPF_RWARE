"""
2D rendering of the Robotic's Warehouse
environment using pyglet
"""

import math
import os
import sys

from gymnasium import error
import numpy as np
import six

from rware.warehouse import Direction

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite


try:
    import pyglet
except ImportError:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )


RAD2DEG = 57.29577951308232
# # Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)
_ORANGE = (255, 165, 0)
_DARKORANGE = (255, 140, 0)
_DARKSLATEBLUE = (72, 61, 139)
_TEAL = (0, 128, 128)
_GRAY = (128, 128, 128)

_BACKGROUND_COLOR = _WHITE
_GRID_COLOR = _BLACK
_SHELF_COLOR = _DARKSLATEBLUE
_SHELF_REQ_COLOR = _TEAL
_AGENT_COLOR = _DARKORANGE
_AGENT_LOADED_COLOR = _RED
_AGENT_DIR_COLOR = _BLACK
_GOAL_COLOR = (60, 60, 60)
_CHARGING_STATION_COLOR = _GREEN

_SHELF_PADDING = 2
_REWARD_BOX_COLOR = (0, 0, 0, 255) # Solid black
_REWARD_TEXT_COLOR = (255, 255, 255, 255) # White


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object):
    def __init__(self, world_size):
        display = get_display(None)
        self.rows, self.cols = world_size

        self.grid_size = 36
        self.icon_size = 20

        margin = 20
        top_margin = 90  # Increased top margin for counters
        self.margin = margin
        self.top_margin = top_margin
        self.width = 1 + self.cols * (self.grid_size + 1) + 2 * margin
        self.height = 2 + self.rows * (self.grid_size + 1) + (self.grid_size * 1.5) + margin + (top_margin - margin)
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=display
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.agent_paths = None

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def set_agent_paths(self, agent_paths):
        self.agent_paths = agent_paths

    def render(self, env, return_rgb_array=False):
        glClearColor(*_BACKGROUND_COLOR, 0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_goals(env)
        self._draw_shelfs(env)
        self._draw_charging_stations(env)
        self._draw_agents(env)
        self._draw_global_timer(env)
        self._draw_reward_list(env)
        if self.agent_paths is not None:
            self._draw_agent_paths(env)

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen

    def _draw_grid(self):
        batch = pyglet.graphics.Batch()
        # HORIZONTAL LINES
        for r in range(self.rows + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        self.margin,  # LEFT X
                        (self.grid_size + 1) * r + 1 + self.margin,  # Y
                        (self.grid_size + 1) * self.cols + self.margin,  # RIGHT X
                        (self.grid_size + 1) * r + 1 + self.margin,  # Y
                    ),
                ),
                ("c3B", (*_GRID_COLOR, *_GRID_COLOR)),
            )

        # VERTICAL LINES
        for c in range(self.cols + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * c + 1 + self.margin,  # X
                        self.margin,  # BOTTOM Y
                        (self.grid_size + 1) * c + 1 + self.margin,  # X
                        (self.grid_size + 1) * self.rows + self.margin,  # TOP Y
                    ),
                ),
                ("c3B", (*_GRID_COLOR, *_GRID_COLOR)),
            )
        batch.draw()

    def _draw_shelfs(self, env):
        batch = pyglet.graphics.Batch()

        for shelf in env.shelfs:
            x, y = shelf.x, shelf.y
            y = self.rows - y - 1  # pyglet rendering is reversed
            shelf_color = _GRAY  # All shelves are gray obstacles

            batch.add(
                4,
                gl.GL_QUADS,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * x + _SHELF_PADDING + 1 + self.margin,  # TL - X
                        (self.grid_size + 1) * y + _SHELF_PADDING + 1 + self.margin,  # TL - Y
                        (self.grid_size + 1) * (x + 1) - _SHELF_PADDING + self.margin,  # TR - X
                        (self.grid_size + 1) * y + _SHELF_PADDING + 1 + self.margin,  # TR - Y
                        (self.grid_size + 1) * (x + 1) - _SHELF_PADDING + self.margin,  # BR - X
                        (self.grid_size + 1) * (y + 1) - _SHELF_PADDING + self.margin,  # BR - Y
                        (self.grid_size + 1) * x + _SHELF_PADDING + 1 + self.margin,  # BL - X
                        (self.grid_size + 1) * (y + 1) - _SHELF_PADDING + self.margin,  # BL - Y
                    ),
                ),
                ("c3B", 4 * shelf_color),
            )
        batch.draw()

    def _draw_goals(self, env):
        batch = pyglet.graphics.Batch()

        # draw goal boxes
        for goal in env.goals:
            x, y = goal
            y = self.rows - y - 1  # pyglet rendering is reversed
            batch.add(
                4,
                gl.GL_QUADS,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * x + 1 + self.margin,  # TL - X
                        (self.grid_size + 1) * y + 1 + self.margin,  # TL - Y
                        (self.grid_size + 1) * (x + 1) + self.margin,  # TR - X
                        (self.grid_size + 1) * y + 1 + self.margin,  # TR - Y
                        (self.grid_size + 1) * (x + 1) + self.margin,  # BR - X
                        (self.grid_size + 1) * (y + 1) + self.margin,  # BR - Y
                        (self.grid_size + 1) * x + 1 + self.margin,  # BL - X
                        (self.grid_size + 1) * (y + 1) + self.margin,  # BL - Y
                    ),
                ),
                ("c3B", 4 * _GOAL_COLOR),
            )
        batch.draw()

        # draw goal labels
        for goal in env.goals:
            x, y = goal
            y = self.rows - y - 1
            label_x = x * (self.grid_size + 1) + (1 / 2) * (self.grid_size + 1) + self.margin
            label_y = (self.grid_size + 1) * y + (1 / 2) * (self.grid_size + 1) + self.margin
            label = pyglet.text.Label(
                "G",
                font_name="Calibri",
                font_size=18,
                bold=False,
                x=label_x,
                y=label_y,
                anchor_x="center",
                anchor_y="center",
                color=(*_WHITE, 255),
            )
            label.draw()

    def _draw_charging_stations(self, env):
        batch = pyglet.graphics.Batch()
        label_params = []
        for cs in getattr(env, 'charging_stations', []):
            if isinstance(cs, tuple):
                x, y = cs[0], cs[1]
            else:
                x, y = cs.x, cs.y
            y = self.rows - y - 1  # pyglet rendering is reversed
            batch.add(
                4,
                gl.GL_QUADS,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * x + 1 + self.margin,  # TL - X
                        (self.grid_size + 1) * y + 1 + self.margin,  # TL - Y
                        (self.grid_size + 1) * (x + 1) + self.margin,  # TR - X
                        (self.grid_size + 1) * y + 1 + self.margin,  # TR - Y
                        (self.grid_size + 1) * (x + 1) + self.margin,  # BR - X
                        (self.grid_size + 1) * (y + 1) + self.margin,  # BR - Y
                        (self.grid_size + 1) * x + 1 + self.margin,  # BL - X
                        (self.grid_size + 1) * (y + 1) + self.margin,  # BL - Y
                    ),
                ),
                ("c3B", 4 * _CHARGING_STATION_COLOR),
            )
            # Store label parameters to draw after batch
            label_x = x * (self.grid_size + 1) + (1 / 2) * (self.grid_size + 1) + self.margin
            label_y = (self.grid_size + 1) * y + (1 / 2) * (self.grid_size + 1) + self.margin
            label_params.append((label_x, label_y))
        batch.draw()
        # Draw all 'C' labels after the green boxes
        for label_x, label_y in label_params:
            label = pyglet.text.Label(
                "C",
                font_name="Calibri",
                font_size=18,
                bold=False,
                x=label_x,
                y=label_y,
                anchor_x="center",
                anchor_y="center",
                color=(*_WHITE, 255),
            )
            label.draw()

    def _draw_battery_timer(self, env, i, agent, x, y):
        # Draw a small battery timer box above the agent
        box_width = self.grid_size // 1.5
        box_height = self.grid_size // 3
        margin = 2
        
        # Position above the agent
        timer_x = x - box_width // 2
        timer_y = y + self.grid_size // 2 + margin
        
        # Draw background (semi-transparent black)
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
            ('v2f', [
                timer_x, timer_y,
                timer_x + box_width, timer_y,
                timer_x + box_width, timer_y + box_height,
                timer_x, timer_y + box_height
            ]),
            ('c4B', [0, 0, 0, 200] * 4)
        )
        
        # Draw border (white)
        pyglet.graphics.draw(4, pyglet.gl.GL_LINE_LOOP,
            ('v2f', [
                timer_x, timer_y,
                timer_x + box_width, timer_y,
                timer_x + box_width, timer_y + box_height,
                timer_x, timer_y + box_height
            ]),
            ('c3B', [255, 255, 255] * 4)
        )
        
        # Draw battery count
        battery_count = env._agent_batteries[i] if hasattr(env, '_agent_batteries') else agent.battery
        battery_label = pyglet.text.Label(
            str(battery_count),
            font_name="Arial",
            font_size=int(self.grid_size * 0.25),
            bold=True,
            x=timer_x + box_width // 2,
            y=timer_y + box_height // 2,
            anchor_x="center",
            anchor_y="center",
            color=(255, 255, 255, 255)
        )
        battery_label.draw()

    def _draw_reward_counter(self, reward, x, y):
        # Draw a wider reward counter box below the agent
        box_width = self.grid_size * 0.7  # Wider box
        box_height = self.grid_size * 0.3
        margin = 2
        # Position below the agent
        counter_x = x - box_width // 2
        counter_y = y - self.grid_size // 2 - box_height - margin
        # Draw background
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
            ('v2f', [
                counter_x, counter_y,
                counter_x + box_width, counter_y,
                counter_x + box_width, counter_y + box_height,
                counter_x, counter_y + box_height
            ]),
            ('c4B', _REWARD_BOX_COLOR * 4)
        )
        # Draw border (white)
        pyglet.graphics.draw(4, pyglet.gl.GL_LINE_LOOP,
            ('v2f', [
                counter_x, counter_y,
                counter_x + box_width, counter_y,
                counter_x + box_width, counter_y + box_height,
                counter_x, counter_y + box_height
            ]),
            ('c3B', [255, 255, 255] * 4)
        )
        # Draw reward value (formatted to 1 decimal place)
        reward_str = f"{reward:.1f}"
        reward_label = pyglet.text.Label(
            reward_str,
            font_name="Arial",
            font_size=int(self.grid_size * 0.25),
            bold=True,
            x=counter_x + box_width // 2,
            y=counter_y + box_height // 2,
            anchor_x="center",
            anchor_y="center",
            color=_REWARD_TEXT_COLOR
        )
        reward_label.draw()

    def _draw_agents(self, env):
        agents = []
        batch = pyglet.graphics.Batch()

        radius = self.grid_size / 3
        resolution = 6

        # Retrieve accumulated rewards, default to list of 0s if not set
        accumulated_rewards = getattr(env, '_agent_accumulated_rewards', [0.0] * len(env.agents))
        # Retrieve failed move info, default to all False
        failed_moves = getattr(env, '_agent_failed_moves', [False] * len(env.agents))
        # Retrieve failed flash frame counters, default to 0
        failed_flash_frames = getattr(env, '_agent_failed_flash_frames', [0] * len(env.agents))

        # Draw agent bodies
        for i, agent in enumerate(env.agents):
            col, row = agent.x, agent.y
            row = self.rows - row - 1  # pyglet rendering is reversed

            # Calculate center position with margin
            center_x = (self.grid_size + 1) * col + self.grid_size // 2 + 1 + self.margin
            center_y = (self.grid_size + 1) * row + self.grid_size // 2 + 1 + self.margin

            # make a circle
            verts = []
            for res_idx in range(resolution):
                angle = 2 * math.pi * res_idx / resolution
                x = radius * math.cos(angle) + center_x
                y = radius * math.sin(angle) + center_y
                verts += [x, y]
            circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))

            # If failed move, flash red for several frames
            if failed_flash_frames[i] > 0:
                draw_color = _RED
                glColor3ub(*draw_color)
                circle.draw(GL_POLYGON)
                # Decrement the flash frame counter
                failed_flash_frames[i] -= 1
            else:
                draw_color = _AGENT_LOADED_COLOR if agent.carrying_shelf else _AGENT_COLOR
                glColor3ub(*draw_color)
                circle.draw(GL_POLYGON)

            # Draw battery timer for this agent (above)
            self._draw_battery_timer(env, i, agent, center_x, center_y)

            # Draw reward counter for this agent (below)
            current_agent_reward = accumulated_rewards[i] # agent.id is 1-based, use index i
            self._draw_reward_counter(current_agent_reward, center_x, center_y)

            # Draw agent index at bottom right of agent (no circle)
            badge_offset_x = radius * 0.7
            badge_offset_y = -radius * 0.7
            badge_x = center_x + badge_offset_x
            badge_y = center_y + badge_offset_y
            self._draw_agent_index_number(i, badge_x, badge_y)

        # Draw agent direction lines
        for agent in env.agents:
            col, row = agent.x, agent.y
            row = self.rows - row - 1  # pyglet rendering is reversed

            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * col
                        + self.grid_size // 2
                        + 1 + self.margin,  # CENTER X
                        (self.grid_size + 1) * row
                        + self.grid_size // 2
                        + 1 + self.margin,  # CENTER Y
                        (self.grid_size + 1) * col
                        + self.grid_size // 2
                        + 1
                        + (
                            radius if agent.dir.value == Direction.RIGHT.value else 0
                        )  # DIR X
                        + (
                            -radius if agent.dir.value == Direction.LEFT.value else 0
                        ) + self.margin,  # DIR X
                        (self.grid_size + 1) * row
                        + self.grid_size // 2
                        + 1
                        + (
                            radius if agent.dir.value == Direction.UP.value else 0
                        )  # DIR Y
                        + (
                            -radius if agent.dir.value == Direction.DOWN.value else 0
                        ) + self.margin,  # DIR Y
                    ),
                ),
                ("c3B", (*_AGENT_DIR_COLOR, *_AGENT_DIR_COLOR)),
            )
        batch.draw()

    def _draw_global_timer(self, env):
        # Draw episode counter above step counter in the upper left
        episode = getattr(env, '_episode', 1)
        episode_str = f"Episode: {episode}"
        step_str = f"Step: {env._cur_steps}"
        global_step = getattr(env, '_global_step', 0)
        global_step_str = f"Global Step: {global_step}"
        # Episode label
        episode_label = pyglet.text.Label(
            episode_str,
            font_name="Arial",
            font_size=14,
            bold=True,
            x=self.margin + 5,
            y=self.height - 10,
            anchor_x="left",
            anchor_y="top",
            color=(0, 0, 0, 255),
        )
        episode_label.draw()
        # Step label
        step_label = pyglet.text.Label(
            step_str,
            font_name="Arial",
            font_size=14,
            x=self.margin + 5,
            y=self.height - 35,
            anchor_x="left",
            anchor_y="top",
            color=(0, 0, 0, 255),
        )
        step_label.draw()
        # Global step label (below step label)
        global_step_label = pyglet.text.Label(
            global_step_str,
            font_name="Arial",
            font_size=14,
            x=self.margin + 5,
            y=self.height - 60,
            anchor_x="left",
            anchor_y="top",
            color=(0, 0, 0, 255),
        )
        global_step_label.draw()

    def _draw_reward_list(self, env):
        # Get the accumulated rewards that are already being tracked
        rewards = getattr(env.unwrapped, '_agent_accumulated_rewards', [])
        team_reward = getattr(env.unwrapped, '_team_reward', None)
        
        # Position for the list (top right corner)
        list_x = self.width - 170  # Adjust x position as needed
        list_y = self.height - 10   # Start y position
        line_height = 20

        # Draw team reward at the top, if available
        current_y = list_y
        if team_reward is not None:
            team_label_text = f"Team Reward: {team_reward:.1f}"
            team_label = pyglet.text.Label(
                team_label_text,
                font_name='Arial',
                font_size=12,
                bold=True,
                x=list_x,
                y=current_y,
                anchor_x='left',
                anchor_y='top',
                color=(0, 0, 0, 255)  # Black text
            )
            team_label.draw()
            current_y -= line_height + 5  # Add extra padding after team reward

        # Draw individual agent rewards below team reward
        for i, reward in enumerate(rewards):
            label_text = f"Agent {i+1}: {reward:.1f}"
            label = pyglet.text.Label(
                label_text,
                font_name='Arial',
                font_size=12,
                x=list_x,
                y=current_y - i * line_height,
                anchor_x='left',
                anchor_y='top',
                color=(0, 0, 0, 255)  # Black text
            )
            label.draw()

    def _draw_badge(self, row, col, index):
        resolution = 6
        radius = self.grid_size / 5

        badge_x = col * (self.grid_size + 1) + (3 / 4) * (self.grid_size + 1)
        badge_y = (
            self.height
            - (self.grid_size + 1) * (row + 1)
            + (1 / 4) * (self.grid_size + 1)
        )

        # make a circle
        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + badge_x
            y = radius * math.sin(angle) + badge_y
            verts += [x, y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_WHITE)
        circle.draw(GL_POLYGON)
        glColor3ub(*_BLACK)
        circle.draw(GL_LINE_LOOP)
        label = pyglet.text.Label(
            str(index),
            font_name="Times New Roman",
            font_size=9,
            bold=True,
            x=badge_x,
            y=badge_y + 2,
            anchor_x="center",
            anchor_y="center",
            color=(*_BLACK, 255),
        )
        label.draw()

    def _draw_agent_paths(self, env):
        # Color palette for agent paths
        path_colors = [
            (255, 0, 0),    # Red
            (0, 128, 255),  # Blue
            (0, 200, 0),    # Green
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]
        for i, path in enumerate(self.agent_paths):
            color = path_colors[i % len(path_colors)]
            for (x, y) in path:
                sx = (self.grid_size + 1) * x + self.grid_size // 2 + 1
                sy = (self.grid_size + 1) * (self.rows - y - 1) + self.grid_size // 2 + 1
                # Draw a small filled circle at (sx, sy)
                num_segments = 12
                radius = self.grid_size // 10
                verts = []
                for j in range(num_segments):
                    angle = 2 * math.pi * j / num_segments
                    verts.append(sx + radius * math.cos(angle))
                    verts.append(sy + radius * math.sin(angle))
                pyglet.graphics.draw(num_segments, pyglet.gl.GL_POLYGON, ('v2f', verts), ('c3B', color * num_segments))

    def _draw_agent_index_number(self, index, x, y):
        # Draw just the agent index number (starting from 0) at the given position
        label = pyglet.text.Label(
            str(index),
            font_name="Arial",
            font_size=int(self.grid_size * 0.22),
            bold=True,
            x=x,
            y=y,
            anchor_x="center",
            anchor_y="center",
            color=(0, 0, 0, 255)
        )
        label.draw()
