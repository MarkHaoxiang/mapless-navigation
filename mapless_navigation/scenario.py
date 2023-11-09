from typing import List, Tuple
import numpy as np

import torch
from vmas.simulator.core import Agent, Landmark, World, Sphere, Line, Color
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import AGENT_OBS_TYPE, AGENT_REWARD_TYPE, ScenarioUtils

from mapless_navigation.maze import Maze, Direction, Room

class Scenario(BaseScenario):
    def make_world(self,
                   batch_dim: int,
                   device: torch.device,
                   robot_radius: float = 0.05,
                   world_size: float = 3,
                   **kwargs) -> World:
        self.world_size = world_size
        self.robot_radius = robot_radius
        minimum_gap_size = robot_radius * 5
        # Initialise World
        world = World(
            batch_dim=batch_dim,
            device=device
        )

        # Add Goal
        goal = Landmark(
            name="goal",
            collide=False
        )

        # Initialise Agent
        agent = Agent(
            "agent",
            shape=Sphere(radius=robot_radius),
            render_action=True,
            sensors = [Lidar(
                world=world,
                n_rays=12,
                max_range=0.35,
                entity_filter = lambda x: False if x == goal else True
            )]
        )
        agent.goal = goal

        # Build Map
        self.maze = Maze(
            w=world_size,
            h=world_size,
            resolution=4,
            minimum_gap_size=minimum_gap_size,
            minimum_room_length=minimum_gap_size*2
        )

        self.walls: List[Tuple[Landmark, Tuple[float, float], Direction]] = []
        for i, (x, y, dir, l) in enumerate(self.maze.walls):
            if dir == Direction.HORIZONTAL:
                x += l / 2
            else:
                y += l / 2
            self.walls.append(
                (   # Wall Object
                    Landmark(
                        name=f"wall {i}",
                        collide=True,
                        shape= Line(length=l),
                        color=Color.BLACK
                    ),
                    # Starting position
                    (x-self.world_size/2,y-self.world_size/2),
                    # Orientation
                    dir
                )
            )
        self.n_walls = len(self.walls)
        self._leaf_weights = np.array([room.area for room in self.maze.leaves])
        self._leaf_weights = self._leaf_weights / self._leaf_weights.sum()

        # Register entities
        world.add_agent(agent)
        world.add_landmark(goal)
        for wall, _, _ in self.walls:
            world.add_landmark(wall)

        self.goal = goal
        self.agent = agent
        return world

    def reward(self, agent: Agent) -> AGENT_REWARD_TYPE:
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1
        )
        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius
        return agent.on_goal
    
    def reset_world_at(self, env_index: int = None):
        # Spawn Walls
        for wall, pos, orientation in self.walls:
            wall.set_pos(
                torch.tensor(
                    [
                       pos[0], pos[1]
                    ],
                    dtype=torch.float32,
                    device=self.world.device
                ),
                batch_index=env_index
            )
            if orientation == Direction.VERTICAL:
                # Vertical Down
                wall.set_rot(
                    torch.tensor(3 * torch.pi / 2, dtype=torch.float32, device=self.world.device),
                    batch_index=env_index
                )
            else:
                # Horizontal Right
                wall.set_rot(
                    torch.tensor(0, dtype=torch.float32, device=self.world.device),
                    batch_index=env_index
                )
        
        # Spawn agent and goal
        self.goal.set_pos(
            self._random_room_position(distance_from_edge=self.robot_radius),
            batch_index=env_index
        )
        self.agent.set_pos(
            self._random_room_position(distance_from_edge=self.robot_radius),
            batch_index=env_index
        )

    def observation(self, agent: Agent) -> AGENT_OBS_TYPE:
        agent.sensors[0].measure()
        return torch.cat(
            [agent.state.pos,agent.state.vel],
            dim=-1
        )

    def _random_room_position(self, distance_from_edge: float = 0) -> torch.tensor:
        # Choose a random room
        room: Room = np.random.choice(self.maze.leaves, p=self._leaf_weights)
        bounding_box = (
            room.offset_x + distance_from_edge,
            room.offset_x+room.w - distance_from_edge,
            room.offset_y + distance_from_edge,
            room.offset_y+room.h - distance_from_edge
        )
        assert bounding_box[0] <= bounding_box[1] and bounding_box[2] <= bounding_box[3], "Invalid room dimensions"

        # Choose a random position within the room
        result = torch.rand(size=(2,)).to(device=self.world.device)
        result[0] = result[0] * (bounding_box[1]-bounding_box[0]) + bounding_box[0]
        result[1] = result[1] * (bounding_box[3]-bounding_box[2]) + bounding_box[2]
        result -= self.world_size / 2
        return result
