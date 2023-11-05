from typing import List, Tuple
import random

import torch
from vmas.simulator.core import Agent, Landmark, World, Sphere, Line, Color
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import AGENT_OBS_TYPE, AGENT_REWARD_TYPE, ScenarioUtils

from mapless_navigation.maze import Maze

class Scenario(BaseScenario):
    def make_world(self,
                   batch_dim: int,
                   device: torch.device,
                   robot_radius: float = 0.05,
                   world_size: float = 3,
                   **kwargs) -> World:
        minimum_gap_size = robot_radius * 5

        # Initialise World
        world = World(
            batch_dim=batch_dim,
            device=device
        )

        # Initialise Agent
        agent = Agent(
            "agent",
            shape=Sphere(radius=robot_radius*5),
            render_action=True,
            sensors = [Lidar(
                world=world,
                n_rays=12,
                max_range=0.35
            )]
        )
        world.add_agent(agent)
        # Build Map
        maze = Maze(
            w=world_size,
            h=world_size,
            resolution=4,
            minimum_gap_size=minimum_gap_size,
            minimum_room_length=minimum_gap_size*2
        )

        walls = []
            # Outer walls
        for i in range(4):
            vertical = i % 2 == 1
            walls.append(
                (
                    Landmark(
                        name = f"Wall {i}",
                        collide=True,
                        shape=Line(length=world_size),
                        color=Color.BLACK
                    ),
                    ((i % 2)*world_size/2*(i-2), (1-(i % 2))*world_size/2*(i-1)),
                    vertical
                )
            )        
        
            # Inner walls
        def build_wall(maze: Maze, wall_count = 0):
            if maze.has_division:
                # Start wall
                wall_length = maze.opening_offset
                x = maze.offset_x 
                y = maze.offset_y 
                if maze.cut_division_start:
                    wall_length = wall_length - minimum_gap_size
                    if maze.has_vertical_division:
                        y += minimum_gap_size
                    else:
                        x += minimum_gap_size
                if maze.has_horizontal_division:
                    y += maze.division_offset
                    x += wall_length / 2
                else:
                    x += maze.division_offset
                    y += wall_length / 2
                if wall_length > 0:
                    walls.append(
                        (   # Wall Object
                            Landmark(
                                name=f"wall {wall_count}",
                                collide=True,
                                shape= Line(length=wall_length),
                                color=Color.BLACK
                            ),
                            # Starting position
                            (x-world_size/2, y-world_size/2),
                            # Orientation
                            maze.division_direction
                        )
                    )
                    wall_count += 1
                # End wall
                wall_length = maze.w if maze.has_horizontal_division else maze.h
                wall_length = wall_length - maze.opening_offset - maze.opening_length
                if maze.cut_division_end:
                    wall_length -= minimum_gap_size
                x = maze.offset_x 
                y = maze.offset_y 
                if maze.has_horizontal_division:
                    y += maze.division_offset
                    x += maze.opening_offset + maze.opening_length + wall_length / 2
                else:
                    x += maze.division_offset
                    y += maze.opening_offset + maze.opening_length + wall_length / 2
                if wall_length > 0:
                    walls.append(
                        (   # Wall Object
                            Landmark(
                                name=f"wall {wall_count}",
                                collide=True,
                                shape= Line(length=wall_length),
                                color=Color.BLACK
                            ),
                            # Starting position
                            (x-world_size/2,y-world_size/2),
                            # Orientation
                            maze.division_direction
                        )
                    )
                    wall_count += 1

                # Recursive Call
                wall_count = build_wall(maze.region_1, wall_count=wall_count)
                wall_count = build_wall(maze.region_2, wall_count=wall_count)

            return wall_count

        self.n_walls = build_wall(maze, wall_count=4)
        self.walls: List[Tuple[Landmark, Tuple[float, float], bool]] = walls
        for wall, _, _ in self.walls:
            world.add_landmark(wall)

        # Add Goal
        goal = Landmark(
            name="goal",
            collide=False
        )

        agent.goal = goal
        world.add_landmark(goal)

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
            if orientation:
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

        # Spawn Agent and Goal
        # TODO (Collision Avoidance)
        # ScenarioUtils.spawn_entities_randomly(
        #     entities=self.world.agents + self.world.landmarks,
        #     world=self.world,
        #     env_index=env_index,
        #     min_dist_between_entities=0.3,
        #     x_bounds=(-1, 1),
        #     y_bounds=(-1, 1)
        # )

    def observation(self, agent: Agent) -> AGENT_OBS_TYPE:
        return torch.cat([
            agent.state.pos,
            agent.state.vel
        ])