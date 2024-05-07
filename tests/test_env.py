import pytest

import torch
from vmas import make_env

from mapless_navigation.maze import Maze
from mapless_navigation.scenario import Scenario

torch.manual_seed(0)
DEVICE = "cpu"


class TestMaze:
    @pytest.mark.parametrize("resolution", [1, 2, 4])
    @pytest.mark.parametrize("w", [50, 100])
    @pytest.mark.parametrize("h", [50, 100])
    @pytest.mark.parametrize("minimum_room_length", [4, 8])
    @pytest.mark.parametrize("minimum_gap_size", [1, 2])
    def test_maze_creation(
        self,
        resolution: int,
        w: float,
        h: float,
        minimum_room_length: float,
        minimum_gap_size: float,
    ):
        maze = Maze(w, h, resolution, minimum_room_length, minimum_gap_size)
        assert len(maze.rooms) > 0, "No rooms found"
        maze.visualise()


class TestEnv:
    def test_env_rollout(self):
        env = make_env(Scenario(), num_envs=1, device="cpu", seed=1)
        env.reset()

        for _ in range(100):
            actions = {}
            for i, agent in enumerate(env.agents):
                action = torch.tensor(
                    list(x.tolist() for x in env.action_space.sample()),
                    dtype=torch.float32,
                )
                action = torch.tensor([[0, 1]])
                actions.update({agent.name: action})
            env.step(actions)
