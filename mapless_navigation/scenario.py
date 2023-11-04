import torch
from vmas.simulator.core import Agent, Landmark, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import AGENT_OBS_TYPE, AGENT_REWARD_TYPE, ScenarioUtils

class Scenario(BaseScenario):
    def make_world(self,
                   batch_dim: int,
                   device: torch.device,
                   **kwargs) -> World:
        # Initialise World
        world = World(
            batch_dim=batch_dim,
            device=device
        )

        # Initialise Agent
        agent = Agent(
            "agent",
            render_action=True,
            sensors = [Lidar(
                world=world,
                n_rays=12,
                max_range=0.35 
            )]
        )
        world.add_agent(agent)

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
        ScenarioUtils.spawn_entities_randomly(
            entities=self.world.agents + self.world.landmarks,
            world=self.world,
            env_index=env_index,
            min_dist_between_entities=0.3,
            x_bounds=(-1, 1),
            y_bounds=(-1, 1)
        )
    
    def observation(self, agent: Agent) -> AGENT_OBS_TYPE:
        return torch.cat([
            agent.state.pos,
            agent.state.vel
        ])