import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import pygame
import random
from monte_carlo_agent import MonteCarloAgent
from q_learning_agent import QLearningAgent
from actor_critic_agent import ActorCriticAgent
from settings import AlgorithmSettings, SettingsUI
from visualization import MetricsVisualizer


class BasurahanMazeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size=(6, 6), cell_size=80, settings=None):
        super(BasurahanMazeEnv, self).__init__()

        self.grid_size = grid_size
        self.cell_size = cell_size
        self.settings = settings

        self.action_space = spaces.Discrete(4)  # 0=Up, 1=Right, 2=Down, 3=Left
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(grid_size[0], grid_size[1], 3),
                                            dtype=np.int32)

        # Environment elements
        self.WALL = -1
        self.TRASH = 1
        self.MINE = -2
        self.EXIT = 9

        # Initialize map
        self._build_maze()

        # Evil robot patrol state
        self.evil_dir_order = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
        self.evil_dir_index = 0

        # Pygame setup
        pygame.init()
        metrics_width = 200  # Width for metrics display
        self.screen = pygame.display.set_mode((grid_size[1]*cell_size + metrics_width, grid_size[0]*cell_size))
        pygame.display.set_caption("Basurahan Maze Simulation")
        self.clock = pygame.time.Clock()  # Add clock initialization

        # Load images (place these in same folder)
        self.wall_e_img = pygame.image.load("wall e.jfif")
        self.evil_img = pygame.image.load("evil robot.jfif")
        self.trash_img = pygame.image.load("trash.png")
        self.exit_img = pygame.image.load("exit point.jpg")
        self.mine_img = pygame.image.load("mine.jfif")
        self.wall_img = pygame.image.load("wall.png")

        # Scale images
        self.wall_e_img = pygame.transform.scale(self.wall_e_img, (cell_size-10, cell_size-10))
        self.evil_img = pygame.transform.scale(self.evil_img, (cell_size-10, cell_size-10))
        self.trash_img = pygame.transform.scale(self.trash_img, (cell_size-10, cell_size-10))
        self.exit_img = pygame.transform.scale(self.exit_img, (cell_size-10, cell_size-10))
        self.mine_img = pygame.transform.scale(self.mine_img, (cell_size-10, cell_size-10))
        self.wall_img = pygame.transform.scale(self.wall_img, (cell_size-10, cell_size-10))

        # Overlay for good/bad step coloring
        self.overlay = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=float)

    def _build_maze(self):
        rows, cols = self.grid_size
        self.maze = np.zeros((rows, cols), dtype=int)

        # Walls
        self.maze[1, 1] = self.WALL
        self.maze[2, 3] = self.WALL
        self.maze[4, 4] = self.WALL

        # Trash
        self.maze[0, 2] = self.TRASH
        self.maze[3, 1] = self.TRASH
        self.maze[5, 3] = self.TRASH

        # Mines
        self.maze[2, 2] = self.MINE
        self.maze[4, 1] = self.MINE

        # Exit point
        self.maze[5, 5] = self.EXIT

        # Start positions
        self.start_pos = (0, 0)
        self.agent_pos = self.start_pos
        self.evil_pos = (self.grid_size[0]//2, self.grid_size[1]//2)

    def reset(self):
        self._build_maze()
        self.agent_pos = self.start_pos
        self.evil_pos = (self.grid_size[0]//2, self.grid_size[1]//2)
        self.evil_dir_index = 0
        self.done = False
        self.total_reward = 0
        self.overlay[:] = 0
        return self._get_obs()

    def update_overlay(self, pos, good=True):
        r, c = pos
        if good:
            self.overlay[r, c] = np.array([0, 0, 255])  # Blue
        else:
            self.overlay[r, c] = np.array([255, 0, 0])  # Red

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, {}

        # Move Wall-E
        old_pos = self.agent_pos
        self.agent_pos = self._move(self.agent_pos, action)

        # Reward system
        reward = -1  # step cost

        if self.agent_pos == old_pos:
            reward -= 1  # hitting wall

        cell = self.maze[self.agent_pos]
        if cell == self.TRASH:
            reward += 10
            self.maze[self.agent_pos] = 0
        elif cell == self.MINE:
            reward -= 20
            self.done = True
        elif cell == self.EXIT:
            reward += 50
            self.done = True

        # Overlay color
        if reward > 0:
            self.update_overlay(self.agent_pos, good=True)
        elif reward < 0:
            self.update_overlay(self.agent_pos, good=False)

        # Evil robot moves
        self._move_evil_robot()

        # Collision with Evil robot
        if self.agent_pos == self.evil_pos:
            reward -= 50
            self.update_overlay(self.agent_pos, good=False)
            self.done = True

        self.total_reward += reward
        return self._get_obs(), reward, self.done, {}

    def _move(self, pos, action):
        r, c = pos
        moves = [(-1,0),(0,1),(1,0),(0,-1)]
        dr, dc = moves[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.grid_size[0] and 0 <= nc < self.grid_size[1]:
            if self.maze[nr, nc] != self.WALL:
                return (nr, nc)
        return (r, c)

    def _move_evil_robot(self):
        attempts = 0
        while attempts < 4:
            dr, dc = self.evil_dir_order[self.evil_dir_index]
            nr, nc = self.evil_pos[0] + dr, self.evil_pos[1] + dc
            if 0 <= nr < self.grid_size[0] and 0 <= nc < self.grid_size[1]:
                if self.maze[nr, nc] != self.WALL:
                    self.evil_pos = (nr, nc)
                    return
            self.evil_dir_index = (self.evil_dir_index + 1) % 4
            attempts += 1

    def _get_obs(self):
        obs = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=int)
        obs[self.agent_pos][0] = 1
        obs[self.evil_pos][1] = 1
        return obs

    def render(self, mode="human"):
        colors = {0: (240, 240, 240)}
        self.overlay = np.clip(self.overlay * 0.9, 0, 255)

        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                rect = pygame.Rect(c*self.cell_size, r*self.cell_size,
                                   self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, colors.get(0), rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

                if self.maze[r, c] == self.WALL:
                    self.screen.blit(self.wall_img, (c*self.cell_size+5, r*self.cell_size+5))
                elif self.maze[r, c] == self.TRASH:
                    self.screen.blit(self.trash_img, (c*self.cell_size+5, r*self.cell_size+5))
                elif self.maze[r, c] == self.MINE:
                    self.screen.blit(self.mine_img, (c*self.cell_size+5, r*self.cell_size+5))
                elif self.maze[r, c] == self.EXIT:
                    self.screen.blit(self.exit_img, (c*self.cell_size+5, r*self.cell_size+5))

                overlay_color = self.overlay[r, c]
                if np.any(overlay_color > 0):
                    surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    surf.fill((*overlay_color.astype(int), 100))
                    self.screen.blit(surf, (c*self.cell_size, r*self.cell_size))

        ar, ac = self.agent_pos
        self.screen.blit(self.wall_e_img, (ac*self.cell_size+5, ar*self.cell_size+5))

        er, ec = self.evil_pos
        self.screen.blit(self.evil_img, (ec*self.cell_size+5, er*self.cell_size+5))

        pygame.display.flip()
        if self.settings:
            self.clock.tick(self.settings.simulation_speed)
        else:
            self.clock.tick(10)


# Main simulation code
if __name__ == "__main__":
    # Initialize settings and show UI
    settings = AlgorithmSettings()
    settings_ui = SettingsUI(settings)
    settings = settings_ui.run()
    
    # Initialize environment with settings
    env = BasurahanMazeEnv(settings=settings)
    visualizer = MetricsVisualizer(env.grid_size, env.cell_size)
    
    # Initialize agent based on selected algorithm
    if settings.algorithm == "monte_carlo":
        agent = MonteCarloAgent(settings)
    elif settings.algorithm == "q_learning":
        agent = QLearningAgent(settings)
    else:  # actor_critic
        state_size = env.observation_space.shape[0] * env.observation_space.shape[1] * 3
        agent = ActorCriticAgent(settings, state_size)
    
    # Main training loop
    for ep in range(settings.episodes):
        visualizer.start_episode()
        obs = env.reset()
        state = (env.agent_pos, env.evil_pos) if settings.algorithm != "actor_critic" else obs.flatten()
        done = False
        episode_data = []
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            
            # Get action from agent
            action = agent.get_action(state)
            
            # Take step in environment
            obs, reward, done, info = env.step(action)
            next_state = (env.agent_pos, env.evil_pos) if settings.algorithm != "actor_critic" else obs.flatten()
            
            # Update visualization
            visualizer.update_heatmap(env.agent_pos)
            
            # Update agent
            if settings.algorithm == "monte_carlo":
                episode_data.append((state, action, reward))
            elif settings.algorithm == "q_learning":
                agent.update(state, action, reward, next_state)
            else:  # actor_critic
                agent.update(state, action, reward, next_state, done)
            
            state = next_state
            
            # Render environment and metrics
            env.render()
            if settings.show_reward_graph or settings.show_heatmap:
                visualizer.render_metrics(env.screen, agent if settings.algorithm == "actor_critic" else None)
            pygame.display.flip()
        
        # Episode cleanup
        if settings.algorithm == "monte_carlo":
            agent.update(episode_data)
        
        visualizer.end_episode(env.total_reward)
        print(f"Episode {ep+1} finished with total reward {env.total_reward}")
    
    time.sleep(2)
    pygame.quit()
