import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns

class MetricsVisualizer:
    def __init__(self, grid_size, cell_size):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.visit_heatmap = np.zeros(grid_size)
        self.value_heatmap = np.zeros(grid_size)
        self.rewards = []
        self.episode_start_time = None
        self.episode_times = []
        
        # Setup matplotlib figure with three subplots
        self.fig = plt.figure(figsize=(4, 8))
        gs = self.fig.add_gridspec(3, 1)
        self.ax1 = self.fig.add_subplot(gs[0])  # Rewards
        self.ax2 = self.fig.add_subplot(gs[1])  # Visit frequency
        self.ax3 = self.fig.add_subplot(gs[2])  # State values
        self.fig.tight_layout(pad=3.0)
        
        # Color maps
        self.visit_cmap = 'YlOrRd'  # Yellow-Orange-Red for visits
        self.value_cmap = 'RdYlGn'  # Red-Yellow-Green for state values
        
    def start_episode(self):
        self.episode_start_time = pygame.time.get_ticks()
        
    def end_episode(self, reward):
        if self.episode_start_time is not None:
            episode_time = (pygame.time.get_ticks() - self.episode_start_time) / 1000.0
            self.episode_times.append(episode_time)
            self.rewards.append(reward)
            
    def update_heatmap(self, pos):
        self.visit_heatmap[pos] += 1
        
    def update_value_heatmap(self, agent):
        if hasattr(agent, 'get_state_values'):
            state_values = agent.get_state_values()
            if state_values:
                self.value_heatmap = np.zeros(self.grid_size)
                for state_key, value in state_values.items():
                    # Convert flattened state back to position
                    state_array = np.array(state_key).reshape(self.grid_size + (3,))
                    agent_pos = np.unravel_index(np.argmax(state_array[:,:,0]), self.grid_size)
                    self.value_heatmap[agent_pos] = value
        
    def render_metrics(self, screen, agent=None):
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Plot rewards
        if self.rewards:
            self.ax1.plot(self.rewards, label='Episode Reward')
            self.ax1.axhline(y=np.mean(self.rewards), color='r', linestyle='--', label='Average')
            self.ax1.set_title('Training Rewards')
            self.ax1.set_xlabel('Episode')
            self.ax1.set_ylabel('Total Reward')
            self.ax1.legend()
            self.ax1.grid(True)
        
        # Plot visit frequency heatmap
        sns.heatmap(self.visit_heatmap, ax=self.ax2, cmap=self.visit_cmap, 
                   xticklabels=False, yticklabels=False)
        self.ax2.set_title('Visit Frequency')
        
        # Plot state values heatmap
        if agent is not None:
            self.update_value_heatmap(agent)
            vmin = min(-1, self.value_heatmap.min())
            vmax = max(1, self.value_heatmap.max())
            sns.heatmap(self.value_heatmap, ax=self.ax3, cmap=self.value_cmap,
                       center=0, vmin=vmin, vmax=vmax,
                       xticklabels=False, yticklabels=False)
            self.ax3.set_title('State Values')
        
        # Convert plot to pygame surface
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.buffer_rgba().tobytes()
        size = canvas.get_width_height()
        
        # Create pygame surface from plot
        plot_surface = pygame.image.fromstring(raw_data, size, "RGBA")
        plot_surface = pygame.transform.scale(plot_surface, (200, 400))
        
        # Add metrics text
        font = pygame.font.Font(None, 24)
        metrics_text = []
        if self.rewards:
            metrics_text.append(f"Avg Reward: {np.mean(self.rewards):.1f}")
        if self.episode_times:
            metrics_text.append(f"Avg Episode Time: {np.mean(self.episode_times):.1f}s")
        
        # Position metrics display
        metrics_surface = pygame.Surface((200, len(metrics_text) * 25))
        metrics_surface.fill((240, 240, 240))
        
        for i, text in enumerate(metrics_text):
            text_surface = font.render(text, True, (0, 0, 0))
            metrics_surface.blit(text_surface, (5, i * 25))
        
        # Blit everything to screen
        screen_width = self.grid_size[1] * self.cell_size
        screen.blit(plot_surface, (screen_width, 0))
        screen.blit(metrics_surface, (screen_width, 410))