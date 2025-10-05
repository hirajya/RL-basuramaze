import tkinter as tk
from tkinter import ttk
import json
import os

class AlgorithmSettings:
    def __init__(self):
        # General settings
        self.algorithm = "monte_carlo"
        self.episodes = 200
        self.epsilon = 0.1
        self.gamma = 0.99
        self.max_steps = 100
        
        # Q-Learning specific
        self.alpha = 0.1
        
        # Actor-Critic specific
        self.actor_lr = 0.1
        self.critic_lr = 0.1
        self.beta = 2.3  # Temperature parameter for policy
        self.entropy_weight = 0.01  # Weight for entropy regularization
        self.value_weight = 0.5  # Weight for value loss
        self.advantage_weight = 1.0  # Weight for policy advantage
        
        # Rewards configuration
        self.step_penalty = -0.1
        self.trash_reward = 10.0
        self.mine_penalty = -10.0
        self.evil_penalty = -10.0
        self.exit_reward = 50.0
        
        # Simulation speed
        self.simulation_speed = 30  # Frames per second
        
        # Visualization
        self.show_reward_graph = True
        self.show_heatmap = True
        self.show_value_map = True  # Show state values for actor-critic
        
    def save_settings(self, filename='settings.json'):
        with open(filename, 'w') as f:
            json.dump(self.__dict__, f)
            
    def load_settings(self, filename='settings.json'):
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                settings = json.load(f)
                for key, value in settings.items():
                    setattr(self, key, value)

class SettingsUI:
    def __init__(self, settings):
        self.settings = settings
        self.root = tk.Tk()
        self.root.title("Algorithm Settings")
        self.create_widgets()
        
    def create_widgets(self):
        # Algorithm selection
        ttk.Label(self.root, text="Algorithm:").grid(row=0, column=0, padx=5, pady=5)
        self.algo_var = tk.StringVar(value=self.settings.algorithm)
        algo_combo = ttk.Combobox(self.root, textvariable=self.algo_var)
        algo_combo['values'] = ('monte_carlo', 'q_learning', 'actor_critic')
        algo_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # Episodes
        ttk.Label(self.root, text="Episodes:").grid(row=1, column=0, padx=5, pady=5)
        self.episodes_var = tk.IntVar(value=self.settings.episodes)
        ttk.Entry(self.root, textvariable=self.episodes_var).grid(row=1, column=1, padx=5, pady=5)
        
        # Epsilon
        ttk.Label(self.root, text="Epsilon:").grid(row=2, column=0, padx=5, pady=5)
        self.epsilon_var = tk.DoubleVar(value=self.settings.epsilon)
        ttk.Entry(self.root, textvariable=self.epsilon_var).grid(row=2, column=1, padx=5, pady=5)
        
        # Gamma
        ttk.Label(self.root, text="Gamma:").grid(row=3, column=0, padx=5, pady=5)
        self.gamma_var = tk.DoubleVar(value=self.settings.gamma)
        ttk.Entry(self.root, textvariable=self.gamma_var).grid(row=3, column=1, padx=5, pady=5)
        
        # Alpha (Q-Learning)
        ttk.Label(self.root, text="Alpha (Q-Learning):").grid(row=4, column=0, padx=5, pady=5)
        self.alpha_var = tk.DoubleVar(value=self.settings.alpha)
        ttk.Entry(self.root, textvariable=self.alpha_var).grid(row=4, column=1, padx=5, pady=5)
        
        # Learning rates (Actor-Critic)
        ttk.Label(self.root, text="Actor LR:").grid(row=5, column=0, padx=5, pady=5)
        self.actor_lr_var = tk.DoubleVar(value=self.settings.actor_lr)
        ttk.Entry(self.root, textvariable=self.actor_lr_var).grid(row=5, column=1, padx=5, pady=5)
        
        ttk.Label(self.root, text="Critic LR:").grid(row=6, column=0, padx=5, pady=5)
        self.critic_lr_var = tk.DoubleVar(value=self.settings.critic_lr)
        ttk.Entry(self.root, textvariable=self.critic_lr_var).grid(row=6, column=1, padx=5, pady=5)
        
        # Simulation Speed
        ttk.Label(self.root, text="Simulation Speed (FPS):").grid(row=8, column=0, padx=5, pady=5)
        self.speed_var = tk.IntVar(value=self.settings.simulation_speed)
        speed_scale = ttk.Scale(self.root, from_=1, to=60, variable=self.speed_var, orient=tk.HORIZONTAL)
        speed_scale.grid(row=8, column=1, padx=5, pady=5)
        speed_scale.set(self.settings.simulation_speed)
        
        # Visualization options
        self.show_reward_graph_var = tk.BooleanVar(value=self.settings.show_reward_graph)
        ttk.Checkbutton(self.root, text="Show Reward Graph", variable=self.show_reward_graph_var).grid(row=9, column=0, columnspan=2, pady=5)
        
        self.show_heatmap_var = tk.BooleanVar(value=self.settings.show_heatmap)
        ttk.Checkbutton(self.root, text="Show Heatmap", variable=self.show_heatmap_var).grid(row=10, column=0, columnspan=2, pady=5)
        
        # Buttons
        ttk.Button(self.root, text="Start Training", command=self.apply_settings).grid(row=11, column=0, columnspan=2, pady=10)
        ttk.Button(self.root, text="Save Settings", command=self.save_settings).grid(row=12, column=0, pady=5)
        ttk.Button(self.root, text="Load Settings", command=self.load_settings).grid(row=12, column=1, pady=5)
        
    def apply_settings(self):
        self.settings.algorithm = self.algo_var.get()
        self.settings.episodes = self.episodes_var.get()
        self.settings.epsilon = self.epsilon_var.get()
        self.settings.gamma = self.gamma_var.get()
        self.settings.alpha = self.alpha_var.get()
        self.settings.actor_lr = self.actor_lr_var.get()
        self.settings.critic_lr = self.critic_lr_var.get()
        self.settings.simulation_speed = self.speed_var.get()
        self.settings.show_reward_graph = self.show_reward_graph_var.get()
        self.settings.show_heatmap = self.show_heatmap_var.get()
        self.root.quit()
        
    def save_settings(self):
        self.apply_settings()
        self.settings.save_settings()
        
    def load_settings(self):
        self.settings.load_settings()
        self.update_ui_from_settings()
        
    def update_ui_from_settings(self):
        self.algo_var.set(self.settings.algorithm)
        self.episodes_var.set(self.settings.episodes)
        self.epsilon_var.set(self.settings.epsilon)
        self.gamma_var.set(self.settings.gamma)
        self.alpha_var.set(self.settings.alpha)
        self.actor_lr_var.set(self.settings.actor_lr)
        self.critic_lr_var.set(self.settings.critic_lr)
        self.speed_var.set(self.settings.simulation_speed)
        self.show_reward_graph_var.set(self.settings.show_reward_graph)
        self.show_heatmap_var.set(self.settings.show_heatmap)
        
    def run(self):
        self.root.mainloop()
        return self.settings