import numpy as np
import pygame

class BasurahanMazeEnvWithHeatmap:
    def __init__(self, grid_size=(6, 6), cell_size=80):
        self.grid_size = grid_size
        self.cell_size = cell_size

        # Track tile colors (fade effect)
        self.overlay = np.zeros((*grid_size, 3), dtype=float)  # RGB overlay [0-255]

        pygame.init()
        self.screen = pygame.display.set_mode((grid_size[1]*cell_size, grid_size[0]*cell_size))
        pygame.display.set_caption("Basurahan Maze with Heatmap")
        self.clock = pygame.time.Clock()

    def update_overlay(self, pos, good=True):
        r, c = pos
        if good:
            self.overlay[r, c] = [0, 0, 255]  # blue
        else:
            self.overlay[r, c] = [255, 0, 0]  # red

    def fade_overlay(self):
        # Slowly fade out colors
        self.overlay = np.maximum(self.overlay - 15, 0)  # reduce each channel

    def render(self, maze, agent_pos, evil_pos, wall_e_img, evil_img, trash_img, exit_img, mine_img, wall_img):
        self.fade_overlay()
        
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                rect = pygame.Rect(c*self.cell_size, r*self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (240, 240, 240), rect)  # base
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)     # grid line

                # Draw overlay fade color
                if self.overlay[r, c].sum() > 0:
                    color = tuple(self.overlay[r, c].astype(int))
                    overlay_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    overlay_surface.fill((*color, 80))  # RGBA with transparency
                    self.screen.blit(overlay_surface, (c*self.cell_size, r*self.cell_size))

                # Objects in maze
                if maze[r, c] == -1:
                    self.screen.blit(wall_img, (c*self.cell_size+5, r*self.cell_size+5))
                elif maze[r, c] == 1:
                    self.screen.blit(trash_img, (c*self.cell_size+5, r*self.cell_size+5))
                elif maze[r, c] == -2:
                    self.screen.blit(mine_img, (c*self.cell_size+5, r*self.cell_size+5))
                elif maze[r, c] == 9:
                    self.screen.blit(exit_img, (c*self.cell_size+5, r*self.cell_size+5))

        # Draw Wall-E
        ar, ac = agent_pos
        self.screen.blit(wall_e_img, (ac*self.cell_size+5, ar*self.cell_size+5))

        # Draw Evil Robot
        er, ec = evil_pos
        self.screen.blit(evil_img, (ec*self.cell_size+5, er*self.cell_size+5))

        pygame.display.flip()
        self.clock.tick(5)

# --- Example integration with step() ---
# Inside your step() after computing reward:
# if reward > 0: env.update_overlay(agent_pos, good=True)
# else: env.update_overlay(agent_pos, good=False)