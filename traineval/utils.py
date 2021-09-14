"""
Author: Dikshant Gupta
Time: 14.09.21 09:51
"""
import pygame
from pygame.locals import K_r, K_TAB, K_q, K_ESCAPE, KMOD_CTRL


class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)
        self.world = world
        for i in range(3):
            self.world.camera_manager.toggle_camera()

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_TAB:
                    self.world.camera_manager.toggle_camera()
                elif event.key == K_r:
                    self.world.camera_manager.toggle_recording()

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)
