import pygame
import numpy as np

CAPTION = "dm_control viewer"


class DmControlViewer():
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(CAPTION)
        self.screen = None

    def loop_once(self, image):
        image = np.swapaxes(image, 0, 1)

        if not self.screen:
            self.screen = pygame.display.set_mode((image.shape[0],
                                                   image.shape[1]))

        pygame.surfarray.blit_array(self.screen, image)
        pygame.display.flip()

    def finish(self):
        pygame.quit()
