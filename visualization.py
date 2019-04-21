from matplotlib import pyplot as plt
import os
import numpy as np
class Visualizer():
    def __init__(self, exp_name, row, col):
        self.fig, self.axs = plt.subplots(row, col)
        self.row, self.col = row, col
        self.exp_name = exp_name
        os.makedirs(exp_name, exist_ok=True)
    def draw_imgs(self, images):
        img_channel = images.shape[3]
        if img_channel == 1:
            img_channel = 0
            cmap = 'gray'
        else:
            cmap = 'rgb'
        for j in range(self.row):
            for i in range(self.col):
                self.axs[j, i].imshow(images[j*self.col+i, :, :, img_channel], cmap = cmap)
                self.axs[j, i].axis('off')
    def save_fig(self, name):
        filename = "{}.png".format(name)
        path = os.path.join(self.exp_name, filename)
        self.fig.savefig(path)