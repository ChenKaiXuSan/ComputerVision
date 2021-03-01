import matplotlib.pyplot as plt
from numpy.lib.npyio import save

def show_train_hist(hist, show=False, save=False, path='./train_hist.png'):

    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='Dlosses')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

import imageio

def show_train_animation(path, save_name):
    images = []
    for name in save_name:
        img_name = path + str(name) + '.png'

    images.append(imageio.imread(img_name))

    imageio.mimsave(path + 'generation-animation.gif', images, fps=5)
    return "sucess save image animation"



    