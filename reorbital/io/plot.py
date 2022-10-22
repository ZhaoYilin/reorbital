import numpy as np
import matplotlib.pyplot as plt

def plot_single_orbital_entropy(soe):
    norbital = soe.size
    order = range(norbital)
    markerline, stemlines, baseline = plt.stem(
        order, sre, linefmt='grey', markerfmt='', bottom=0)
    markerline.set_markerfacecolor('none')

    fig, ax = plt.subplots()
    ax.imshow(mi)
    plt.show()
