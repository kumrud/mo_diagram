import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools as it

__author__ = 'Kumru'

on_alpha = 1.0
off_alpha = 0.1

def generate_circle_graph(num_point, adjacency=None):
    """ Generates complete graph arranged in a circle

    Parameters
    ----------
    num_points : int
        Number of vertices on the graph
    adjacency : np.ndarray(K, K)
        Adjacency matrix of the graph
    """

    # Set points in circle
    theta = np.linspace(0, 2 * np.pi, num_point+1)
    x_point = np.cos(theta)
    y_point = np.sin(theta)
    point = [list(i) for i in zip(x_point, y_point)]

    # Plot points
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_point, y_point, color='black')
    # ax.plot(x_point, y_point, color='black', picker=True)

    if adjacency is None:
        adjacency = np.ones((num_point, num_point), dtype=bool)
        adjacency -= np.diag(np.diag(adjacency))
    else:
        assert adjacency.shape[0] == adjacency.shape[1] == num_point, ''
        'Given adjacency matrix does not match the number of points'

    # Plot lines
    lines = []
    for indices in zip(*np.nonzero(np.triu(adjacency))):
        x_line = [x_point[i] for i in indices]
        y_line = [y_point[i] for i in indices]
        lines.append(ax.plot(x_line, y_line, color='blue', alpha=on_alpha, linestyle='dashed', solid_capstyle='round', linewidth=2.5, picker=10)[0])

    # Set line visibility by clicking
    def onpick(event):
        thisline = event.artist
        if thisline.get_alpha() == on_alpha:
            thisline.set_alpha(off_alpha)
        elif thisline.get_alpha() == off_alpha:
            thisline.set_alpha(on_alpha)
        fig.canvas.draw()
    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()
    return fig


# adjacency = np.array([[0, 1, 1, 0],
#                          [1, 0, 0, 1],
#                          [1, 0, 0, 1],
#                          [0, 1, 1, 0]])
generate_circle_graph(10, )
