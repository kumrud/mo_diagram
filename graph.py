import matplotlib.pyplot as plt
import numpy as np
import itertools as it

__author__ = 'Kumru'


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
    plt.scatter(x_point, y_point, color='black')

    if adjacency is None:
        adjacency = np.ones((num_point, num_point), dtype=bool)
        adjacency -= np.diag(np.diag(adjacency))
    else:
        assert adjacency.shape[0] == adjacency.shape[1] == num_point, ''
        'Given adjacency matrix does not match the number of points'

    lines = []
    # Plot lines
    for indices in zip(*np.nonzero(np.triu(adjacency))):
        x_line = [x_point[i] for i in indices]
        y_line = [y_point[i] for i in indices]
        lines.append(plt.plot(x_line, y_line, color='blue', linestyle='dashed', solid_capstyle='round', linewidth=2.5)[0])

    # Set line visibility by clicking
    def onpick(event):
        artist = event.artist
        vis = not line.get_visible()
        artist.set_visible(vis)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()


adjacency = np.array([[0, 1, 1, 0],
                         [1, 0, 0, 1],
                         [1, 0, 0, 1],
                         [0, 1, 1, 0]])
generate_circle_graph(4, adjacency)
