import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools as it

__author__ = 'Kumru'

def generate_circle_graph(fig, ax, num_points, adjacency=None, on_alpha=1.0, off_alpha=0.1):
    """ Generates complete graph arranged in a circle

    Parameters
    ----------
    num_pointss : int
        Number of vertices on the graph
    adjacency : np.ndarray(K, K)
        Adjacency matrix of the graph
    """

    # Set points in circle
    theta = np.linspace(0, 2 * np.pi, num_points+1)
    x_point = np.cos(theta)[:-1]
    y_point = np.sin(theta)[:-1]
    point = [list(i) for i in zip(x_point, y_point)]

    # Plot points
    scat = ax.scatter(x_point, y_point, color='black')
    # ax.plot(x_point, y_point, color='black', picker=True, linestyle='None')

    if adjacency is None:
        adjacency = np.ones((num_points, num_points), dtype=bool)
        adjacency -= np.diag(np.diag(adjacency))
    else:
        assert adjacency.shape[0] == adjacency.shape[1] == num_points, ''
        'Given adjacency matrix does not match the number of points'

    # Plot lines
    lines = []
    for indices in zip(*np.where(np.triu(adjacency))):
        x_line = [x_point[i] for i in indices]
        y_line = [y_point[i] for i in indices]
        lines.append(ax.plot(x_line, y_line, color='blue', alpha=on_alpha, linestyle='dashed', solid_capstyle='round', linewidth=2.5, picker=10)[0])

    # Set line visibility by clicking
    def onpick(event):
        thisline = event.artist
        old_alpha = thisline.get_alpha()
        new_alpha = on_alpha+off_alpha-old_alpha
        thisline.set_alpha(new_alpha)
        fig.canvas.draw()
    fig.canvas.mpl_connect('pick_event', onpick)
    return scat

# adjacency = np.array([[0, 1, 1, 0],
#                          [1, 0, 0, 1],
#                          [1, 0, 0, 1],
#                          [0, 1, 1, 0]])
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# scat = generate_circle_graph(fig, ax, 10, )
# plt.show()
