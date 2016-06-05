import matplotlib.pyplot as plt
import numpy as np

__author__ = 'Kumru'


def generate_graph(num_point, theta=None):
    """ Generates correlation graph as circle

    Parameters
    ----------
    theta : np.array
        The angular values
    """

    # Set points in circle
    if theta is None:
        theta = np.linspace(0, 2 * np.pi, num_point+1)
    x_point = np.cos(theta)
    y_point = np.sin(theta)
    point = [list(i) for i in zip(x_point, y_point)]

    xs = [i[0] for i in point]
    ys = [i[1] for i in point]

    # Plot points
    plt.scatter(xs, ys, color='red')

    # Plot lines
    for i in point:
        for j in point:
            if i != j:
                print i, j
                x_line = [i[0], j[0]]
                y_line = [i[1], j[1]]
                plt.plot(x_line, y_line, color='green')
                print np.asarray(x_line)-np.asarray(y_line)
            else:
                pass
    plt.show()

