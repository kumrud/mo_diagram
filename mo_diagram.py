__author__ = 'kumru'

import matplotlib.pyplot as plt
import pylab as ply
from energy import horton_energy, fchk_energy, quambo_energy

y_energy = quambo_energy('ch3_rohf_sto3g_g03.fchk', 'aambs.gbs')
print y_energy

occupation = horton_energy('ch3_rohf_sto3g_g03.fchk')[3]
print occupation

# in case of degeneracy give different x values to MOs
v = [f - i for i, f in zip(y_energy, y_energy[1:])]
print v

# set degeneracy tolerance to 0.1
check_degen = [i > 0.1 for i in v]
print check_degen

fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
points_with_annotation = []
for i in range(len(y_energy)):
    # check for 1 degeneracies
    if i > 0 and not check_degen[i - 1]:
        x = [0.9, 1.5]
    else:
        x = [0.2, 0.8]

    y = [y_energy[i], y_energy[i]]
    # color for occupation?

    occupation = exp_alpha.occupations.astype(bool)
    if occupation[i] == True:
        line, = plt.plot(x,y,color='green')
    else:
        line, = plt.plot(x,y,color='red')
    print line

    line_length = x[1]-x[0]
    num_boxes = 100
    delta_x = line_length/num_boxes
    for j in range(num_boxes):
        annotation = ax.annotate(str(y[0]),
            xy=(x[0]+j*delta_x, y[1]), xycoords='data',
            xytext=(x[1], y[1]+0.2), textcoords='data',
            horizontalalignment="left",
            #arrowprops=dict(arrowstyle="simple",
                            #connectionstyle="arc3,rad=0.0"),
            bbox=dict(boxstyle="round", facecolor="w",
                      edgecolor="0.5", alpha=0.9)
            )
        annotation.set_visible(False)
        points_with_annotation.append([line, annotation])
        # by default, disable the annotation visibility

print 'energy level difference  = {0}'.format(v)

# add label to y-axis
ply.ylabel("E")

# set x and y range
ply.xlim([0, 2])
ply.ylim([min(y_energy)-1, max(y_energy)+1])

# remove x-axis label
plt.gca().xaxis.set_major_locator(plt.NullLocator())

# hovering over data points show annotation
def on_move(event):
    visibility_changed = False
    for point, annotation in points_with_annotation:
        should_be_visible = (point.contains(event)[0] == True)

        if should_be_visible != annotation.get_visible():
            visibility_changed = True
            annotation.set_visible(should_be_visible)

    if visibility_changed:
        plt.draw()

on_move_id = fig.canvas.mpl_connect('motion_notify_event', on_move)

# show plot
plt.show()

# save plot in a eps file
plt.savefig('ELD.eps')

print occupation
