__author__ = 'kumru'

import matplotlib.pyplot as plt
import pylab as ply
from energy import fock_horton, fock_numerical
from quasi import QuasiTransformation
from wrapper_horton import HortonData
import numpy as np

def get_quambo_data(fchkfile, cao_basis_files='aambs.gbs'):
    hd = HortonData(fchkfile, cao_basis_files)
    coeff_ab_mo_sep = hd.coeff_ab_mo_sep
    olp_ab_ab_sep = hd.olp_ab_ab_sep
    olp_cao_ab_sep = hd.olp_cao_ab_sep
    olp_cao_cao_sep = hd.olp_cao_cao_sep
    occupations_sep = hd.occupations_sep
    mo_energies_sep = hd.energies_sep
    # generate quambo energies
    fock_quambo_sep = []
    for (coeff_ab_mo,
         olp_ab_ab,
         olp_cao_ab,
         olp_cao_cao,
         mo_energies,
         occupations) in zip(coeff_ab_mo_sep,
                             olp_ab_ab_sep,
                             olp_cao_ab_sep,
                             olp_cao_cao_sep,
                             mo_energies_sep,
                             occupations_sep):
        quasi = QuasiTransformation(coeff_ab_mo,
                                    olp_ab_ab,
                                    olp_cao_ab,
                                    olp_cao_cao,
                                    occupations.astype(bool))
        coeff_ab_quambo = quasi.quambo()
        '''fock_quambo_quambo is QUAMBO energy
        '''
        fock_ab = fock_numerical(coeff_ab_mo, mo_energies)
        fock_quambo = coeff_ab_quambo.T.dot(fock_ab).dot(coeff_ab_quambo)
        fock_quambo_sep.append(fock_quambo)
    quambo_energies = np.hstack([np.diag(fock) for fock in fock_quambo_sep])

    mo_energies = np.hstack([mo_energies for mo_energies in mo_energies_sep])
    occupations = np.hstack([occs for occs in occupations_sep])
    return mo_energies, occupations, quambo_energies


def generate_one_diagram(energies, occupations, degens, fig, ax, center=0, line_width=1.0, line_sep=0.1):
    # display
    points_with_annotation = []
    counter = 0
    for degen in degens:
        x_leftmost = -(degen//2)*(line_width+line_sep) - (degen%2)*line_width/2.0
        x_inits = [x_leftmost + i*(line_width+line_sep) + center for i in range(degen)]
        x_finals = [x_leftmost+line_width + i*(line_width+line_sep) + center for i in range(degen)]
        xs_degen = zip(x_inits, x_finals)

        y_inits = energies[counter:counter+degen]
        y_finals = energies[counter:counter+degen]
        ys_degen = zip(y_inits, y_finals)

        occs_degen = occupations[counter:counter+degen]
        for x,y,occ in zip(xs_degen, ys_degen, occs_degen):
            if occ > 0.01:
                line, = plt.plot(x, y, color='green')
            else:
                line, = plt.plot(x, y, color='red')

            num_boxes = 100
            delta_x = line_width/num_boxes
            for i in range(num_boxes):
                # hovering and plotting
                # xy = coordinate of mouse
                # xytext = coordinate of annotation
                # horizontalalignment = horizontal alignment
                # bbox = annotation box style
                annotation = ax.annotate(str(y[0]),
                                        xy=(x[0]+i*delta_x, y[0]), xycoords='data',
                                        xytext=(x[1], y[1]+0.2), textcoords='data',
                                        horizontalalignment="center",
                                        #arrowprops=dict(arrowstyle="simple",
                                        #connectionstyle="arc3,rad=0.0"),
                                        bbox=dict(boxstyle="round", facecolor="w",
                                                edgecolor="0.5", alpha=0.9)
                )
                # by default, disable the annotation visibility
                annotation.set_visible(False)
                # keep list of annotation coordinates for user interaction
                points_with_annotation.append([line, annotation])
        counter += degen

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


def generate_all_mo_diagrams(list_energies, list_occupations):
    list_sorted_energies = []
    list_sorted_occupations = []
    list_sorted_degens = []
    for energies, occupations in zip(list_energies, list_occupations):
        # sort energies
        sort_indices = np.argsort(energies)
        energies = energies[sort_indices]
        occupations = occupations[sort_indices]

        energy_deltas = np.diff(energies)
        tol_degen = 0.01
        energy_indices = [i for i,j in enumerate(energy_deltas) if j>tol_degen]
        energy_bins = [energies[0]]+[energies[i+1] for i in energy_indices]+[energies[-1]+1]
        degens = np.histogram(energies, bins=energy_bins)[0]

        list_sorted_energies.append(energies)
        list_sorted_occupations.append(occupations)
        list_sorted_degens.append(degens)

    fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
    max_degens = [max(degens) for degens in list_sorted_degens]
    line_width = 1.0
    line_sep = 0.1
    diagram_widths = [(line_width+line_sep)*max_degen-line_sep for max_degen in max_degens]
    diagram_sep = 0.5
    total_width = sum(diagram_widths) + diagram_sep*(len(diagram_widths)-1)
    leftmost_center = + diagram_widths[0]/2.0
    for i, energies, occupations, degens in zip(range(len(diagram_widths)),
                                                list_sorted_energies,
                                                list_sorted_occupations,
                                                list_sorted_degens):
        center = -total_width/2.0 + sum(diagram_widths[:i]) + i*diagram_sep + diagram_widths[i]/2.0
        generate_one_diagram(energies, occupations, degens, fig, ax, center=center, line_width=line_width, line_sep=line_sep)

        # add label to y-axis
    ply.ylabel("Energy (Hartree)")

    # set x range
    ply.xlim([-total_width/2.0-0.5, total_width/2.0+0.5])

    # set y range
    min_energies = [min(energies) for energies in list_energies]
    max_energies = [max(energies) for energies in list_energies]
    ply.ylim([min(min_energies)-1, max(max_energies)+1])

    # remove x-axis label
    plt.gca().xaxis.set_major_locator(plt.NullLocator())

    # show plot
    plt.show()

    # save plot in a eps file
    # plt.savefig('ELD.eps')
energies, occupations, quambo_energies = get_quambo_data('ch4_hf.fchk')
generate_all_mo_diagrams([energies, quambo_energies], [occupations]*2)
