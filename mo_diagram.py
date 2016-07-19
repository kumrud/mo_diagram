__author__ = 'kumru'

import numpy as np
import matplotlib.pyplot as plt
import pylab as ply
from energy import fock_horton, fock_numerical
from quasibasis.quasi import QuasiTransformation
from quasibasis.wrapper_horton import HortonData

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
        fock_ab = fock_numerical([coeff_ab_mo], [mo_energies])[0]
        fock_quambo = coeff_ab_quambo.T.dot(fock_ab).dot(coeff_ab_quambo)
        fock_quambo_sep.append(fock_quambo)
    quambo_energies = np.hstack([np.diag(fock) for fock in fock_quambo_sep])

    mo_energies = np.hstack([mo_energies for mo_energies in mo_energies_sep])
    occupations = np.hstack([occs for occs in occupations_sep])
    return mo_energies, occupations, quambo_energies


def generate_one_diagram(energies, occupations, degens, ax, center=0, line_width=1.0, line_sep=0.1):
    # display
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
        for x, y, occ in zip(xs_degen, ys_degen, occs_degen):
            if occ > 0.01:
                line, = ax.plot(x, y, color='green', alpha=1.0, picker=10)
            else:
                line, = ax.plot(x, y, color='red', alpha=1.0, picker=10)

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            text = ax.text(x[1]+0.2, y[1]+0.4, y[0], fontsize=14, verticalalignment='top', bbox=props)
            text.set_visible(False)
        counter += degen

def generate_all_mo_diagrams(fig, ax, list_energies, list_occupations, pick_event_energy=True, weights=None):
    list_sorted_energies = []
    list_sorted_occupations = []
    list_sorted_degens = []
    # sort energies and find degeneracies
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

    # find diagram width and spacing parameters
    max_degens = [max(degens) for degens in list_sorted_degens]
    line_width = 1.0
    line_sep = 0.1
    diagram_widths = [(line_width+line_sep)*max_degen-line_sep for max_degen in max_degens]
    diagram_sep = 1.0
    total_width = sum(diagram_widths) + diagram_sep*(len(diagram_widths)-1)
    leftmost_center = + diagram_widths[0]/2.0

    # generate each diagram
    for i, energies, occupations, degens in zip(range(len(diagram_widths)),
                                                list_sorted_energies,
                                                list_sorted_occupations,
                                                list_sorted_degens):
        center = -total_width/2.0 + sum(diagram_widths[:i]) + i*diagram_sep + diagram_widths[i]/2.0
        generate_one_diagram(energies, occupations, degens, ax, center=center, line_width=line_width, line_sep=line_sep)

    # make lines
    if isinstance(weights, np.ndarray):
        assert weights.shape == tuple(len(i) for i in list_energies), 'Given weights must have the right shape'
        assert np.all(weights >= 0), 'Given weights cannot be negative'
        line_threshold = 1e-5
        flat_indices, = np.where(weights.flat > line_threshold)
        indices = [np.unravel_index(index, weights.shape) for index in flat_indices]
        for index in indices:
            # assume first diagram is the mo diagram
            to_x_coords, to_y_coords = ax.lines[index[0]].get_data()
            for i,j in enumerate(index[1:]):
                # find the index of the line that corresponds to ab
                from_line_index = sum(weights.shape[:i+1]) + j
                # get the coordinates of this line
                from_x_coords, from_y_coords = ax.lines[from_line_index].get_data()
                # make new line coordinates
                line_data = [[to_x_coords[1], from_x_coords[0]],
                             [to_y_coords[1], from_y_coords[0]]]
                # make line
                line, = ax.plot(*line_data, color='blue', alpha=1.0, picker=10)

    # add label to y-axis
    ax.set_ylabel("Energy (Hartree)")

    # set x range
    ax.set_xlim([-total_width/2.0-0.5, total_width/2.0+0.5])

    # set y range
    min_energies = [min(energies) for energies in list_energies]
    max_energies = [max(energies) for energies in list_energies]
    ax.set_ylim([min(min_energies)-1, max(max_energies)+1])

    # remove x-axis label
    ax.xaxis.set_visible(False)
    # clicking on line show annotation
    def on_pick(event):
        thisline = event.artist
        index = ax.lines.index(thisline)
        visibility = ax.texts[index].get_visible()
        ax.texts[index].set_visible(not visibility)
        fig.canvas.draw()

    if pick_event_energy:
        fig.canvas.mpl_connect('pick_event', on_pick)
    # on_move_id = fig.canvas.mpl_connect('motion_notify_event', on_move)

# show plot
fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
energies, occupations, quambo_energies = get_quambo_data('ch4_svp_minao_iao.fchk')
weights = np.ones(tuple(len(i) for i in [energies, quambo_energies]))
generate_all_mo_diagrams(fig, ax, [energies, quambo_energies], [occupations]*2, weights=weights)
plt.show()

# save plot in a eps file
# plt.savefig('ELD.eps')
