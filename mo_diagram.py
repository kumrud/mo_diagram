__author__ = 'kumru'

import numpy as np
import matplotlib.pyplot as plt
from energy import fock_horton, fock_numerical
from quasibasis.quasi import QuasiTransformation, project
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
    coeff_quambo_mo_sep = []
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
        # get coefficients
        olp_quambo_quambo = coeff_ab_quambo.T.dot(olp_ab_ab).dot(coeff_ab_quambo)
        olp_quambo_mo = coeff_ab_quambo.T.dot(olp_ab_ab).dot(coeff_ab_mo)
        coeff_quambo_mo_sep.append(project(olp_quambo_quambo, olp_quambo_mo))
    quambo_energies = np.hstack([np.diag(fock) for fock in fock_quambo_sep])
    coeff_quambo_mo = np.hstack(coeff_quambo_mo_sep)

    mo_energies = np.hstack(mo_energies_sep)
    occupations = np.hstack(occupations_sep)
    return mo_energies, occupations, quambo_energies, coeff_quambo_mo


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

def generate_all_mo_diagrams(fig, ax, list_energies, list_occupations, pick_event_energy=True, pairwise_weights=None):
    """ Creates a collection of MO diagrams

    The different MO diagrams can be linked to one another by providing the weights of contribution

    Parameters
    ----------
    # CHECK!
    fig : matplotlib.Figure
    ax : matplotlib.Axis
    list_energies : list of list
    list_occupations : list of list
    pick_event_energy : bool
        Flag for using pick event
    pairwise_weights : Dictionary of tuple of integers to np.ndarray
        Key is the tuple of the two diagrams that is being compared
        The orbitals of the first diagram are the contributing orbitals
        The orbitals of the second diagram are the resulting orbitals
        i.e. the contributions of the orbitals from the first diagram is given for each orbital
        of the second diagram
        The weights describe the contribution of the orbitals from the first diagram (rows) to
        the orbitals of the second diagram (columns)

    Returns
    -------

    """
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

    # generate each diagram
    for i, energies, occupations, degens in zip(range(len(diagram_widths)),
                                                list_sorted_energies,
                                                list_sorted_occupations,
                                                list_sorted_degens):
        center = -total_width/2.0 + sum(diagram_widths[:i]) + i*diagram_sep + diagram_widths[i]/2.0
        generate_one_diagram(energies, occupations, degens, ax, center=center, line_width=line_width, line_sep=line_sep)

    # make a translator from the packaged index (which specifies which system the orbital belongs to)
    # the flattened index (first come first served)
    # FIME: wording
    packaged_flattened_index_dict = {}
    flattened_packaged_index_dict = {}
    counter = 0
    for i, energy in enumerate(list_energies):
        for j in range(len(energy)):
            packaged_flattened_index_dict[(i, j)] = counter
            flattened_packaged_index_dict[counter] = (i, j)
            counter += 1

    # find number of orbitals in each diagram
    diagram_num_orbs = [len(energies) for energies in list_energies]
    # find the offset of orbital indices
    orb_offsets = [sum(diagram_num_orbs[:i]) for i in range(len(list_energies))]

    # make lines
    dict_packaged_index_lines = {}
    if pairwise_weights is not None:
        # check if good value
        if not isinstance(pairwise_weights, dict):
            raise TypeError('pairwise_weights should be a dictionary')
        if any(weights.shape != (len(list_energies[i]), len(list_energies[j]))
               for (i, j), weights in pairwise_weights.items()):
            raise ValueError('Given weights must have the right shape')
        if any(np.all(weights < 0) for weights in pairwise_weights.values()):
            raise ValueError('Given weights must be positive')

        for (diagram_i_one, diagram_i_two), weights in pairwise_weights.items():
            line_threshold = 1e-5
            orb_pair_indices = zip(*np.where(weights > line_threshold))
            for (orb_i_one, orb_i_two) in orb_pair_indices:
                # get coordinates of the lines
                # NOTE: orbital indices are offset(by which diagram it belongs to)
                x_one, y_one= ax.lines[orb_i_one + orb_offsets[diagram_i_one]].get_data()
                x_two, y_two = ax.lines[orb_i_two + orb_offsets[diagram_i_two]].get_data()
                # make new line coordinates
                line_data = [[x_one[1], x_two[0]], [y_one[1], y_two[0]]]
                # make line
                line, = ax.plot(*line_data, color='blue', alpha=0.0)
                # save the line
                try:
                    dict_packaged_index_lines[(diagram_i_two, orb_i_two)].append((diagram_i_one, orb_i_one, line))
                except KeyError:
                    dict_packaged_index_lines[(diagram_i_two, orb_i_two)] = [(diagram_i_one, orb_i_one, line)]

                # TODO: add arrow to distinguish between the two directions
                # NOTE: if one orbital has significant contribution to another orbital,
                # this does not mean that the latter significantly contributes to the former

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
        for text in ax.texts:
            text.set_visible(False)
        visibility = ax.texts[index].get_visible()
        ax.texts[index].set_visible(not visibility)
        # clicking on weighted orbitals result in the display of all the contributing orbitals
        if pairwise_weights is not None:
            # FIXME: all of the degenerate orbitals are selected
            # turn off all contributions
            for line in ax.lines[sum(diagram_num_orbs):]:
                line.set_alpha(0.0)
            # turn on appropriate contributions
            diagram_i_to, orb_i_to = flattened_packaged_index_dict[index]
            try:
                for diagram_i_from, orb_i_from, line in dict_packaged_index_lines[(diagram_i_to, orb_i_to)]:
                    weight = pairwise_weights[diagram_i_from, diagram_i_to][orb_i_from, orb_i_to]
                    line.set_alpha(weight)
            except KeyError:
                print('There are no data on the contributions of the selected orbitals')
            # FIXME: if the same orbital is selected, then the contributions should turn off
        fig.canvas.draw()

    if pick_event_energy:
        fig.canvas.mpl_connect('pick_event', on_pick)
    # on_move_id = fig.canvas.mpl_connect('motion_notify_event', on_move)

def get_weights(coeffs_ab_mo=None, olp_ab_mo=None, option=0):
    if option == 0 and coeffs_ab_mo is not None:
        weights = np.ones(coeffs_ab_mo.shape)
    elif option == 0 and olp_ab_mo is not None:
        weights = np.ones(olp_ab_mo.shape)
    elif option == 1 and coeffs_ab_mo is not None:
        weights = np.abs(coeffs_ab_mo)
        # normalize
        weights /= np.max(weights, axis=1)[:, np.newaxis]
    elif option == 2 and coeffs_ab_mo is not None:
        weights = coeffs_ab_mo**2
        # normalize
        weights /= np.max(weights, axis=1)[:, np.newaxis]
    elif option == 3 and olp_ab_mo is not None:
        weights = olp_ab_mo**2
        weights /= np.max(weights, axis=1)[:, np.newaxis]
    return weights

# examples
# atomic basis - mo
hd = HortonData('CO.fchk', 'aambs.gbs')

fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))

coeff_ab_mo_sep = hd.coeff_ab_mo_sep
coeff_ab_mo = np.hstack(coeff_ab_mo_sep)

mo_energies_sep = hd.energies_sep
mo_energies = np.hstack(mo_energies_sep)

fock_ab_sep = fock_numerical(coeff_ab_mo_sep, mo_energies_sep)
ab_energies = np.hstack([np.diag(fock_ab) for fock_ab in fock_ab_sep])

occupations = np.hstack(hd.occupations_sep)

pairwise_weights = {(0, 1):get_weights(coeff_ab_mo, option=1)}
generate_all_mo_diagrams(fig, ax, [ab_energies, mo_energies], [np.array([1]*ab_energies.size), occupations], pairwise_weights=pairwise_weights)
plt.show()

# atomic basis - mo (separating out the atomic basis by atom type)
hd = HortonData('CO.fchk', 'aambs.gbs')

fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))

# Messy because of the alpha beta generalization
#ab_basis_map_sep = np.array(hd.ab_basis_map_sep)
#list_unique_atoms_sep = [set(ab_basis_map) for ab_basis_map in ab_basis_map_sep]
#ab_atom_indices_sep = [{unique_atom:np.where(ab_basis_map_sep == unique_atom) for unique_atom in list_unique_atoms} for list_unique_atoms in list_unique_atoms_sep]
# If we assume that the orbitals are spatial
ab_basis_map = np.array(hd.ab_basis_map_sep)[0]
list_unique_atoms = set(ab_basis_map)
ab_atom_indices = {unique_atom:np.where(ab_basis_map == unique_atom) for unique_atom in list_unique_atoms}

coeff_ab_mo_sep = hd.coeff_ab_mo_sep
coeff_ab_mo = np.hstack(coeff_ab_mo_sep)

mo_energies_sep = hd.energies_sep
mo_energies = np.hstack(mo_energies_sep)

fock_ab_sep = fock_numerical(coeff_ab_mo_sep, mo_energies_sep)
ab_energies = np.hstack([np.diag(fock_ab) for fock_ab in fock_ab_sep])
ab_energies_atom_separated = [ab_energies[ab_atom_indices[atom]] for atom in list_unique_atoms]
ab_occupations_atom_separated = [np.array([1]*len(ab_energies)) for ab_energies in ab_energies_atom_separated]

mo_occupations = np.hstack(hd.occupations_sep)

# pu the mo diagram at far right
pairwise_weights = {(i, len(list_unique_atoms)):get_weights(coeff_ab_mo[indices], option=1) for i,indices in ab_atom_indices.items()}

generate_all_mo_diagrams(fig,
                         ax,
                         ab_energies_atom_separated+[mo_energies],
                         ab_occupations_atom_separated+[mo_occupations],
                         pairwise_weights=pairwise_weights)
plt.show()

# quambo - mo
fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))

energies, occupations, quambo_energies, coeff_quambo_mo = get_quambo_data('ch4_svp_minao_iao.fchk')

pairwise_weights = {(0, 1):get_weights(coeff_quambo_mo, option=1)}
generate_all_mo_diagrams(fig, ax, [quambo_energies, energies], [occupations]*2, pairwise_weights=pairwise_weights)

plt.show()

# save plot in a eps file
# plt.savefig('ELD.eps')
