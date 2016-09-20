__author__ = 'kumru'

import numpy as np
from energy import fock_numerical
from quasibasis.quasi import QuasiTransformation, project
from quasibasis.wrapper_horton import HortonData

def get_quambo_data(fchkfile, cao_basis_files='aambs.gbs'):
    """ Retrieves the necessary QUAMBO data

    Think of it as a wrapper from dumbo.quasibasis to this module

    Parameters
    ----------
    fchkfile : str
        Location of the formatted checkpoint file used in Gaussian
    cao_basis_file : str
        Basis file (gbs or nwchem) of the reference "chemical" orbitals
        Default is AAMBS

    Returns
    -------
    mo_energies : np.ndarray
        Molecular orbital energies
        The alpha and the beta parts are stacked side by side
    occupations : np.ndarray
        Occupations of the molecular orbitals
        The alpha and the beta parts are stacked side by side
    quambo_energies : np.ndarray
        QUAMBO energies
        The alpha and the beta parts are stacked side by side
    coeff_quambo_mo : np.ndarray
        Transformation matrix from QUAMBO's to MO's
        The alpha and the beta parts are stacked side by side

    Note
    ----
    The alpha and beta parts are stacked side by side. Normally, they would be
    separated as a tuple (e.g. (alpha_part, beta_part)) but this added some degree
    of complexity to the tentative code. Now that the code is less tentative, we should
    add back the alpha and beta separation.

    """
    # extract information from fchkfile using horton wrapper in dumbo
    hd = HortonData(fchkfile, cao_basis_files)
    coeff_ab_mo_sep = hd.coeff_ab_mo_sep
    olp_ab_ab_sep = hd.olp_ab_ab_sep
    olp_cao_ab_sep = hd.olp_cao_ab_sep
    olp_cao_cao_sep = hd.olp_cao_cao_sep
    occupations_sep = hd.occupations_sep
    mo_energies_sep = hd.energies_sep

    # generate QUAMBO orbitals and energies
    fock_quambo_sep = []
    coeff_quambo_mo_sep = []
    # alpha and beta parts (if they exist) are separated in dumbo
    # we can loop over them to make sure we create alpha QUAMBO's
    # and beta QUAMBO's
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
        # container class for the quambo related methods
        quasi = QuasiTransformation(coeff_ab_mo,
                                    olp_ab_ab,
                                    olp_cao_ab,
                                    olp_cao_cao,
                                    occupations.astype(bool))
        # generate coefficients
        coeff_ab_quambo = quasi.quambo()
        # generate fock matrix F
        # F_{ij} = <\phi_i | f | \phi_j> where \phi is an orbital
        fock_ab = fock_numerical([coeff_ab_mo], [mo_energies])[0]
        # diagonals of fock_ab are the energies of the atomic basis functions
        fock_quambo = coeff_ab_quambo.T.dot(fock_ab).dot(coeff_ab_quambo)
        # diagonals of fock_quambo are the energies of the QUAMBO's
        fock_quambo_sep.append(fock_quambo)
        # get transformation matrix from quambo to mo
        olp_quambo_quambo = coeff_ab_quambo.T.dot(olp_ab_ab).dot(coeff_ab_quambo)
        olp_quambo_mo = coeff_ab_quambo.T.dot(olp_ab_ab).dot(coeff_ab_mo)
        coeff_quambo_mo_sep.append(project(olp_quambo_quambo, olp_quambo_mo))
    # collapse alpha and beta separation
    mo_energies = np.hstack(mo_energies_sep)
    occupations = np.hstack(occupations_sep)
    quambo_energies = np.hstack([np.diag(fock) for fock in fock_quambo_sep])
    coeff_quambo_mo = np.hstack(coeff_quambo_mo_sep)
    return mo_energies, occupations, quambo_energies, coeff_quambo_mo


def generate_one_diagram(energies, occupations, degens, ax, center=0, line_width=1.0, line_sep=0.1):
    """ Generates one MO diagram

    Parameters
    ----------
    energies : iterable
        Energies of all of the orbitals in the diagram
    occupations : iterable
        Occupations of all of the orbitals in the diagram
    degens : iterable
        Degeneracies of the orbitals in the diagram
    ax : matplotlib.Axis
        The object that we will be storing the diagram
    center : float
        Horizontal offset that determines the center of the diagram
    line_width : float
        Width of the line that describes each orbital
    line_sep : float
        Distance between two adjacent (degenerate) orbital lines

    Note
    ----
    The energies and occupations need to be organized so that we group together
    degenerate orbitals. If the orbitals are degenerate, then the orbital lines
    will be placed side by side. If they are not, then they will be placed
    vertically. If adjacent orbitals are very close in energy and are not degenerate,
    then it will be difficult to distinguish between the two.
    e.g. energies = [0.0012, 0.0011]
         occupations = [2, 1]
         degens = [1, 1]
    will have two orbitals stacked vertically where as
         energies = [0.0012, 0.0011]
         occupations = [2, 1]
         degens = [2]
    will place the two orbitals side by side.
    """
    # display
    counter = 0
    for degen in degens:
        # within each degeneracy
        # position of the left most point of the left most orbital line (of this degeneracy)
        x_leftmost = -(degen//2)*(line_width+line_sep) - (degen%2)*line_width/2.0
        # left positions of all of the orbital lines (of this degeneracy)
        x_inits = [x_leftmost + i*(line_width+line_sep) + center for i in range(degen)]
        # right positions
        x_finals = [x_leftmost+line_width + i*(line_width+line_sep) + center for i in range(degen)]
        # left and right positions
        xs_degen = zip(x_inits, x_finals)

        # energies (height) of the orbital lines
        ys = energies[counter:counter+degen]
        # start y and final y positions
        ys_degen = zip(ys, ys)

        # occupations (divided by degeneracy)
        occs_degen = occupations[counter:counter+degen]

        # create a line for each orbital
        for x, y, occ in zip(xs_degen, ys_degen, occs_degen):
            # different color depending on the occupation
            if occ > 0.01:
                line, = ax.plot(x, y, color='green', alpha=1.0, picker=10)
            else:
                line, = ax.plot(x, y, color='red', alpha=1.0, picker=10)
            # setting for the annotation box
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # construct annotation (label of the orbital)
            text = ax.text(x[1]+0.2, y[1]+0.4, y[0], fontsize=14, verticalalignment='top', bbox=props)
            # make the annotation invisible
            text.set_visible(False)
        counter += degen

def generate_all_mo_diagrams(fig, ax, list_energies, list_occupations, pick_event_energy=True, pairwise_weights=None):
    """ Create MO diagrams side by side

    The different MO diagrams can be linked to one another by providing the weights of contribution

    Parameters
    ----------
    fig : matplotlib.Figure
        Matplotlib object that contains the whole MO diagrams
    ax : matplotlib.Axis
        Matplotlib object that contains the coordinate and axis information
    list_energies : list of list
        List of list of orbital energies
        Each list of orbital energies correspond to a different MO diagram
    list_occupations : list of list
        List of list of orbital occupations
        Each list of orbital occupations correspond to a different MO diagram
    pick_event_energy : bool
        Flag for using pick event (clicking orbital line)
    pairwise_weights : Dictionary of tuple of integers to np.ndarray
        Key is the tuple of the two diagrams that are being compared
        The orbitals of the first diagram are the contributing orbitals
        The orbitals of the second diagram are the resulting orbitals
        i.e. the contributions of the orbitals from the first diagram is given for each orbital
        of the second diagram
        The weights describe the contribution of the orbitals from the first diagram (rows) to
        the orbitals of the second diagram (columns)

    """
    list_sorted_energies = []
    list_sorted_occupations = []
    list_sorted_degens = []

    # tolerance for degeneracy (if the energy difference is less than this value
    # then the orbitals are degenerate)
    tol_degen = 0.01
    # tolerance for the connectivity weights (if two orbitals from different MO diagrams
    # have a connectivity weight (given by pairwise_weights) is greater than this value,
    # then a line is created connecting these two orbitals)
    connectivity_threshold = 1e-5

    # for each mo diagram, sort the energies and find the degeneracies
    for i, energies, occupations in zip(range(len(list_energies)), list_energies, list_occupations):
        # sort energies (smallest to largest)
        sort_indices = np.argsort(energies)
        # sort the energies and occupations accordingly
        energies = energies[sort_indices]
        occupations = occupations[sort_indices]
        # sort the weights accordingly
        new_pairwise_weights = {}
        if pairwise_weights is not None:
            for pair, weights in pairwise_weights.items():
                if pair[1] == i:
                    new_pairwise_weights[pair] = weights[:, sort_indices]
                if pair[0] == i:
                    new_pairwise_weights[pair] = weights[sort_indices, :]
                else:
                    new_pairwise_weights[pair] = weights
            pairwise_weights = new_pairwise_weights

        # find energy difference between (energetically) adjacent orbitals
        energy_deltas = np.diff(energies)
        energy_indices = [i for i,j in enumerate(energy_deltas) if j > tol_degen]
        # range that corresponds to each "degeneracy box". If many orbitals have
        # energies that are between two adjacent energy_bin values, then all these
        # orbitals will be considered degenerate
        # e.g. energy_bins = [1.00, 1.1, 1.150, 1.158]
        # and we have orbitals with energies 1.150, 1.155 and 1.158, all of
        # these orbitals will be considered degenerate
        energy_bins = [energies[0]]+[energies[i+1] for i in energy_indices]+[energies[-1]+1]
        # number of orbitals in each "degeneracy bin"
        degens = np.histogram(energies, bins=energy_bins)[0]

        list_sorted_energies.append(energies)
        list_sorted_occupations.append(occupations)
        list_sorted_degens.append(degens)

    # find diagram width and spacing parameters
    max_degens = [max(degens) for degens in list_sorted_degens]
    # length of each orbital line
    line_width = 1.0
    # distance between degenerate orbital lines
    line_sep = 0.1
    # width of each diagram
    diagram_widths = [(line_width+line_sep)*max_degen-line_sep for max_degen in max_degens]
    # distance between adjacent diagrams
    diagram_sep = 1.0
    # total width of the MO diagram
    total_width = sum(diagram_widths) + diagram_sep*(len(diagram_widths)-1)

    # generate each diagram
    for i, energies, occupations, degens in zip(range(len(diagram_widths)),
                                                list_sorted_energies,
                                                list_sorted_occupations,
                                                list_sorted_degens):
        # position of the center of the diagram
        center = -total_width/2.0 + sum(diagram_widths[:i]) + i*diagram_sep + diagram_widths[i]/2.0
        # generate diagram
        generate_one_diagram(energies, occupations, degens, ax, center=center, line_width=line_width, line_sep=line_sep)

    # matplotlib.Axes stores everything in a single list. This means that all the
    # orbitals that we've made (even the ones that come from different MO diagrams)
    # are simply stored in a single list.
    # Here, we make a translator that goes from the index in this list to the
    # more readable tuple of indices (one for the MO diagram and one for the orbital).
    # and vice versa
    packaged_flattened_index_dict = {}
    flattened_packaged_index_dict = {}
    counter = 0
    for i, energy in enumerate(list_energies):
        for j in range(len(energy)):
            packaged_flattened_index_dict[(i, j)] = counter
            flattened_packaged_index_dict[counter] = (i, j)
            counter += 1

    # number of orbitals in each diagram
    diagram_num_orbs = [len(energies) for energies in list_energies]
    # find the offset of orbital indices
    orb_offsets = [sum(diagram_num_orbs[:i]) for i,j in enumerate(list_energies)]

    # Make lines between orbitals from different MO diagrams
    dict_packaged_index_lines = {}
    if pairwise_weights is not None:
        # given pairwise_weights is good
        if not isinstance(pairwise_weights, dict):
            raise TypeError('pairwise_weights should be a dictionary')
        if any(weights.shape != (len(list_energies[i]), len(list_energies[j]))
               for (i, j), weights in pairwise_weights.items()):
            raise ValueError('Given weights must have the right shape')
        if any(np.all(weights < 0) for weights in pairwise_weights.values()):
            raise ValueError('Given weights must be positive')

        # for each given MO diagram pair
        for (diagram_i_one, diagram_i_two), weights in pairwise_weights.items():
            # find orbital pairs that have weights greater than the threshold
            orb_pair_indices = zip(*np.where(weights > connectivity_threshold))
            # create line between these orbitals
            for (orb_i_one, orb_i_two) in orb_pair_indices:
                # get coordinates of the lines
                # NOTE: orbital indices are offset (by which diagram it belongs to)
                # FIXME: probably should use the packaged_flattened_index_dict.
                #        the following should be implemented and checked
                # new_index_one = packaged_flittened_index_dict[(diagram_i_one, orb_i_one)]
                # x_one, y_one= ax.lines[new_index_one].get_data()
                # new_index_two = packaged_flittened_index_dict[(diagram_i_one, orb_i_one)]
                # x_two, y_two = ax.lines[new_index_two].get_data()
                #         If this works, we should delete the orb_offsets
                x_one, y_one= ax.lines[orb_i_one + orb_offsets[diagram_i_one]].get_data()
                x_two, y_two = ax.lines[orb_i_two + orb_offsets[diagram_i_two]].get_data()
                # make new line coordinates
                line_data = [[x_one[1], x_two[0]], [y_one[1], y_two[0]]]
                # make line
                line, = ax.plot(*line_data, color='blue', alpha=0.0)
                # save the line
                try:
                    # if the key exists
                    dict_packaged_index_lines[(diagram_i_two, orb_i_two)].append((diagram_i_one, orb_i_one, line))
                except KeyError:
                    # else
                    dict_packaged_index_lines[(diagram_i_two, orb_i_two)] = [(diagram_i_one, orb_i_one, line)]

                # TODO: add arrow to distinguish between the two directions
                # NOTE: if one orbital has significant contribution to another orbital,
                # this does not mean that the latter significantly contributes to the former

    # set x range
    ax.set_xlim([-total_width/2.0-0.5, total_width/2.0+0.5])
    # set y range
    min_energies = [min(energies) for energies in list_energies]
    max_energies = [max(energies) for energies in list_energies]
    ax.set_ylim([min(min_energies)-1, max(max_energies)+1])
    # remove x-axis label
    ax.xaxis.set_visible(False)
    # add label to y-axis
    ax.set_ylabel("Energy (Hartree)")

    # set event
    def on_pick(event):
        """ Set event for the clicking of orbital line

        Clicking should result in the annotation (orbital energy) and
        the connectivity between orbitals of different MO diagram to appear

        """
        # the line object (that got clicked)
        thisline = event.artist
        # index of this line
        index = ax.lines.index(thisline)
        # turn off all annotations
        for text in ax.texts:
            text.set_visible(False)
        # turn on the annotation for the selected line
        visibility = ax.texts[index].get_visible()
        ax.texts[index].set_visible(not visibility)
        # display all the contributing (connected) orbitals
        if pairwise_weights is not None:
            # FIXME: all of the degenerate orbitals are selected
            # turn off all contributions
            for line in ax.lines[sum(diagram_num_orbs):]:
                line.set_alpha(0.0)
            # turn on appropriate contributions
            diagram_i_to, orb_i_to = flattened_packaged_index_dict[index]
            try:
                # set intensity (transparency) of the line by the weight
                for diagram_i_from, orb_i_from, line in dict_packaged_index_lines[(diagram_i_to, orb_i_to)]:
                    weight = pairwise_weights[diagram_i_from, diagram_i_to][orb_i_from, orb_i_to]
                    line.set_alpha(weight)
            except KeyError:
                print('There are no data on the contributions of the selected orbitals')
            # FIXME: if the same orbital is selected, then the contributions should turn off
        fig.canvas.draw()

    if pick_event_energy:
        fig.canvas.mpl_connect('pick_event', on_pick)

def get_weights(coeffs_ab_mo=None, olp_ab_mo=None, option=0):
    """ Constructs the connectivity weights between the given two orbital sets

    Parameters
    ----------
    coeffs_ab_mo : np.ndarray
        Transformation matrix from the atomic basis functions to the molecular
        orbitals
    olp_ab_mo : np.ndarray
        Overlap between the atomic basis functions and the molecular orbitals
    option : int
        0 results in complete bipartite connectivity (orbitals from one set is connected
        to all of the orbitals of another set)
        1 results in connectivity from the absolute value of the transformation matrix
        (larger contribution results in greater weight)
        2 results in connectivity from the squared value of the transformation matrix
        (larger contribution results in greater weight)
        3 results in connectivity from the squared value of the overlap matrix
        (larger contribution results in greater weight)

    Returns
    -------
    weights : np.ndarray
        Weights for the connectivity between orbitals from different sets.
        Row indices correspond to "atomic basis functions" (rows of coeffs_ab_mo
        and olp_ab_mo)
        Column indices correspond to "molecular orbitals" (columns of coeffs_ab_mo
        and olp_ab_mo)

    """
    if option == 0:
        if coeffs_ab_mo is not None:
            weights = np.ones(coeffs_ab_mo.shape)
        elif olp_ab_mo is not None:
            weights = np.ones(olp_ab_mo.shape)
        else:
            raise ValueError('Not enough information (coeffs_ab_mo or olp_ab_mo) is '
                             'given to construct the weights')
    elif option == 1:
        if coeffs_ab_mo is None:
            raise ValueError('coeffs_ab_mo needs to be provided')
        weights = np.abs(coeffs_ab_mo)
        # normalize
        weights /= np.max(weights, axis=1)[:, np.newaxis]
    elif option == 2:
        if coeffs_ab_mo is None:
            raise ValueError('coeffs_ab_mo needs to be provided')
        weights = coeffs_ab_mo**2
        # normalize
        weights /= np.max(weights, axis=1)[:, np.newaxis]
    elif option == 3:
        assert olp_ab_mo is not None
        if olp_ab_mo is None:
            raise ValueError('olp_ab_mo needs to be provided')
        weights = olp_ab_mo**2
        weights /= np.max(weights, axis=1)[:, np.newaxis]
    return weights
'''
import matplotlib.pyplot as plt

# examples
#### atomic basis - mo ####
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

#### atomic basis - mo (separating out the atomic basis by atom type) ####
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

# the mo diagram was put on the farther right side
pairwise_weights = {(i, len(list_unique_atoms)):get_weights(coeff_ab_mo[indices], option=1) for i,indices in ab_atom_indices.items()}

generate_all_mo_diagrams(fig,
                         ax,
                         ab_energies_atom_separated+[mo_energies],
                         ab_occupations_atom_separated+[mo_occupations],
                         pairwise_weights=pairwise_weights)
plt.show()

# you can also manually reorder the diagrams to put the mo diagram in the middle
fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
pairwise_weights[(0,1)], pairwise_weights[(2,1)] = pairwise_weights[(0,2)], pairwise_weights[(1,2)]
del pairwise_weights[(0,2)]
del pairwise_weights[(1,2)]
generate_all_mo_diagrams(fig,
                         ax,
                         [ab_energies_atom_separated[0], mo_energies, ab_energies_atom_separated[1]],
                         [ab_occupations_atom_separated[0], mo_occupations, ab_occupations_atom_separated[1]],
                         pairwise_weights=pairwise_weights)
plt.show()
# but this looks weird because (well, just see)

#### quambo - mo ####
fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))

energies, occupations, quambo_energies, coeff_quambo_mo = get_quambo_data('ch4_svp_minao_iao.fchk')

pairwise_weights = {(0, 1):get_weights(coeff_quambo_mo, option=1)}
generate_all_mo_diagrams(fig, ax, [quambo_energies, energies], [occupations]*2, pairwise_weights=pairwise_weights)

plt.show()

# save plot in a eps file
# plt.savefig('ELD.eps')
'''
