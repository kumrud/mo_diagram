__author__ = 'kumru'

import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
from energy import fock_horton, fock_numerical
from quasibasis.quasi import QuasiTransformation, project
from quasibasis.wrapper_horton import HortonData

class MoDiagram(object):
    '''
    Generate Molecular Orbital Diagram using matplotlib.

    Attributes
    ----------
    _mo_energies: np.ndarray
        Molecular/QAO energies
    _ab_energies: np.ndarray
        Molecular/QAO energies
    _coeff_ab_mo: np.ndarray
        Transformation matrix from atomic basis to molecular/qa orbitals
    _occupations: np,ndarray of {float, int}
        Occupations of molecular/qa orbitals

    Raises
    ------
    AssertionError
        If mo_energies is not one dimentional numpy array
        If coeff_ab_mo is not two dimentional numpy array
        If occupations is not one dimentional numpy array
        If number of occupations not consistent for mo_energies
        If occupations is given as float

    '''
    def __init__(self, mo_energies, ab_energies, coeff_ab_mo, occupations):
        
        # Assert input quality
        assert (isinstance(mo_energies, np.ndarray) and
                len(mo_energies.shape) == 1),\
            'Mo energies is not given as a one dimensional numpy array'
        assert (isinstance(ab_energies, np.ndarray) and
                len(ab_energies.shape) == 1),\
            'Ao energies is not given as a one dimensional numpy array'
        assert (ab_energies.shape[0] <= mo_energies.shape[0]),\
            'Ao number is greater than Mo'
        assert (isinstance(coeff_ab_mo, np.ndarray) and
            len(coeff_ab_mo.shape) == 2), \
            'Coefficient matrix given is not a two dimentional array'
        assert (isinstance(occupations, np.ndarray) and
                len(occupations.shape)) == 1,\
            'Occupations is not given as a one dimensional numpy array'
        assert (len(occupations) == len(mo_energies)),\
            'Number of occupations is not equal to number of molecular orbitals'
        assert occupations.dtype in [float],\
            'Occupancies is not float'

        # Initialize private variables
        self._mo_energies = mo_energies
        self._ab_energies = ab_energies
        self._coeff_ab_mo = coeff_ab_mo
        self._occupations = occupations

    @property
    def mo(self):
        '''Molecular/QAO energies
        '''
        return self._mo_energies

    @property
    def ao(self):
        '''Atomic orbital energies
        '''
        return self._ab_energies

    @property
    def coeff(self):
        '''Coefficient matrix from atomic basis to molecular
        '''
        return self._coeff_ab_mo

    @property
    def occupations(self):
        '''Occupation of molecular orbitals
        '''
        return self._occupations

    @property
    def degen_ao(self):
        '''Degeneracy atomic orbital
        '''
        degen_ao = degenerate(self._ab_energies)
        return degen_ao

    @property
    def degen_mo(self):
        '''Degeneracy atomic orbital
        '''
        degen_mo = degenerate(self._mo_energies)
        return degen_mo

    @property
    def num_ao(self):
        ''' Number of atomic orbitals
        '''
        return self._ab_energies[0]

    @property
    def num_mo(self):
        ''' Number of atomic orbitals
        '''
        return self._mo_energies[0]

    # def __getattr__(self, attr):
    #     ''' Return class attribute
    #     '''
    #     return self._dict__[attr]


class OrbitalPlot(object):

    ''' Manages matplotlib ugliness

    Attributes
    ----------
    x_lenght: float
        Lenght of lines
    x_sep: float
        Dinstance between two lines
    y_length: float
        Thickness of lines

    
    Methods
    -------
    line_data
        Coordinated for lines
    make_line
        Creates lines as matplotlib.collections.BrokenBarHCollection instance
        TODO: specify graph type (ao, mo or quambo)
            Selecting orbitals to graph
    
    TODO:line_connect, How to connect two lines on different graphs


    '''
    def __init__(self, data, x_length=1.0, x_sep=0.1, y_length=0.1):
        self._x_length = x_length
        self._x_sep = x_sep
        self._y_length = y_length

        if not isinstance(data, MoDiagram):
            raise TypeError('Given data has to be an MoDiagram instance')
        self._data = data

    @property
    def x_length(self):
        ''' Length of a line
        '''
        return self._x_length

    @property
    def x_sep(self):
        ''' Separation between each degenerate level
        '''
        return self._x_sep

    @property
    def x_shift(self):
        ''' Length of which a line will be shifted in case of degeneracy
        '''
        return self._x_length + self._x_sep

    @property
    def y_length(self):
        ''' Thickness of the line
        '''
        return self._y_length

    @property
    def occupations(self):
        return self._data.occupations

    def line_data(self, energies):
        '''
        Sets x and y coordinates for lines

        Parameters
        ----------
        energies: np.ndarray
            Molecular/Atomic orbital energies

        Returns
        -------
        x_coor, y_coor: list of tuples
            Start and length of x and y coordinates
        line_color: tuple
            Color lines according to occupation

        Example
        -------
        energies = [0 1 1 2 3 3 3]
        x_coor = [(0.55, 1), (0.0, 1), (1.1, 1), (0.55, 1), (-0.44999999999999996, 1),
            (0.6500000000000001, 1), (1.7500000000000002, 1)]
        y_coor = [(0, 0.01), (1, 0.01), (1, 0.01), (2, 0.01), (3, 0.01), (3, 0.01), (3, 0.01)]

        '''
        degens = degenerate(energies)

        # leftmost x coordinates for a line
        x_start = [(d & 2) * self.x_shift / 2 - self.x_shift * (d - 1) // 2 for d in degens]

        # unpack histogram degeracy to match energy levels
        x_data = []
        for i,j in zip(degens, x_start):
            if i != 1: x_data += [j + k * (self.x_shift) for k in range(int(i))]
            if i == 1: x_data.append(j)

        # zip start and length of x and y coordinates
        x_coor = [(i, self.x_length) for i in x_data]
        y_coor = [(i, self.y_length) for i in energies]

        # line colors as occupation
        line_color = []
        for i in self.occupations:
            if i == 2: line_color.append("green")
            elif i == 1: line_color.append("yellow")
            elif i == 0: line_color.append("red")
            # Should not happen
            else: line_color.append("blue")
        tuple(line_color)

        return x_coor, y_coor, line_color

    def make_line(self, *args):
        '''TODO: Check arguments
            shift multiple plots
        '''
        fig, ax = plt.subplots()
        ax.set_xlim(-4,4)
        ax.set_ylim(-21,2)
        for energy in args:
            x, y, color = self.line_data(energy)

            line = [ax.broken_barh([x[i]], y[i], facecolor=color[i]) for i,j in enumerate(energy)]
            return line

    def make_graph(self, *args):
        pass


def degenerate(energies, tol=0.01):
    '''
    Sets degenerate energy levels using histogram

    Parameters
    ----------
    energies: np.ndarray
        Molecular/Atomic orbital energies
    tol: float
        Tolerance for degeneracy between two energies

    Returns
    -------
    degens: np.ndarray
        Degeneracy of orbitals

    Example
    -------
    energies = np.array([2, 2, 5, 6, 7, 7, 8])
    degens = array([2, 1, 1, 2, 1])

    '''

    energy_diff = np.diff(energies)
    # a new line after these indices
    degen_indices = [i for i,j in enumerate(energy_diff) if j>tol]
    print degen_indices
    # histogram range right non-inclusive
    energy_bins = [energies[0]]+[energies[i+1] for i in degen_indices]+[energies[-1]+1]
    print energy_bins
    print energies
    degens = np.histogram(energies, bins=energy_bins)[0]
    return degens

