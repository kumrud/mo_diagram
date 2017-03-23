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
                len(coeff_ab_mo.shape) == 2),\
            'Coefficient matrix given is not a two dimentional array'
        assert (isinstance(occupations, np.ndarray) and
                len(occupations.shape) == 1,\
            'Occupations is not given as a one dimensional numpy array',\
        assert (len(occupations) == len(mo_energies)),\
            'Number of occupations is not equal to number of molecular orbitals'
        assert occupations.dtype in [float],\
            'Occupancies is not float'

        # Initialize private variables
        self._mo_energies = mo_energies
        self._ab_energies = ab_energies
        self._coeff_ab_mo = coeff_ab_mo
        self._occupations = occupations
        self._hard_way = False

        @property
        def mo_energies(self):
            '''Molecular/QAO energies
            '''
            if not self._hard_way:
                return self._mo_energies

        @property
        def ab_energies(self):
            '''Atomic orbital energies
            '''
            if not self._hard_way:
                return self._ab_energies

        @property
        def coeff_ab_mo(self):
            '''Atomic orbital energies
            '''
            if not self._hard_way:
                return self._coeff_ab_mo

        @property
        def occupations(self):
            '''Atomic orbital energies
            '''
            return self._occupations

        @property
        def degen_ao(self):
            '''Degeneracy atomic orbital
            '''
            degen_ao = degen(ab_energies)
            return degen_ao

        @property
        def degen_mo(self):
            '''Degeneracy atomic orbital
            '''
            degen_mo = degen(self._mo_energies)
            return degen_mo

        @property
        def num_ao:
            ''' Number of atomic orbitals
            '''
            return self._ab_energies[0]

        @property
        def num_mo:
            ''' Number of atomic orbitals
            '''
            return self._mo_energies[0]

        def degenerate(self, energies, tol=0.01):
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
            degen: np.ndarray
                Degeneracy of orbitals

            Example
            -------
            energies = np.array([2, 2, 5, 6, 7, 7, 8])
            degens = array([2, 1, 1, 2, 1])

            '''

            energy_diff = np.diff(energies)
            # a new line after these indices
            degen_indices = [i for i,j in enumerate(energy_diff) if j>tol]
            # histogram range right non-inclusive 
            energy_bins = [energies[0]]+[energies[i+1] for i in degen_indices]+[energies[-1]+1]
            degen = np.histogram(energies, bins=energy_bins)[0]

            return degen



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
    
    TODO:
    line_connect
        How to connect two lines on different graphs

    '''
    def __init__(self, x_length=1.0, x_sep=0.1, y_length=0.01):
        self._x_length = x_length
        self._x_sep = x_sep
        self._y_length = y_length

    @property
    def x_length(self):
        return self._x_length

    @property
    def x_sep(self):
        return self._x_sep

    def line_data(self, degens, energies):
        x_start = -(degens//2)*(self.x_length+self.x_sep) - (degens%2)*self.x_length/2.0

        x_data = []
        for i,j in zip(degens, x_start):
            if i != 1: x_data += [j + k * (self.x_length + self.x_sep) for k in range(int(i))]
            if i == 1: x_data.append(j)

        x_coor = [(i, self.x_length) for i in x_data]
        y_coor = [(i, self.y_length) for i in energies]

        return x_coor, y_coor

    def make_line(self, energies):
        degens = degenerate(energies)
        x_coor, y_coor = line_data(degens, energies)

        fig, ax = plt.subplots()
        line = [ax.broken_barh([x_coor[i]], y_coor[i]) for i,j in enumerate(energies)]
        return line


    
