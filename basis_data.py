""" Classes for containing data for orbitals

Classes
-------
OrbitalData(energies, occupations=None, indices_subsystem=None, indices_orbtype=None)
    Container for orbital energies, occupations, subsystem indices and orbtype indices
"""
import numpy as np


class OrbitalData(object):
    """ Data concerning the orbitals of a basis set

    Attributes
    ----------
    energies : list of float
        Energy of each orbital
    occupations : list of float
        Occupation of each orbital
    indices_subsystem : list of int
        Index for the subsystem of each orbital
        For example, atomic orbitals would have indices that correspond to different atoms
    indices_orbtype : list of int
        Index for the type of orbital
    """
    def __init__(self, energies, occupations=None, indices_subsystem=None, indices_orbtype=None):
        """
        Parameters
        ----------
        energies : list, tuple, np.ndarray
            Energies of the orbitals
        occupations : list, tuple, np.ndarray
            Occupations of orbitals
            Default is a list with appropriate number of np.nan
        indices_subsystem : list, tuple, np.ndarray
            Indices that describe which subsystem each orbital belongs
            Default is a list with appropriate number of np.nan
        indices_orbtype : list, tuple, np.ndarray
            Indices that describes orbital types of each orbital
            Default is a list with appropriate number of np.nan
        """
        if not isinstance(energies, (list, tuple, np.ndarray)):
            raise TypeError('Given energies must be a list or tuple or numpy array')

        if occupations is None:
            occupations = [np.nan] * len(energies)
        elif not isinstance(occupations, (list, tuple, np.ndarray)):
            raise TypeError('Given occupations must be a list or tuple or numpy array')

        if indices_subsystem is None:
            indices_subsystem = [np.nan] * len(energies)
        elif not isinstance(indices_subsystem, (list, tuple, np.ndarray)):
            raise TypeError('Given indices_subsystem must be a list or tuple or numpy array')

        if indices_orbtype is None:
            indices_orbtype = [np.nan] * len(energies)
        elif not isinstance(indices_orbtype, (list, tuple, np.ndarray)):
            raise TypeError('Given indices_orbtype must be a list or tuple or numpy array')

        if len(energies) != len(occupations) != len(indices_subsystem) != len(indices_orbtype):
            raise ValueError('Given number of orbital energies, occupations, subsystem indices and '
                             'orbital indices do not match with one another')

        self._energies = np.array(energies, dtype=float)
        self._occupations = np.array(occupations, dtype=float)
        self._indices_subsystem = np.array(indices_subsystem)
        self._indices_orbtype = np.array(indices_orbtype)

    @property
    def num_orbs(self):
        """ Number of orbitals
        """
        return self._energies.size

    def get_values(self, indices=None, occ_lower=0, occ_upper=2, included_subsystems=None,
                   included_orbtypes=None):
        """ Select values using various criteria

        Parameters
        ----------
        indices : np.ndarray
            List of orbital indices that selects the orbitals
            Default is all orbital indices
        occ_lower : int
            Lower bound of allowed occupation
        occ_upper : int
            Upper bound of allowed occupation
        included_subsystems : np.ndarray of int
            List of allowed subsystem indices
        included_orbtypes : np.ndarray of int
            List of allowed orbtype indiceds

        Returns
        -------
        values : tuple
            tuple of energies, occupations, indices_subsystem and indices_orbtype that satisfies the
            given conditions
        """
        sel_indices = np.zeros(self.num_orbs, dtype=bool)
        # indices
        if indices is None:
            indices = np.arange(self.num_orbs)
        elif not isinstance(indices, np.ndarray):
            raise TypeError('Indices must be given as a numpy array')
        sel_indices[indices] = True
        # occupations
        if not (isinstance(occ_lower, (int, float)) and isinstance(occ_upper, (int, float))):
            raise TypeError('Occupation boundary must be given as an integer or float')
        sel_indices = np.logical_and(sel_indices, occ_lower <= self._occupations)
        sel_indices = np.logical_and(sel_indices, self._occupations <= occ_upper)
        # subsystem indices
        if included_subsystems is None:
            included_subsystems = self._indices_subsystem
        elif not (isinstance(included_subsystems, np.ndarray)
                  and np.issubdtype(included_subsystems.dtype, np.integer)):
            raise TypeError('Included subsystems must be given as a numpy array of integers')
        sel_indices = np.logical_and(sel_indices,
                                     np.in1d(self._indices_subsystem, included_subsystems))
        # orbital type indices
        if included_orbtypes is None:
            included_orbtypes = self._indices_orbtype
        elif not (isinstance(included_orbtypes, np.ndarray)
                  and np.issubdtype(included_orbtypes.dtype, np.integer)):
            raise TypeError('Included subsystems must be given as a numpy array of integers')
        sel_indices = np.logical_and(sel_indices,
                                     np.in1d(self._indices_orbtype, included_orbtypes))

        return (self._energies[sel_indices],
                self._occupations[sel_indices],
                self._indices_subsystem[sel_indices],
                self._indices_orbtype[sel_indices])

    def get_energies(self, indices=None, occ_lower=0, occ_upper=2, included_subsystems=None,
                     included_orbtypes=None):
        """ Select orbital energies using various criteria

        Parameters
        ----------
        indices : np.ndarray
            List of orbital indices that selects the orbitals
            Default is all orbital indices
        occ_lower : int
            Lower bound of allowed occupation
        occ_upper : int
            Upper bound of allowed occupation
        included_subsystems : np.ndarray of int
            List of allowed subsystem indices
        included_orbtypes : np.ndarray of int
            List of allowed orbtype indiceds

        Returns
        -------
        energies : np.ndarray
            orbital energies that satisfies the given conditions
        """
        return self.get_values(indices, occ_lower, occ_upper, included_subsystems,
                               included_orbtypes)[0]

    def get_occupations(self, indices=None, occ_lower=0, occ_upper=2, included_subsystems=None,
                        included_orbtypes=None):
        """ Select orbital occupations using various criteria

        Parameters
        ----------
        indices : np.ndarray
            List of orbital indices that selects the orbitals
            Default is all orbital indices
        occ_lower : int
            Lower bound of allowed occupation
        occ_upper : int
            Upper bound of allowed occupation
        included_subsystems : np.ndarray of int
            List of allowed subsystem indices
        included_orbtypes : np.ndarray of int
            List of allowed orbtype indiceds

        Returns
        -------
        occupations : np.ndarray
            orbital occupations that satisfies the given conditions
        """
        return self.get_values(indices, occ_lower, occ_upper, included_subsystems,
                               included_orbtypes)[1]

    def get_indices_subsystem(self, indices=None, occ_lower=0, occ_upper=2,
                              included_subsystems=None, included_orbtypes=None):
        """ Select orbital indices_subsystem using various criteria

        Parameters
        ----------
        indices : np.ndarray
            List of orbital indices that selects the orbitals
            Default is all orbital indices
        occ_lower : int
            Lower bound of allowed occupation
        occ_upper : int
            Upper bound of allowed occupation
        included_subsystems : np.ndarray of int
            List of allowed subsystem indices
        included_orbtypes : np.ndarray of int
            List of allowed orbtype indiceds

        Returns
        -------
        indices_subsystem : np.ndarray
            orbital subsystem indices that satisfies the given conditions
        """
        return self.get_values(indices, occ_lower, occ_upper, included_subsystems,
                               included_orbtypes)[2]

    def get_indices_orbtype(self, indices=None, occ_lower=0, occ_upper=2, included_subsystems=None,
                            included_orbtypes=None):
        """ Select orbital indices_subsystem using various criteria

        Parameters
        ----------
        indices : np.ndarray
            List of orbital indices that selects the orbitals
            Default is all orbital indices
        occ_lower : int
            Lower bound of allowed occupation
        occ_upper : int
            Upper bound of allowed occupation
        included_subsystems : np.ndarray of int
            List of allowed subsystem indices
        included_orbtypes : np.ndarray of int
            List of allowed orbtype indiceds

        Returns
        -------
        indices_orbtype : np.ndarray
            orbital orbtype indices that satisfies the given conditions
        """
        return self.get_values(indices, occ_lower, occ_upper, included_subsystems,
                               included_orbtypes)[3]


class BasisData(object):
    """ Data concerning the relationship between different orbital basis sets

    Attribute
    ---------
    orbital_data : list of OrbitalData
        List of orbital data
    coeffs : dict of index pair to np.ndarray
        Dictionary from pair of orbital data indices to coefficient matrix
    """
    def __init__(self, orbital_data, coeffs={}):
        """
        Parameters
        ----------
        orbital_data : list of OrbitalData
            Orbital data containers
        coeffs : dict of np.ndarray
            Dictionary
        """
    pass
