from horton import *
import numpy as np
from quasibasis.wrapper_horton import HortonData
from quasibasis.quasi import QuasiTransformation, project

log._level=0
np.set_printoptions(linewidth=200)

def extract_fchk(fchk_file):
    # Data from fchk file
    mol = IOData.from_file(fchk_file)
    # Get atomic numbers
    atom_numbers = mol.pseudo_numbers
    # Get coordinates
    coords = mol.coordinates
    # Get basis set
    obasis = mol.obasis
    # Get expansion information
    mol_exp_list = [mol.exp_alpha]
    if hasattr(mol, 'exp_beta'):
        mol_exp_list += [mol.exp_beta]
    return atom_numbers, coords, obasis, mol_exp_list

def fock_horton(pseudo_numbers, coordinates, obasis, mol_exp_list, run_scf=False):
    '''
    Parameters
    ----------
    pseudo_numbers : np.ndarray(K,)
        Atomic numbers of the atoms in system
    coordinates : np.ndarray(K,)
        Coordinates of the atoms in system
    obasis : horton.gbasis.GOBasis
        Gaussian orbital basis set used in HORTON
    mol_exp_list : list of horton.matrix.dense.DenseExpansion
        Molecular orbital expansion used in HORTON
        If there is only one item, then the orbitals are spatial
        If there is two items, then the orbitals are spin (alpha then beta)

    Returns
    ------
    fock_ab_ab:list of numpy.ndarray(K,K)
        Converged Fock matrix from Hartree-Fock calculation
    '''
    # Get integrals from horton
    lf = DenseLinalgFactory(obasis.nbasis)

    # # Integals in atomic basis
    olp = obasis.compute_overlap(lf)
    kin = obasis.compute_kinetic(lf)
    na = obasis.compute_nuclear_attraction(coordinates, pseudo_numbers, lf)
    er = obasis.compute_electron_repulsion(lf)

    dm_list = [exp.to_dm() for exp in mol_exp_list]

    # Construct the restricted HF effective Hamiltonian
    external = {'nn': compute_nucnuc(coordinates, pseudo_numbers)}
    terms = [
        RTwoIndexTerm(kin, 'kin'),
        RDirectTerm(er,'hartree'),
        RExchangeTerm(er,'x_hf'),
        RTwoIndexTerm(na,'ne'),
    ]
    ham = REffHam(terms, external)

    fock_list = []
    # NO SCF
    if run_scf == False:
        for dm in dm_list:
            ham.reset(dm)
            fock = lf.create_two_index()
            ham.compute_fock(fock)
            fock_list.append(fock._array)
    # YES SCF
    else:
        # Create alpha orbital expansion
        exp_alpha = lf.create_expansion()
        exp_alpha._coeffs = np.identity(obasis.nbasis)
        # Decide how to occupy the orbitals (5 alpha electrons)
        occ_model = AufbauOccModel(5)
        # Converge WFN with plain SCF
        scf_solver = PlainSCFSolver()
        scf_solver(ham, lf, olp, occ_model, exp_alpha)
        # retrieve fock matrix
        fock_alpha = lf.create_two_index()
        ham.compute_fock(fock_alpha)
        fock_list.append(fock_alpha._array)
    return fock_list

def fock_integrals(pseudo_numbers, coordinates, obasis, mol_exp_list):
    '''
    Parameters
    ----------
    pseudo_numbers : np.ndarray(K,)
        Atomic numbers of the atoms in system
    coordinates : np.ndarray(K,)
        Coordinates of the atoms in system
    obasis : horton.gbasis.GOBasis
        Gaussian orbital basis set used in HORTON
    mol_exp_list : list of horton.matrix.dense.DenseExpansion
        Molecular orbital expansion used in HORTON
        If there is only one item, then the orbitals are spatial
        If there is two items, then the orbitals are spin (alpha then beta)

    Returns
    ------
    fock_ab_ab:list of numpy.ndarray(K,K)
        Converged Fock matrix from Hartree-Fock calculation
    '''
    # Get basis set
    obasis = obasis

    # Integals in atomic basis
    lf = DenseLinalgFactory(obasis.nbasis)
    olp_ab_ab = obasis.compute_overlap(lf)._array

    kin_ab_ab = obasis.compute_kinetic(lf)._array
    na_ab_ab = obasis.compute_nuclear_attraction(coordinates, pseudo_numbers, lf)._array
    core_ab = kin_ab_ab + na_ab_ab

    er_ab_ab = obasis.compute_electron_repulsion(lf)._array

    fock_list = []
    for exp in mol_exp_list:
        # Data from fchk file
        coeff_ab_mo = exp.coeffs
        occs = exp.occupations
        density_ab_ab = (coeff_ab_mo*occs).dot(coeff_ab_mo.T)

        part_one = np.einsum('ijkl,jl->ik', er_ab_ab, density_ab_ab)
        part_two = np.einsum('ijlk,jl->ik', er_ab_ab, density_ab_ab)

        fock_list.append(core_ab + 2*part_one - part_two)
    return fock_list

def fock_numerical(coeff_ab_mo_list, mo_energies_list):
    """ Calculates the Fock matrix from the mo coefficients and hte mo energies

    ..math::
      \epsilon &= C^T F C\\
      (C^{-1})^T \epsilon C^{-1} &= F

    where :math:`\epsilon` is a diagonal matrix of molecular orbital energies
          :math:`C` is a transformation matrix from atomic basis to molecular orbitals
          :math:`C^{-1}` is pseudoinverse of :math:`C`
          :math:`F` is a Fock matrix in atomic basis

    Assume number of mo's (N) is less than or equal to the number of atomic basis functions (K)

    Parameters
    ----------
    coeff_ab_mo_list : list of np.ndarray(K,N)
        Transformation matrix from the atomic basis set to the molecular orbitals
        If there is only one item, then the orbitals are spatial
        If there is two items, then the orbitals are spin (alpha then beta)
    mo_energies : list of np.ndarray(N,)
    """
    fock_list = []
    for coeff_ab_mo, mo_energies in zip(coeff_ab_mo_list, mo_energies_list):
        inv_coeff_ab_mo = np.linalg.pinv(coeff_ab_mo)
        fock_list.append(inv_coeff_ab_mo.T.dot(np.diag(mo_energies)).dot(inv_coeff_ab_mo))
    return fock_list

def quambo_energy(fchk_file, cao_basis_file, fock):
    '''
    Parameters
    ----------
    fchk_file: str
        File name of the formatted chk file that contains the molecular orbital
        and atomic basis information

    cao_basis_file: str
        File name of the file that contains the chemical basis information
        Files supported: nwchem basis set file, fchk file

    Returns
    ------
    fock_quambo_quambo: list of numpy.ndarray(N,N)
        Orbital energies in QUAMBO basis
    '''
    hd = HortonData(fchk_file, cao_basis_file)
    coeff_ab_mo_sep = hd.coeff_ab_mo_sep
    olp_ab_ab_sep = hd.olp_ab_ab_sep
    olp_cao_ab_sep = hd.olp_cao_ab_sep
    olp_cao_cao_sep = hd.olp_cao_cao_sep
    occupations_sep = hd.occupations_sep
    for (coeff_ab_mo,
         olp_ab_ab,
         olp_cao_ab,
         olp_cao_cao,
         occupations) in zip(coeff_ab_mo_sep,
                             olp_ab_ab_sep,
                             olp_cao_ab_sep,
                             olp_cao_cao_sep,
                             occupations_sep):
        quasi = QuasiTransformation(coeff_ab_mo,
                                    olp_ab_ab,
                                    olp_cao_ab,
                                    olp_cao_cao,
                                    occupations.astype(bool))
        coeff_ab_quambo = quasi.quambo()
        '''fock_quambo_quambo is QUAMBO energy
        '''
        fock_quambo_quambo = coeff_ab_quambo.T.dot(fock).dot(coeff_ab_quambo)
    # occupations = np.array([i for i in occupations_sep[0] if i>0])*2
    # print occupations_sep

    return fock_quambo_quambo

def test_fock():
    pseudo_numbers, coordinates, obasis, mol_exp_list = extract_fchk('ch4_hf.fchk')
    coeff_ab_mo = mol_exp_list[0].coeffs
    energies = mol_exp_list[0].energies

    lf = DenseLinalgFactory(obasis.nbasis)
    olp = obasis.compute_overlap(lf)._array

    test1 = fock_integrals(pseudo_numbers, coordinates, obasis, mol_exp_list)[0]
    # print np.sum(np.abs((test1).dot(coeff_ab_mo) - olp.dot(coeff_ab_mo).dot(np.diag(energies))))
    test1 = coeff_ab_mo.T.dot(test1).dot(coeff_ab_mo)
    print np.diag(test1),'From the integrals'
    # print np.sum(np.abs(test1-np.diag(np.diag(test1))))

    test2 = fock_horton(pseudo_numbers, coordinates, obasis, mol_exp_list, run_scf=False)[0]
    # print np.sum(np.abs((test2).dot(coeff_ab_mo) - olp.dot(coeff_ab_mo).dot(np.diag(energies))))
    test2 = coeff_ab_mo.T.dot(test2).dot(coeff_ab_mo)
    print np.diag(test2),'From HORTON, No SCF'
    # print np.sum(np.abs(test2-np.diag(np.diag(test2))))

    test3 = fock_horton(pseudo_numbers, coordinates, obasis, mol_exp_list, run_scf=True)[0]
    # print np.sum(np.abs((test3).dot(coeff_ab_mo) - olp.dot(coeff_ab_mo).dot(np.diag(energies))))
    test3 = coeff_ab_mo.T.dot(test3).dot(coeff_ab_mo)
    print np.diag(test3),'From HORTON, Yes SCF'
    # print np.sum(np.abs(test3-np.diag(np.diag(test3))))

    test4 = fock_numerical([coeff_ab_mo], [energies])[0]
    # print np.sum(np.abs((test4).dot(coeff_ab_mo) - olp.dot(coeff_ab_mo).dot(np.diag(energies))))
    test4 = coeff_ab_mo.T.dot(test4).dot(coeff_ab_mo)
    print np.diag(test4), 'From artificial Fock'
    # print np.sum(np.abs(test4-np.diag(np.diag(test4))))

    #print energies, 'From fchk'
# diagonalized quambo energy
# [-19.22095484  -5.31946042  -5.05850352  -4.97057566  -5.0586551   -4.04029956  -4.04029956  -4.04012546]
# Non-diagonilized quambo energy
# [[ -1.92209548e+01   3.61699454e-01  -1.01959690e-05   1.00322563e-02  -3.01443110e-16  -1.46601143e+00  -1.46601143e+00  -1.46559176e+00]
#  [  3.61699454e-01  -5.31946042e+00  -7.92584001e-05   6.01043828e-02  -8.20176594e-16  -2.84172140e+00  -2.84172140e+00  -2.84133317e+00]
#  [ -1.01959690e-05  -7.92584001e-05  -5.05850352e+00  -7.39575136e-04   3.17331035e-16  -1.12645989e+00  -1.12645989e+00   2.25242419e+00]
#  [  1.00322563e-02   6.01043828e-02  -7.39575136e-04  -4.97057566e+00   1.70786468e-15   3.14018642e-01   3.14018642e-01   3.31490341e-01]
#  [ -3.24845768e-16  -9.12839360e-16   2.95309854e-16   1.70562168e-15  -5.05865510e+00  -1.95333808e+00   1.95333808e+00   3.91181664e-17]
#  [ -1.46601143e+00  -2.84172140e+00  -1.12645989e+00   3.14018642e-01  -1.95333808e+00  -4.04029956e+00  -1.28466291e+00  -1.28659156e+00]
#  [ -1.46601143e+00  -2.84172140e+00  -1.12645989e+00   3.14018642e-01   1.95333808e+00  -1.28466291e+00  -4.04029956e+00  -1.28659156e+00]
#  [ -1.46559176e+00  -2.84133317e+00   2.25242419e+00   3.31490341e-01   3.72065880e-17  -1.28659156e+00  -1.28659156e+00  -4.04012546e+00]]

