from horton import *
import numpy as np
from wrapper_horton import HortonData
from quasi import QuasiTransformation, project

def horton_energy(fchk_file):
    '''
    Parameters
    ----------
    fchk_file: str
        File name of the formatted chk file that contains the molecular orbital
        and atomic basis information

    Returns
    ------
    fock_ab_ab:list of numpy.ndarray(K,K)
        Converged Fock matrix from Hartree-Fock calculation
    mo_energy: np.ndarray(N)
        Molecular orbital energies
    coeffs_ab_mo: list of np.ndarray(K,N)
        Transformation matrix from atomic basis to molecular orbitals
    occupation: np.ndarray of {bool, int}
        The indices of the molecular orbitals
    '''
    # Get integrals from horton
    np.set_printoptions(linewidth=200)

    # Data from fchk file
    mol = IOData.from_file(fchk_file)

    # Get basis set
    obasis = mol.obasis

    lf = DenseLinalgFactory(obasis.nbasis)

    # Integals in atomic basis
    olp = obasis.compute_overlap(lf)
    kin = obasis.compute_kinetic(lf)
    na = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers, lf)
    er = obasis.compute_electron_repulsion(lf)

    # Create alpha orbital expansion
    exp_alpha = mol.lf.create_expansion()

    # Get coefficient matrix in
    exp_alpha._coeffs = np.identity(obasis.nbasis)
    exp_mo = mol.exp_alpha
    coeffs_ab_mo = exp_mo.coeffs

    # Decide how to occupy the orbitals (5 alpha electrons)
    occ_model = AufbauOccModel(5)

    # Construct the restricted HF effective Hamiltonian
    external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    terms = [
                RTwoIndexTerm(kin, 'kin'),
                RDirectTerm(er,'hartree'),
                RExchangeTerm(er,'x_hf'),
                RTwoIndexTerm(na,'ne'),
                            ]
    ham = REffHam(terms, external)

    # Converge WFN with plain SCF
    scf_solver = PlainSCFSolver()
    scf_solver(ham, lf, olp, occ_model, exp_alpha)
    dm_alpha = ham.cache['dm_alpha']

    # Derive orbitals from the Fock and density matrices
    fock_alpha = lf.create_two_index()
    ham.reset(dm_alpha)
    ham.compute_energy()
    ham.compute_fock(fock_alpha)

    #Fock Matrix (in atomic basis)
    fock_ab_ab = fock_alpha._array
    mo_energy = exp_mo.energies
    occupation = exp_alpha.occupations
    print fock_ab_ab
    return mo_energy, fock_ab_ab, coeffs_ab_mo, occupation

def horton_energy2(fchk_file):
    '''
    Parameters
    ----------
    fchk_file: str
        File name of the formatted chk file that contains the molecular orbital
        and atomic basis information

    Returns
    ------
    fock_ab_ab:list of numpy.ndarray(K,K)
        Converged Fock matrix from Hartree-Fock calculation
    mo_energy: np.ndarray(N)
        Molecular orbital energies
    coeffs_ab_mo: list of np.ndarray(K,N)
        Transformation matrix from atomic basis to molecular orbitals
    occupation: np.ndarray of {bool, int}
        The indices of the molecular orbitals
    '''
    # Get integrals from horton
    np.set_printoptions(linewidth=200)

    # Data from fchk file
    mol = IOData.from_file(fchk_file)
    coeff_ab_mo = mol.exp_alpha.coeffs
    occs = mol.exp_alpha.occupations
    density_ab_ab = (coeff_ab_mo*occs).dot(coeff_ab_mo.T)

    # Get basis set
    obasis = mol.obasis

    # Integals in atomic basis
    lf = DenseLinalgFactory(obasis.nbasis)
    olp_ab_ab = obasis.compute_overlap(lf)._array

    kin_ab_ab = obasis.compute_kinetic(lf)._array
    na_ab_ab = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers, lf)._array
    core_ab = kin_ab_ab + na_ab_ab

    er_ab_ab = obasis.compute_electron_repulsion(lf)._array
    part_one = np.einsum('ijkl,jl->ik', er_ab_ab, density_ab_ab)
    part_two = np.einsum('ijlk,jl->ik', er_ab_ab, density_ab_ab)

    fock_ab = core_ab + 2*part_one - part_two
    return fock_ab

    # coulomb = np.einsum('ijkl->ik', er_ab_ab)
    # exchange = np.einsum('ijlk->ik', er_ab_ab)
    # fock = core_ab + 2*coulomb - exchange
    # return coeff_ab_mo.T.dot(fock).dot(coeff_ab_mo)


def fchk_energy(fchk_file):
    '''
    Parameters
    ----------
    fchk_file: str
        File name of the formatted chk file that contains the molecular orbital
        and atomic basis information

    Returns
    ------
    coeff_mo_mo:numpy.ndarray(N)
        Molecular orbital energies from formated check point file
    '''
    def read_coords(f, line, end_tag):
    # reads all the coordinates and strips the unwanted columns
        line_list = []
        while end_tag not in line:
            line = f.readline().strip()
            line_list.extend(line.split())
        return line_list

    with open(fchk_file) as f:
        line = f.readline()
        while line:
            line = f.readline().strip()
            # read from output the final geometry (as np.ndarray)
            if 'Alpha Orbital Energies' in line:
                coeff_mo_mo = read_coords(f, line, 'Alpha MO coefficients')
                if 'Alpha' in coeff_mo_mo:
                    get_index = coeff_mo_mo.index('Alpha')
                    del coeff_mo_mo[get_index:]
                    return coeff_mo_mo

def quambo_energy(fchk_file, cao_basis_file):
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
        fock_quambo_quambo = coeff_ab_quambo.T.dot(horton_energy(fchk_file)[1]).dot(coeff_ab_quambo)
    # occupations = np.array([i for i in occupations_sep[0] if i>0])*2
    # print occupations_sep

    return fock_quambo_quambo,quasi.indices_span

if __name__ == '__main__':
    print "HORTON ENERGY"
    print horton_energy('ch3_rohf_sto3g_g03.fchk')[0]
    print horton_energy('ch3_rohf_sto3g_g03.fchk')[3]
    print '+'*50
    print "GAUSSIAN ENERGY"
    print '+'*50
    print fchk_energy('ch3_rohf_sto3g_g03.fchk')
    print '+'*50
    print "QUAMBO ENERGY"
    print quambo_energy('ch3_rohf_sto3g_g03.fchk', 'aambs.gbs')[1]
    print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    mol = IOData.from_file('ch3_rohf_sto3g_g03.fchk')
    coeff_ab_mo = mol.exp_alpha.coeffs
    test1 = horton_energy2('ch3_rohf_sto3g_g03.fchk')
    test1 = coeff_ab_mo.T.dot(test1).dot(coeff_ab_mo)
    print np.diag(test1),'x'*10
    print np.sum(np.abs(test1-np.diag(np.diag(test1))))
    # print fchk_energy('ch3_rohf_sto3g_g03.fchk')
    test2 = horton_energy('ch3_rohf_sto3g_g03.fchk')[1]
    test2 = coeff_ab_mo.T.dot(test2).dot(coeff_ab_mo)
    print np.diag(test2),'y'*10
    print np.sum(np.abs(test2-np.diag(np.diag(test2))))
    #print type(fchk_energy('ch3_rohf_sto3g_g03.fchk'))
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

