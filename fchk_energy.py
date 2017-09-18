from quasibasis.wrapper_horton import HortonData
from energy import *
import numpy as np
from diagram import MoDiagram, OrbitalPlot, degenerate

hd = HortonData('CO.fchk', 'aambs.gbs')

coeff_ab_mo_sep = hd.coeff_ab_mo_sep
coeff_ab_mo = np.hstack(coeff_ab_mo_sep)

mo_energies_sep = hd.energies_sep
mo_energies = np.hstack(mo_energies_sep)

fock_ab_sep = fock_numerical(coeff_ab_mo_sep, mo_energies_sep)
ab_energies = np.hstack([np.diag(fock_ab) for fock_ab in fock_ab_sep])

occupations = np.hstack(hd.occupations_sep)
basis_map, = hd.cao_basis_map_sep
basis_orb = hd.cao_basis_orbtype_sep
#print basis_orb

print  "~~~~~~~~~~~~~~BASIS MAP~~~~~~~~~~~~~~ "
print basis_map
print "~~~~~~~~~~~~~~MO ENERGIES~~~~~~~~~~~~~~ "
print mo_energies
print "~~~~~~~~~~~~~~AB ENERGIES~~~~~~~~~~~~~~ " 
print ab_energies
print "~~~~~~~~~~~~~~COEFF MATRIX~~~~~~~~~~~~~~ " 
print coeff_ab_mo
m = MoDiagram(mo_energies, ab_energies, coeff_ab_mo, occupations, basis_map)

#def get_aos(energies, basis_map, option=None):
#    return [energies[i] for i,j in enumerate(basis_map) if j in option]
#    #return [i for i,j in enumerate(basis_map) if j in option]
##return [energies[i] for i,j in enumerate(basis_map) for j in option]
#print get_aos(ab_energies, basis_map, option)

o=OrbitalPlot(m)
#C = o.make_line(m.get_aos(m.ao, basis_map, [0]))
C = m.get_aos(m.ao, basis_map, [0])
print 'x_shift',o.x_length
#print m._indices
#print m.coeff[m.indices]
O = m.get_aos(m.ao, basis_map, [1])
CO = m.mo
#print m.indices
#print "~~~~~~~~~~~~~~New COEFF MATRIX~~~~~~~~~~~~~~ " 
#print m.coeff
#O = o.make_line(m.get_aos(m.ao, basis_map, [1]))
#CO = o.make_line(m.mo)
#plt.show()
coeff = m.coeff
o.make_line(C, CO, O)
plt.show()
#print getattr(o, 'connection_lines')
