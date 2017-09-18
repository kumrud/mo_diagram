import numpy as np
# from energy import *
from diagram import *
#from horton import * 

energies = np.array([0, 1, 2, 1, 4, 1, 3])

#print 'energies',energies

energy_deltas = np.diff(energies)
print energy_deltas
switch = [i+1 for i in range(len(energy_deltas)) if energy_deltas[i]<0] + [len(energies)]
print switch
dict={}
a=0
atom=1
for i in switch:
    print i, a
    b = energies[a:i]
    dict[atom] = b
    atom += 1
    a = i
print dict



# tol_degen = 0.01
# energy_indices = [i for i,j in enumerate(energy_deltas) if j>tol_degen]
#print "energy_indices",energy_indices

# energy_bins = [energies[0]]+[energies[i+1] for i in energy_indices]+[energies[-1]+1]
#print 'energy_bins',energy_bins

# degen = np.histogram(energies, bins=energy_bins)[0]
# [1 3 1 1]
#print 'degen',type(degen)

# line_width = 1.0
# line_sep = 0.1
#
# x_leftmost = -(degen//2)*(line_width+line_sep) - (degen%2)*line_width/2.0
# [-0.5 -1.6 -0.5 -0.5]
# -1.6 should be -3.8


#print x_leftmost
# x_temp = [+0.55 * (d % 2) - 1.1 * (d - 1)//2 for d in degen]

#print x_temp, 'x_temp'
#print x_leftmost


#x_leftmost = []
#for i in degen:
#    if i== 1:
#        x_leftmost.append(-0.5)
#    if i!=1:
##        print 'it is degen'
#        if i%2 != 0:
##            print 'it is odd'
#            new=-0.5-i//2*1.1
#            x_leftmost.append(new)
#        if i%2 ==0:
##            print 'it is even'
#            new=-0.5-0.55-(i//2-1)*1.1
#            x_leftmost.append(new)
##print 'x_leftmost',x_leftmost




##print 'x_leftmost',x_leftmost
##print 'degen//2',degen//2
##print 'line_width+line_sep',line_width+line_sep 
#
#x_inits = [x_leftmost + i*(line_width+line_sep)]
##print 'x_inits', x_inits
#
#x_finals = [x_leftmost+line_width + i*(line_width+line_sep)]
##print 'x_finals',x_finals


# t=[]
# for i,j in zip(degen,x_temp):
#     if i != 1:
#         t += [j+k*1.1 for k in range(int(i))]
#     if i == 1:
#         t.append(j)
#print t

import matplotlib.pyplot as plt

#fig, ax = plt.subplots()
#occupation = np.array([2.,2.,2.,2.,1.,0.,0.])
## Needs to be a tuple of colors as string
## 2='green', 1='yellow', 0='red'
#fc=[]
#for i in occupation:
#    if i==2:
#        fc.append('green')
#    elif i==1:
#        fc.append('yellow')
#    elif i==0:
#        fc.append('red')
#tuple(fc)
##print fc, 'face_colors'
#
#x_coor = [(i,1) for i in t]
#y_coor = [(i,0.01) for i in energies]
##print x_coor,y_coor,"x_coor, y_coor"
#
#[ax.broken_barh([x_coor[i]], y_coor[i], facecolors=fc[i]) for i,j in enumerate(energies)]
#
#def read_coords(f, line, end_tag):
## reads all the coordinates and strips the unwanted columns
#    line_list = []
#    while end_tag not in line:
#        line = f.readline().strip()
#        line_list.extend(line.split())
#    return line_list


#def manual_energy(fchk_file):
#    with open(fchk_file) as f:
#        line = f.readline()
#        while line:
#            line = f.readline().strip()
#            # read from output the final geometry (as np.ndarray)
#            if 'Alpha Orbital Energies' in line:
#                coeff_mo_mo = read_coords(f, line, 'Alpha MO coefficients')
#                if 'Alpha' in coeff_mo_mo:
#                    get_index = coeff_mo_mo.index('Alpha')
#                    del coeff_mo_mo[get_index:]
#                    return coeff_mo_mo


#plt.show()
#pseudo_numbers, coordinates, obasis, mol_exp_list = extract_fchk('CO.fchk')
#coeff_ab_mo = np.array(mol_exp_list[0].coeffs)
#mo_energies = np.array(mol_exp_list[0].energies)
#moenergies = manual_energy("CO.fchk")
#print type(moenergies)
#print type(mo_energies)
#ao_energies = np.diag(coeff_ab_mo)
# hd = HortonData('CO.fchk', 'aambs.gbs')
#
# coeff_ab_mo_sep = hd.coeff_ab_mo_sep
# coeff_ab_mo = np.hstack(coeff_ab_mo_sep)
#
# mo_energies_sep = hd.energies_sep
# mo_energies = np.hstack(mo_energies_sep)
#
# fock_ab_sep = fock_numerical(coeff_ab_mo_sep, mo_energies_sep)
# ab_energies = np.hstack([np.diag(fock_ab) for fock_ab in fock_ab_sep])
#
# occupations = np.hstack(hd.occupations_sep)
# print ab_energies
# print mo_energies
# print coeff_ab_mo.shape
# mo = MoDiagram(mo_energies, ab_energies, coeff_ab_mo, occupations)
# print mo.ao
# o = OrbitalPlot(mo)
# print '###########################'
# print mo.ao
# print o.ao
# o.line_data(mo.ao)
#if all(mo.ao == ab_energies):
#    print 'yeah'
#else:
#    print 'fuck'



#OrbitalPlot(mo).make_line(mo.ao[0:5])
#plt.show()
