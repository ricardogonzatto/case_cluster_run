from numba import cuda
import os, multiprocessing as mp
from src.E8  import E8
from src.E10  import E10
import numpy as np
import time


def run_topology(topology, LAx, LAy, LBx, LBz, LCy, l_max = 10, l_min = 2, x0 = np.array([0.0, 0.0, 0.0]), l_range= np.array([[2,2]]), lp_range= np.array([[2,2]])):
  
  if topology == 'E8' or 'E10' :
    import parameter_files.default_E8 as parameter_file

    param = parameter_file.parameter
    param['topology'] = topology
    param['LAx'] = LAx
    param['LAy'] = LAy
    param['LBx'] = LBx
    param['LBz'] = LBz
    param['LCy'] = LCy

  param['l_max'] = l_max
  param['l_min'] = l_min
  param['node'] = 1
  param['x0'] = x0

  if param['topology'] == 'E8':
    a = E8(param=param, make_run_folder=True)
  elif param['topology'] == 'E10':
    a = E10(param=param, make_run_folder=True)
  else:
    exit()

  c_ell = a.calculate_c_lmlpmp(
    plot_param={
      'l_ranges':l_range,
      'lp_ranges': lp_range,
    }
  )
 
  return c_ell


#"/share/storage5/ricardogonzatto/cov_matrices_spaces/"
l_max = 40

L_circle = 1
LAy = 0 
LBx = 0.7
LBz = 1
LCy = 1.4

for a in np.array([0.9, 1, 1.1, 1.2, 1.3]):
    C_lmlpmp_untilted = run_topology(topology = 'E8', LAx = a * L_circle, LAy = LAy, LBx = LBx , LBz = LBz, LCy = LCy, 
              l_max = l_max, x0= np.array([0.0, 0.0, 0.0], dtype=np.float64), 
              l_range = np.array([[2,l_max]]), lp_range = np.array([[2,l_max]]))
    
    np.save(f'CROSS_c_lmlpmp_untilted_E8_LA{a * L_circle}_LAy{LAy}_LBx{LBx}_LBz{LBz}_LCy{LCy}_x=0_0_0_lmax{l_max}', C_lmlpmp_untilted)

L_circle = 0.73
LAy = 0 
LBx = 0.7
LBz = 1
LCy = 1.4

for a in np.array([0.9, 1, 1.1, 1.2, 1.3]):
    C_lmlpmp_tilted = run_topology(topology = 'E8', LAx = a * L_circle, LAy = LAy, LBx = LBx, LBz = LBz, LCy = LCy, 
             l_max = l_max, x0= np.array([-0.1, 0.34, 0.0], dtype=np.float64), 
             l_range = np.array([[2,l_max]]), lp_range = np.array([[2,l_max]]))
    np.save(f'CROSS_c_lmlpmp_tilted_E8_LA{a * L_circle}_LAy{LAy}_LBx{LBx}_LBz{LBz}_LCy{LCy}_x=-0.1_0.34_0.0_lmax{l_max}', C_lmlpmp_tilted)

L_circle = 1
LAy = 0 
LBx = 0.7
LBz = 1
LCy = 1.4

for a in np.array([0.9, 1, 1.1, 1.2, 1.3]):
    C_lmlpmp_untilted = run_topology(topology = 'E10', LAx = a * L_circle, LAy = LAy, LBx = LBx, LBz = LBz, LCy = LCy, 
              l_max = l_max, x0= np.array([0.0, 0.0, 0.0], dtype=np.float64), 
              l_range = np.array([[2,l_max]]), lp_range = np.array([[2,l_max]]))
    
    np.save(f'CROSS_c_lmlpmp_untilted_E10_LA{a * L_circle}_LAy{LAy}_LBx{LBx}_LBz{LBz}_LCy{LCy}_x=0_0_0_lmax{l_max}', C_lmlpmp_untilted)

L_circle = 0.73
LAy = 0 
LBx = 0.7
LBz = 1
LCy = 1.4

for a in np.array([0.9, 1, 1.1, 1.2, 1.3]):
    C_lmlpmp_tilted = run_topology(topology = 'E10', LAx = a * L_circle, LAy = LAy, LBx = LBx, LBz = LBz, LCy = LCy, 
             l_max = l_max, x0= np.array([-0.1, 0.34, 0.0], dtype=np.float64), 
             l_range = np.array([[2,l_max]]), lp_range = np.array([[2,l_max]]))
    np.save(f'CROSS_c_lmlpmp_tilted_E10_LA{a * L_circle}_LAy{LAy}_LBx{LBx}_LBz{LBz}_LCy{LCy}_x=-0.1_0.34_0.0_lmax{l_max}', C_lmlpmp_tilted)

