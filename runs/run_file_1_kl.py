import os
import multiprocessing
import numpy as np
import gc

# ============================================================================
# SETTINGS
# ============================================================================

# Number of processes. The RAM of the covariance step grows with this number
# (each process holds a full copy of the matrix, 1.33 GB at l_max = 60):
#   l_max = 60:  1 -> ~7 GB,  2 -> ~9 GB,  4 -> ~15 GB,  8 -> ~25 GB,  16 -> ~41 GB
# Use at most the number of cores you asked the cluster for.
n_processes = 4
multiprocessing.cpu_count = lambda: n_processes   # must come before importing src

l_max = 60

# Folder where the final matrices are saved (created if it does not exist)
save_dir = '/share/storage5/ricardogonzatto/cov_matrices_spaces/'

# ============================================================================

from src.E8_kl import E8      # E8 using topology_kl.py / tools_kl.py (float32 integrand and wigner_d)


def run_topology(topology, LAx, LAy, LBx, LBz, LCy, l_max, l_min = 2, x0 = np.array([0.0, 0.0, 0.0]), l_range= np.array([[2,2]]), lp_range= np.array([[2,2]])):
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

  a = E8(param=param, make_run_folder=True)

  c_ell = a.calculate_c_lmlpmp(
    plot_param={
      'l_ranges':l_range,
      'lp_ranges': lp_range,
    }
  )

  return c_ell


if __name__ == '__main__':
  os.makedirs(save_dir, exist_ok=True)

  L_circle = 1

  for a in np.array([0.9, 1, 1.1, 1.2, 1.3]):
      C_lmlpmp_untilted = run_topology(topology = 'E8', LAx = a * L_circle, LAy = 0, LBx = 0.7, LBz = 1.0, LCy = 1.4,
                l_max = l_max, x0= np.array([0.0, 0.0, 0.0], dtype=np.float64),
                l_range = np.array([[2,l_max]]), lp_range = np.array([[2,l_max]]))

      np.save(os.path.join(save_dir, f'c_lmlpmp_untilted_E8_LAx{a * L_circle}_LAy{0}_LBx{0.7}_LBz{1.0}_LCy{1.4}_x=0_0_0_lmax{l_max}'), C_lmlpmp_untilted)

      # Clear the memory
      del C_lmlpmp_untilted
      gc.collect()

  L_circle = 0.73

  for a in np.array([0.9, 1, 1.1, 1.2, 1.3]):
      C_lmlpmp_tilted = run_topology(topology = 'E8', LAx = a * L_circle, LAy = 0, LBx = 0.7, LBz = 1.0, LCy = 1.4,
               l_max = l_max, x0= np.array([-0.1, 0.34, 0.0], dtype=np.float64),
               l_range = np.array([[2,l_max]]), lp_range = np.array([[2,l_max]]))

      np.save(os.path.join(save_dir, f'c_lmlpmp_tilted_E8_LAx{a * L_circle}_LAy{0}_LBx{0.7}_LBz{1.0}_LCy{1.4}_x=-0.1_0.34_0.0_lmax{l_max}'), C_lmlpmp_tilted)

      # Clear the memory
      del C_lmlpmp_tilted
      gc.collect()
