from numba import cuda
import os, multiprocessing as mp
from src.E8  import E8
from src.E10  import E10
import numpy as np
import time
import gc


"""
print(f"CUDA   : {cuda.is_available()}")
print(f"GPUs   : {len(cuda.gpus)} → {[g.name for g in cuda.gpus]}")
print(f"Current GPU : {cuda.get_current_device().name}")
print(f"CPU threads : {os.cpu_count()} (multiprocessing uses {mp.cpu_count()}) by default)")

dev = cuda.get_current_device()
warp_size = dev.WARP_SIZE  # Hardware-specific (32 on A100 and most NVIDIA GPUs)
threads_per_block = 8 * warp_size  # Computes to 256 automatically; adjust multiplier if needed (e.g., 4 for 128)

# Safety: Clamp to device's max (1024 on A100)
max_threads = dev.MAX_THREADS_PER_BLOCK
threads_per_block = min(threads_per_block, max_threads)

# Print the results
print(f"Hardware WARP Size: {warp_size}")
print(f"Max Threads per Block (Hardware Limit): {max_threads}")
print(f"Calculated Threads per Block: {threads_per_block}")
"""

def run_topology(topology, LAx, LAy, LBx, LBz, LCy, l_max, l_min = 2, x0 = np.array([0.0, 0.0, 0.0]), l_range= np.array([[2,2]]), lp_range= np.array([[2,2]])):
  if topology == 'E8' or 'E10' :
    import parameter_files.default_E8 as parameter_file

    param = parameter_file.parameter
    param['topology'] = topology
    param['LAx'] = LAx
    param['LAy'] = LAy
    param['LBx'] = LBx
    param['LBz'] = LBz
    param['LCy'] = LCy
  #  param['L1y'] = L1y
  #  param['L2x'] = L2x
  #  param['L2z'] = L2z

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


l_max = 60

L_circle = 1

for a in np.array([0.9, 1, 1.1, 1.2, 1.3]):
    C_lmlpmp_untilted = run_topology(topology = 'E10', LAx = a * L_circle, LAy = 0, LBx = 0.7, LBz = 1.0, LCy = 1.4, 
              l_max = l_max, x0= np.array([0.0, 0.0, 0.0], dtype=np.float64), 
              l_range = np.array([[2,l_max]]), lp_range = np.array([[2,l_max]]))
    
    np.save(f'/share/storage5/ricardogonzatto/cov_matrices_spaces/c_lmlpmp_untilted_E10_LAx{a * L_circle}_LAy{0}_LBx{0.7}_LBz{1.0}_LCy{1.4}_x=0_0_0_lmax70', C_lmlpmp_untilted)
    
    # Clear the memory
    del C_lmlpmp_untilted
    gc.collect()

L_circle = 0.73

for a in np.array([0.9, 1, 1.1, 1.2, 1.3]):
    C_lmlpmp_tilted = run_topology(topology = 'E10', LAx = a * L_circle, LAy = 0, LBx = 0.7, LBz = 1.0, LCy = 1.4, 
             l_max = l_max, x0= np.array([-0.1, 0.34, 0.0], dtype=np.float64), 
             l_range = np.array([[2,l_max]]), lp_range = np.array([[2,l_max]]))
    np.save(f'/share/storage5/ricardogonzatto/cov_matrices_spaces/c_lmlpmp_tilted_E10_LAx{a * L_circle}_LAy{0}_LBx{0.7}_LBz{1.0}_LCy{1.4}_x=-0.1_0.34_0.0_lmax70', C_lmlpmp_tilted)

    # Clear the memory
    del C_lmlpmp_tilted
    gc.collect()
