from numba import cuda
import os, multiprocessing as mp
from src.E7  import E7
from src.E9  import E9
import numpy as np
import time

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

def run_topology(topology, LAx, LAy, L1y = 0.1, L2x = 0.1, L2z = 0.1, l_max = 10, l_min = 2, x0 = np.array([0.0, 0.0, 0.0]), l_range= np.array([[2,2]]), lp_range= np.array([[2,2]])):
  
  if topology == 'E7' or 'E9' :
    import parameter_files.default_E7 as parameter_file

    param = parameter_file.parameter
    param['topology'] = topology
    param['LAx'] = LAx
    param['LAy'] = LAy
    param['L1y'] = L1y
    param['L2x'] = L2x
    param['L2z'] = L2z

  param['l_max'] = l_max
  param['l_min'] = l_min
  param['node'] = 1
  param['x0'] = x0

  if param['topology'] == 'E7':
    a = E7(param=param, make_run_folder=True)
  if param['topology'] == 'E9':
    a = E9(param=param, make_run_folder=True)
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
L_Ay = 0 
L_1y = 1.2
L_2x = 0.7
L_2z = 1 

for a in np.array([0.9, 1, 1.1, 1.2, 1.3]):
    C_lmlpmp_untilted = run_topology(topology = 'E7', LAx = a * L_circle, LAy = L_Ay, L1y = L_1y, L2x = L_2x, L2z = L_2z, 
              l_max = l_max, x0= np.array([0.0, 0.0, 0.0], dtype=np.float64), 
              l_range = np.array([[2,l_max]]), lp_range = np.array([[2,l_max]]))
    
    np.save(f'c_lmlpmp_untilted_E7_LA{a * L_circle}_LAy{L_Ay}_Ly{L_1y}_L2x{L_2x}_L2z{L_2z}_x=0_0_0_lmax60', C_lmlpmp_untilted)

L_circle = 0.73
L_Ay = 0 
L_1y = 1.2
L_2x = 0.7
L_2z = 1 

for a in np.array([0.9, 1, 1.1, 1.2, 1.3]):
    C_lmlpmp_tilted = run_topology(topology = 'E7', LAx = a * L_circle, LAy = L_Ay, L1y = L_1y, L2x = L_2x, L2z = L_2z, 
             l_max = l_max, x0= np.array([-0.1, 0.34, 0.0], dtype=np.float64), 
             l_range = np.array([[2,l_max]]), lp_range = np.array([[2,l_max]]))
    np.save(f'c_lmlpmp_tilted_E7_LA{a * L_circle}_LAy{L_Ay}_Ly{L_1y}_L2x{L_2x}_L2z{L_2z}_x=-0.1_0.34_0.0_lmax60', C_lmlpmp_tilted)

L_circle = 1
L_1y = 1.2
L_Ay = L_1y / 2
L_2x = 0.7
L_2z = 1 

for a in np.array([0.9, 1, 1.1, 1.2, 1.3]):
    C_lmlpmp_untilted = run_topology(topology = 'E9', LAx = a * L_circle, LAy = L_Ay, L1y = L_1y, L2x = L_2x, L2z = L_2z, 
              l_max = l_max, x0= np.array([0.0, 0.0, 0.0], dtype=np.float64), 
              l_range = np.array([[2,l_max]]), lp_range = np.array([[2,l_max]]))
    
    np.save(f'c_lmlpmp_untilted_E9_LA{a * L_circle}_LAy{L_Ay}_Ly{L_1y}_L2x{L_2x}_L2z{L_2z}_x=0_0_0_lmax60', C_lmlpmp_untilted)

L_circle = 0.73
L_1y = 1.2
L_Ay = L_1y / 2
L_2x = 0.7
L_2z = 1 

for a in np.array([0.9, 1, 1.1, 1.2, 1.3]):
    C_lmlpmp_tilted = run_topology(topology = 'E9', LAx = a * L_circle, LAy = L_Ay, L1y = L_1y, L2x = L_2x, L2z = L_2z, 
             l_max = l_max, x0= np.array([-0.1, 0.34, 0.0], dtype=np.float64), 
             l_range = np.array([[2,l_max]]), lp_range = np.array([[2,l_max]]))
    np.save(f'c_lmlpmp_tilted_E9_LA{a * L_circle}_LAy{L_Ay}_Ly{L_1y}_L2x{L_2x}_L2z{L_2z}_x=-0.1_0.34_0.0_lmax60', C_lmlpmp_tilted)

