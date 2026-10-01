from .tools_kl import *
from .topology_kl import Topology
import numpy as np
import quaternionic
import spherical
from numpy import pi, sin, tan, sqrt, conjugate
from numba import njit, prange, jit
import multiprocessing
from multiprocessing import Manager, shared_memory
from concurrent.futures import ProcessPoolExecutor

class E1(Topology):
  def __init__(self, param, debug=True, make_run_folder = False):
    self.L_LSS = 13824.9 * 2
    self.Lx = param['Lx'] * self.L_LSS
    self.Ly = param['Ly'] * self.L_LSS
    self.Lz = param['Lz'] * self.L_LSS
    self.alpha = param['alpha'] * np.pi / 180
    self.beta = param['beta'] * np.pi / 180
    self.V = self.Lx * self.Ly * self.Lz * sin(self.beta) * sin(self.alpha)
    self.l_max = param['l_max']
    self.l_min = param['l_min']

    self.param = param

    if param['Lx'] == param['Ly'] and param['Ly'] == param['Lz'] and param['beta'] == 90 and param['alpha'] == 90:
      print('Cubic Torus')
      self.cubic = True
    else:
      self.cubic = False

    print('Running - E1 l_max={}, Lx={}'.format(self.l_max, int(self.Lx)))
    self.root = 'runs/{}_Lx_{}_Ly_{}_Lz_{}_beta_{}_alpha_{}_l_max_{}_accuracy_{}_percent/'.format(
        param['topology'],
        "{:.2f}".format(param['Lx']),
        "{:.2f}".format(param['Ly']),
        "{:.2f}".format(param['Lz']),
        int(param['beta']),
        int(param['alpha']),
        self.l_max,
        int(param['c_l_accuracy']*100)
    )

    Topology.__init__(self, param, debug, make_run_folder)

  def get_c_lmlpmp_per_process_multi(
    self,
    process_i,
    return_dict,
    min_ell,
    max_ell,
    V,
    k_amp, 
    phi, 
    theta_unique_index,
    k_amp_unique_index,
    k_max_list,
    l_max,
    l_min,
    lm_2_index,
    wigner_d_l_m_2, 
    integrand,
    ell_range,
    ell_p_range
  ):
    # This function seems unnecessary, but Numba does not allow return_dict
    # which is of type multiprocessing.Manager
    return_dict[process_i] = get_full_c_lmlpmp(
      min_ell,
      max_ell,
      V,
      k_amp, 
      phi, 
      theta_unique_index,
      k_amp_unique_index,
      k_max_list,
      l_max,
      l_min,
      lm_2_index,
      wigner_d_l_m_2, 
      integrand,
      ell_range,
      ell_p_range
    )
  
  


  def get_list_of_k_phi_theta(self):
    # k_amp, phi, theta = get_list_of_k_phi_theta(max(self.k_max_list), self.k_list[0], self.Lx, self.Ly, self.Lz, self.beta, self.alpha)
    if (self.Lx*self.Ly*self.Lz/self.L_LSS**3) > 0.124:
      k_amp, phi, theta = get_list_of_k_phi_theta(max(self.k_max_list), self.k_list[0], self.Lx, self.Ly, self.Lz, self.beta, self.alpha)
      print(f"max k_list={max(self.k_max_list)} and the min_k_list={self.k_list[0]}")
    else:
      k_amp, phi, theta = get_list_of_k_phi_theta(self.k_list[-1000], self.k_list[0], self.Lx, self.Ly, self.Lz, self.beta, self.alpha)
      print(f"max k_list={self.k_list[-1000]} and the min_k_list={self.k_list[0]}")

    return k_amp, phi, theta
  
  



@njit(parallel = False)
def get_list_of_k_phi_theta(k_max, k_min, L_x, L_y, L_z, beta, alpha):
    # Returns list of k, phi, and theta for this topology

    tan_b_inv = 1/tan(beta)
    sin_b_inv = 1/sin(beta)
    tan_a_inv = 1/tan(alpha)
    sin_a_inv = 1/sin(alpha)
    print('inv sin and tan', sin_b_inv, tan_b_inv)

    n_x_max = int(np.ceil(k_max * L_x / (2*pi)))
    n_y_max = int(np.ceil(k_max * L_y / (2*pi)))
    n_z_max = int(np.ceil(k_max * L_z / (2*pi)))
 
    k_amp = np.zeros(n_x_max * n_y_max * n_z_max * 8, dtype= np.float64) # *2(n_x> or n_x<0) *2(n_y> or n_y<0)*2(n_z> or n_z<0)
    phi = np.zeros(n_x_max * n_y_max * n_z_max * 8, dtype= np.float64)
    theta = np.zeros(n_x_max * n_y_max * n_z_max * 8, dtype= np.float64)
    


    cur_index = 0

    for n_x in prange(-n_x_max, n_x_max + 1):
      k_x = 2*pi * n_x / L_x

      for n_z in range(-n_z_max, n_z_max+1):        
        k_z = 2*pi * n_z * sin_b_inv / L_z - k_x * tan_b_inv #gamma= 2 *Pi
        k_xz_squared = k_x*k_x + k_z*k_z

        if k_xz_squared > k_max*k_max:
          continue

        for n_y in range(-n_y_max, n_y_max+1):
          k_y = 2*pi * n_y * sin_a_inv / L_y - k_x * tan_a_inv

          k_xyz = sqrt(k_xz_squared + k_y*k_y)
          if k_xyz > k_max or k_xyz < k_min:
            continue

          k_amp[cur_index] = k_xyz
          cur_phi, cur_theta = cart2spherical(np.array([k_x, k_y, k_z])/k_xyz)
          phi[cur_index] = cur_phi
          theta[cur_index] = cur_theta
 
          cur_index += 1
    
    k_amp = k_amp[:cur_index]
    phi = phi[:cur_index]   
    theta = theta[:cur_index]
    
    print('Final num of elements:', k_amp.size, 'Minimum k_amp', np.amin(k_amp), 'n_x_max', n_x_max, 'n_z_max', n_z_max)
    return k_amp, phi, theta







@njit(nogil=True, parallel = False)
def get_c_lmlpmp_diag(
    min_index,
    max_index,
    V,
    k_amp, 
    phi, 
    theta_unique_index,
    k_amp_unique_index,
    k_max_list, 
    l_max,
    l_min,
    lm_index,
    wigner_d_l_m_2,
    integrand,
    progress
    ):

    #ell_range[1] = max_l  ell_range[0] = min_l
    total_num_l_m = (l_max - l_min + 1) * (l_max + l_min + 2) // 2
    num_l = l_max + 1 - l_min
    C_lmlpmp = np.zeros((6, num_l), dtype=np.complex128)    

    min_k_amp = np.min(k_amp)
    for i in range(min_index, max_index):
      k_amp_cur = k_amp[i]
      k_amp_cur = k_amp[i]
      k_unique_index_cur = k_amp_unique_index[i]
      wigner_d_index = theta_unique_index[i] # wigner_d for especific theta[i]

      k_unique_index_cur = k_amp_unique_index[i]

      m_list = np.arange(0, l_max+1)
      phase_list_minus = np.exp(-1j * phi[i] * m_list)
      phase_list_plus = np.exp(1j * phi[i] * m_list)

      for l in range(l_min, l_max + 1):
        if k_amp_cur > k_max_list[l] or k_amp_cur < min_k_amp:
          continue
        l_index_cur = l - l_min # l - 2
        coeff_EE_BB_ell = pi / 2
        coeff_TT_ell = pi * (l + 2) * (l + 1) * l * (l - 1) / 2 
        coeff_TE_ell = pi * sqrt((l + 2) * (l + 1) * l * (l - 1) / 4)
        for m in range(-l, l + 1):
          abs_m = np.abs(m)
          wigner_cur_index = lm_index[l, abs_m]

          if m<0:
            wigner_D_l_m_plus2 = (-1)**abs_m * wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_cur_index] * phase_list_plus[abs_m] 
            wigner_D_l_m_minus2 = (-1)**abs_m * wigner_d_l_m_2[wigner_d_index, wigner_cur_index] * phase_list_plus[abs_m] 
          else:
            wigner_D_l_m_plus2 = wigner_d_l_m_2[wigner_d_index, wigner_cur_index] * phase_list_minus[abs_m] 
            wigner_D_l_m_minus2 =  wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_cur_index] * phase_list_minus[abs_m]

          Xi_plus = wigner_D_l_m_plus2* np.conjugate(wigner_D_l_m_plus2) + wigner_D_l_m_minus2 *np.conjugate(wigner_D_l_m_minus2)
          Xi_minus = wigner_D_l_m_minus2 *np.conjugate(wigner_D_l_m_minus2) - wigner_D_l_m_plus2* np.conjugate(wigner_D_l_m_plus2)

          C_lmlpmp[0, l_index_cur] += coeff_TT_ell* integrand[0, k_unique_index_cur, l_index_cur] * Xi_plus
          C_lmlpmp[1, l_index_cur] += coeff_EE_BB_ell * integrand[1, k_unique_index_cur, l_index_cur] * Xi_plus
          C_lmlpmp[2, l_index_cur] += coeff_EE_BB_ell* integrand[2, k_unique_index_cur, l_index_cur] * Xi_plus
          C_lmlpmp[3, l_index_cur] += coeff_TE_ell * integrand[3, k_unique_index_cur, l_index_cur] * Xi_plus
          C_lmlpmp[4, l_index_cur] += coeff_EE_BB_ell * integrand[4, k_unique_index_cur, l_index_cur] * Xi_minus
          C_lmlpmp[5, l_index_cur] += coeff_TE_ell * integrand[5, k_unique_index_cur, l_index_cur] * Xi_minus     
            
      if progress != None: progress.update(1)

    C_lmlpmp *= pi*pi / (2 * V)
    
    return C_lmlpmp





@njit(nogil=True, parallel = False)
def get_full_c_lmlpmp(
    min_ell,
    max_ell,
    V,
    k_amp, 
    phi, 
    theta_unique_index,
    k_amp_unique_index,
    k_max_list, 
    l_max,
    l_min,
    lm_index,
    wigner_d_l_m_2,
    integrand,
    ell_range,
    ell_p_range,
    ):

    total_num_l_m = (l_max - l_min + 1)*(l_max + l_min + 2) // 2 # the total number of wigner_d matrices for \lambda =+/-2 
    num_l_m = ell_range[1] * (ell_range[1] + 1) + ell_range[1] + 1 - ell_range[0] * ell_range[0]      # = Sum[2 l+1, {l , l_min, l_max}]
    C_lmlpmp = np.zeros((6, num_l_m, num_l_m), dtype=np.complex128) 

    eig_num = k_amp.size
    ipow = np.array([1, 1j, -1, -1j])
    m_list = np.arange(0, l_max+1)
    shortle = np.array([1, -1])
    for i in prange(eig_num):
        k_amp_cur = k_amp[i]
        k_unique_index_cur = k_amp_unique_index[i]
        wigner_d_index = theta_unique_index[i] # wigner_d for especific theta[i]
        phase_list_minus = np.exp(-1j * phi[i] * m_list)
        phase_list_plus  = np.exp(1j * phi[i] * m_list)
        
        for l in range(min_ell, max_ell + 1):

          coeff_E_B_ell = sqrt(pi* (2 * l + 1) / 2)
          coeff_T_ell = sqrt(pi* (2 * l + 1) * (l + 2) * (l + 1) * l * (l - 1) / 2 ) 

          if ell_p_range[0] > l:
            l_start = ell_p_range[0]
          else:
            l_start = l
          for l_p in range(l_start, ell_p_range[1]+1):
            if k_amp_cur > np.sqrt(k_max_list[l]*k_max_list[l_p]):
              continue

            coeff_E_B_lp = sqrt(pi* (2 * l_p + 1) / 2)
            coeff_T_lp = sqrt(pi* (2 * l_p + 1) * (l_p + 2) * (l_p + 1) * l_p * (l_p - 1) / 2 ) 

            ell_ell_p_integ_TT = (coeff_T_ell * coeff_T_lp * 
                                integrand[0, k_unique_index_cur, l, l_p] 
                                * ipow[(l-l_p)%4])
            ell_ell_p_integ_EE = (coeff_E_B_ell * coeff_E_B_lp * 
                                  integrand[1, k_unique_index_cur, l, l_p] 
                                  * ipow[(l-l_p)%4])
            ell_ell_p_integ_BB = (coeff_E_B_ell * coeff_E_B_lp * 
                                  integrand[2, k_unique_index_cur, l, l_p] 
                                  * ipow[(l-l_p)%4])
            ell_ell_p_integ_TE = (coeff_T_ell * coeff_E_B_lp * 
                                  integrand[3, k_unique_index_cur, l, l_p] 
                                  * ipow[(l-l_p)%4])
            ell_ell_p_integ_EB = (coeff_E_B_ell * coeff_E_B_lp * 
                                  integrand[4, k_unique_index_cur, l, l_p] 
                                  * ipow[(l-l_p)%4])
            ell_ell_p_integ_TB = (coeff_T_ell * coeff_E_B_lp * 
                                  integrand[5, k_unique_index_cur, l, l_p] 
                                  * ipow[(l-l_p)%4])
            
            #cross-correlations
            ell_p_ell_integ_TE = (coeff_T_lp * coeff_E_B_ell * 
                                  integrand[3, k_unique_index_cur, l_p, l] 
                                  * ipow[(l_p-l)%4] )
            ell_p_ell_integ_EB = (coeff_E_B_lp * coeff_E_B_ell * 
                                  integrand[4, k_unique_index_cur, l_p, l] 
                                  * ipow[(l_p-l)%4])
            ell_p_ell_integ_TB = (coeff_T_lp * coeff_E_B_ell * 
                                  integrand[5, k_unique_index_cur, l_p, l] 
                                  * ipow[(l_p-l)%4])

            # for non-zero m and m'
            for m in range(-l, l + 1):
              lm_index_cur = l * (l+1) + m - ell_range[0] * ell_range[0] # l**2 + l +m - l_min**2
              abs_m = np.abs(m)
              wigner_cur_index = lm_index[l, abs_m]

              if m<0:
                wigner_D_l_m_plus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_cur_index] * phase_list_plus[abs_m]
                wigner_D_l_m_minus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_index, wigner_cur_index] * phase_list_plus[abs_m]
              else:
                wigner_D_l_m_plus2 = wigner_d_l_m_2[wigner_d_index, wigner_cur_index] * phase_list_minus[abs_m] 
                wigner_D_l_m_minus2 =  wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_cur_index] * phase_list_minus[abs_m]

              
              # Only do m-mp = 0 mod 2  
              for m_p in range(0, l_p + 1):
                if (m_p-m)%2 ==1:
                  continue
                lm_p_index_cur = l_p * (l_p+1) + m_p  - ell_p_range[0] * ell_p_range[0]
                wigner_p_cur_index = lm_index[l_p, m_p]

                wigner_D_l_m_p_plus2 = wigner_d_l_m_2[wigner_d_index, wigner_p_cur_index] * phase_list_minus[m_p] 
                wigner_D_l_m_p_minus2 =  wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_p_cur_index] * phase_list_minus[m_p] 

                Xi_plus = wigner_D_l_m_plus2* np.conjugate(wigner_D_l_m_p_plus2) + wigner_D_l_m_minus2 *np.conjugate(wigner_D_l_m_p_minus2)
                Xi_minus = wigner_D_l_m_minus2 *np.conjugate(wigner_D_l_m_p_minus2) - wigner_D_l_m_plus2* np.conjugate(wigner_D_l_m_p_plus2)
              
                
                # TT correlations
                C_lmlpmp[0, lm_index_cur, lm_p_index_cur] += ell_ell_p_integ_TT * Xi_plus
                # EE correlations
                C_lmlpmp[1, lm_index_cur, lm_p_index_cur] += ell_ell_p_integ_EE * Xi_plus
                # BB correlations
                C_lmlpmp[2, lm_index_cur, lm_p_index_cur] += ell_ell_p_integ_BB * Xi_plus
                # TE correlations
                C_lmlpmp[3, lm_index_cur, lm_p_index_cur] += ell_ell_p_integ_TE * Xi_plus
                # EB correlations
                C_lmlpmp[4, lm_index_cur, lm_p_index_cur] += ell_ell_p_integ_EB * Xi_minus
                # TB correlations
                C_lmlpmp[5, lm_index_cur, lm_p_index_cur] += ell_ell_p_integ_TB * Xi_minus
                if l != l_p:
                  # TE correlations
                  C_lmlpmp[3, lm_p_index_cur, lm_index_cur] += ell_p_ell_integ_TE * conjugate(Xi_plus)
                  # EB correlations
                  C_lmlpmp[4, lm_p_index_cur, lm_index_cur] += ell_p_ell_integ_EB * conjugate(Xi_minus)
                  # TB correlations
                  C_lmlpmp[5, lm_p_index_cur, lm_index_cur] += ell_p_ell_integ_TB * conjugate(Xi_minus)


    for l in prange(min_ell, max_ell + 1):
      for l_p in range(l, ell_p_range[1]+1):
        for m in range(-l, l + 1):
          lm_index_new = l * (l+1) + m - ell_range[0]*ell_range[0] # l**2 + l +m - l_min**2
          lm_index_cal = l * (l+1) - m - ell_range[0]*ell_range[0]
          for m_p in range(-l_p, 0):
            lm_p_index_new = l_p * (l_p+1) + m_p  - ell_range[0]*ell_range[0]
            lm_p_index_cal = l_p * (l_p+1) - m_p  - ell_range[0]*ell_range[0]
            C_lmlpmp[:, lm_index_new, lm_p_index_new] = shortle[(m+m_p)%2] * conjugate(C_lmlpmp[:, lm_index_cal, lm_p_index_cal])
            if l != l_p:
              C_lmlpmp[3:, lm_p_index_new, lm_index_new] = shortle[(m+m_p)%2] * conjugate(C_lmlpmp[3:, lm_p_index_cal, lm_index_cal])

    C_lmlpmp *= pi*pi / (2 * V)

    return C_lmlpmp



# @njit(nogil=True, parallel = False)
def get_full_c_lmlpmp_vectorize(
    min_ell,
    max_ell,
    V,
    k_amp, 
    phi, 
    theta_unique_index,
    k_amp_unique_index,
    k_max_list, 
    l_max,
    lm_index,
    wigner_d_l_m_2,
    integrand,
    ell_p_range,
    ):

    total_num_l_m = int((l_max - min_ell + 1)*(l_max + min_ell + 2)/2) # the total number of wigner_d matrices for \lambda =+/-2 
    num_l_m = max_ell * (max_ell + 1) + max_ell + 1 - min_ell * min_ell     # = Sum[2 l+1, {l , l_min, l_max}]
    num_l_m_p = ell_p_range[1] * (ell_p_range[1] + 1) + ell_p_range[1] + 1 - min_ell * min_ell 
    C_lmlpmp = np.zeros((6, num_l_m, num_l_m_p), dtype=np.complex128) 
    C_lmlpmp_cross = np.zeros((3, num_l_m_p, num_l_m), dtype=np.complex128)  
    # integrand[:, :, :, :] = 1.0
 
    ipow = np.array([1, 1j, -1, -1j])
    shortle = np.array([1, -1])

    for l in range(min_ell, max_ell + 1):

      coeff_E_B_ell = sqrt(pi* (2 * l + 1) / 2)
      coeff_T_ell = sqrt(pi* (2 * l + 1) * (l + 2) * (l + 1) * l * (l - 1) / 2 ) 

      for l_p in range(l, ell_p_range[1]+1):
        coeff_E_B_lp = sqrt(pi* (2 * l_p + 1) / 2)
        coeff_T_lp = sqrt(pi* (2 * l_p + 1) * (l_p + 2) * (l_p + 1) * l_p * (l_p - 1) / 2 ) 

        # for non-zero m and m'
        for m in range(-l, l + 1):
          lm_index_cur = l * (l+1) + m - min_ell*min_ell # l**2 + l +m - l_min**2
          abs_m = np.abs(m)
          wigner_cur_index = lm_index[l, abs_m]

          if m<0:
            wigner_D_l_m_plus2 = shortle[abs_m%2] * wigner_d_l_m_2[theta_unique_index[:], total_num_l_m + wigner_cur_index] * np.exp(1j * abs_m * phi[:])
            wigner_D_l_m_minus2 = shortle[abs_m%2] * wigner_d_l_m_2[theta_unique_index[:], wigner_cur_index] * np.exp(1j * abs_m * phi[:])
          else:
            wigner_D_l_m_plus2 = wigner_d_l_m_2[theta_unique_index[:], wigner_cur_index] * np.exp(-1j * abs_m * phi[:]) 
            wigner_D_l_m_minus2 =  wigner_d_l_m_2[theta_unique_index[:], total_num_l_m + wigner_cur_index] * np.exp(-1j * abs_m * phi[:])

          
          # Only do m-mp = 0 mod 2  
          for m_p in range(-l_p, l_p + 1):
            if (m_p-m)%2 ==1:
              continue

            lm_p_index_cur = l_p * (l_p+1) + m_p  - min_ell*min_ell

            abs_m_p = np.abs(m_p)
            wigner_p_cur_index = lm_index[l_p, abs_m_p]

            if m_p <0:
              wigner_D_l_m_p_plus2 = shortle[abs_m_p%2] * wigner_d_l_m_2[theta_unique_index[:], total_num_l_m + wigner_p_cur_index] * np.exp(1j * abs_m_p * phi[:])
              wigner_D_l_m_p_minus2 = shortle[abs_m_p%2] * wigner_d_l_m_2[theta_unique_index[:], wigner_p_cur_index] * np.exp(1j * abs_m_p * phi[:])
            else:
              wigner_D_l_m_p_plus2 = wigner_d_l_m_2[theta_unique_index[:], wigner_p_cur_index] * np.exp(-1j * abs_m_p * phi[:])
              wigner_D_l_m_p_minus2 =  wigner_d_l_m_2[theta_unique_index[:], total_num_l_m + wigner_p_cur_index] * np.exp(-1j * abs_m_p * phi[:]) 

            Xi_plus = wigner_D_l_m_plus2* np.conjugate(wigner_D_l_m_p_plus2) + wigner_D_l_m_minus2 *np.conjugate(wigner_D_l_m_p_minus2)
            Xi_minus = wigner_D_l_m_minus2 *np.conjugate(wigner_D_l_m_p_minus2) - wigner_D_l_m_plus2* np.conjugate(wigner_D_l_m_p_plus2)
          
            
            # TT correlations
            C_lmlpmp[0, lm_index_cur, lm_p_index_cur] += (coeff_T_ell * coeff_T_lp * ipow[(l-l_p)%4] *
                                                                    np.sum(integrand[0, k_amp_unique_index[:], l, l_p] 
                                                                     * Xi_plus)
                                                        )

          
            # EE correlations
            C_lmlpmp[1, lm_index_cur, lm_p_index_cur] += (coeff_E_B_ell * coeff_E_B_lp * ipow[(l-l_p)%4] *
                                                                    np.sum(integrand[1, k_amp_unique_index[:], l, l_p] 
                                                                     * Xi_plus)
                                                        )
            
            # BB correlations
            C_lmlpmp[2, lm_index_cur, lm_p_index_cur] += (coeff_E_B_ell * coeff_E_B_lp * ipow[(l-l_p)%4] *
                                                                    np.sum(integrand[2, k_amp_unique_index[:], l, l_p] 
                                                                     * Xi_plus)
                                                        )
            # TE correlations
            C_lmlpmp[3, lm_index_cur, lm_p_index_cur] += (coeff_T_ell * coeff_E_B_lp * ipow[(l-l_p)%4] *
                                                                    np.sum(integrand[3, k_amp_unique_index[:], l, l_p] 
                                                                     * Xi_plus)
                                                        )
            # EB correlations
            C_lmlpmp[4, lm_index_cur, lm_p_index_cur] += (coeff_E_B_ell * coeff_E_B_lp * ipow[(l-l_p)%4] *
                                                                    np.sum(integrand[4, k_amp_unique_index[:], l, l_p] 
                                                                     * Xi_minus)
                                                        )
            # TB correlations
            C_lmlpmp[5, lm_index_cur, lm_p_index_cur] += (coeff_T_ell * coeff_E_B_lp * ipow[(l-l_p)%4] *
                                                                    np.sum(integrand[5, k_amp_unique_index[:], l, l_p] 
                                                                     * Xi_minus)
                                                        )
            if l != l_p:
              # TE correlations

              C_lmlpmp_cross[0, lm_p_index_cur, lm_index_cur] += (coeff_T_lp * coeff_E_B_ell * ipow[(l_p-l)%4] *
                                                                      np.sum(integrand[3, k_amp_unique_index[:], l_p, l] 
                                                                      * conjugate(Xi_plus))
                                                                  )
              # EB correlations

              C_lmlpmp_cross[1, lm_p_index_cur, lm_index_cur] += (coeff_E_B_lp * coeff_E_B_ell * ipow[(l_p-l)%4] *
                                                                      np.sum(integrand[4, k_amp_unique_index[:], l_p, l] 
                                                                      * conjugate(Xi_minus))
                                                                  )

              # TB correlations
              C_lmlpmp_cross[2, lm_p_index_cur, lm_index_cur] += (coeff_T_lp * coeff_E_B_ell * ipow[(l_p-l)%4] *
                                                                      np.sum(integrand[5, k_amp_unique_index[:], l_p, l] 
                                                                      * conjugate(Xi_minus))
                                                                  )

    C_lmlpmp *= pi*pi / (2 * V)
    C_lmlpmp_cross *= pi*pi / (2 * V)

    
    return C_lmlpmp, C_lmlpmp_cross
