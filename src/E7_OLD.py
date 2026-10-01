from .tools import *
from .topology import Topology
import numpy as np
from numpy import pi, sin, tan, sqrt, conjugate, exp, dot
from numba import njit, prange, jit
from numba_progress import ProgressBar


class E7(Topology):
  def __init__(self, param, debug=True, make_run_folder = False):
    L_LSS = 13824.9 * 2
    self.LAx = param['LAx'] * L_LSS
    self.LAy = param['LAy'] * L_LSS
    self.L1y = param['L1y'] * L_LSS
    self.L2x = param['L2x'] * L_LSS
    self.L2z = param['L2z'] * L_LSS
    self.x0 = param['x0'] * L_LSS
 
    self.V = self.LAx * self.L1y * self.L2z 
    self.l_max = param['l_max']
    self.l_min = param['l_min']

    self.param = param
    
    print('Running - E7 l_max={}_LAx_{}_LAy_{}_L1y_{}_L2x_{}_L2z_{}_x_{}_y_{}_z_{}'.format(self.l_max, 
    int(self.LAx), int(self.LAy), 
    int(self.L1y), int(self.L2x), int(self.L2z), 
    "{:.2f}".format(param['x0'][0]),
    "{:.2f}".format(param['x0'][1]),
    "{:.2f}".format(param['x0'][2])))
    self.root = 'runs/{}_LAx_{}_LAy_{}_L1y_{}_L2x_{}_L2z_{}_x_{}_y_{}_z_{}_l_max_{}/'.format(
        param['topology'],
        "{:.2f}".format(param['LAx']),
        "{:.2f}".format(param['LAy']),
        "{:.2f}".format(param['L1y']),
        "{:.2f}".format(param['L2x']),
        "{:.2f}".format(param['L2z']),
        "{:.2f}".format(param['x0'][0]),
        "{:.2f}".format(param['x0'][1]),
        "{:.2f}".format(param['x0'][2]),
        self.l_max
    )

    Topology.__init__(self, param, debug, make_run_folder)


  def get_c_lmlpmp_per_process_multi(
    self,
    process_i,
    return_dict,
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
    ell_range,
    ell_p_range
  ):
    # This function seems unnecessary, but Numba does not allow return_dict
    # which is of type multiprocessing.Manager 

    return_dict[process_i] = get_c_lmlpmp(
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
      ell_range,
      ell_p_range,
      self.tilde_xi,
      self.first_eigenmode_index_split,
      self.second_eigenmode_index_split
    )



  def get_list_of_k_phi_theta(self):

    k_amp, phi, theta, self.tilde_xi, self.first_eigenmode_index_split, self.second_eigenmode_index_split = get_list_of_k_phi_theta(max(self.k_max_list), self.k_list[0], self.LAx, 
                                                                       self.LAy, self.L1y, self.L2x, self.L2z, self.x0)
    return k_amp, phi, theta 
  


#@njit(parallel = False)
def get_list_of_k_phi_theta(k_max, k_min, L_Ax, L_Ay, L_1y, L_2x, L_2z, x0):
    
#    x0 = np.array([0.0, 0.0, 0.0]) 
    M_A = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    T_A = np.array([L_Ax, L_Ay, 0])

    n_x_max = int(np.ceil((k_max * L_Ax) / np.pi))
    n_y_max = int(np.ceil((k_max * L_1y) / (2 * np.pi)))
    n_z_max = int(np.ceil(((k_max * L_2x) + (k_max * L_2z)) / (2 * np.pi)))

    list_length = n_x_max * n_y_max * n_z_max * 8
    k_amp = np.zeros(list_length)
    phi = np.zeros(list_length)
    theta = np.zeros(list_length)

    tilde_xi = np.zeros((list_length, 2), dtype=np.complex128)
    
    cur_index = 0
    
    # 1 Subspace
    n_x_array = np.concatenate([np.arange(-n_x_max, 0), np.arange(1, n_x_max+1)])
    n_z_array = np.concatenate([np.arange(-n_z_max, 0), np.arange(1, n_z_max+1)])

    k_y = 0 
    for n_x in n_x_array:
        for n_z in n_z_array:

            k_x = (np.pi * n_x) / L_Ax
            k_z = (2 * np.pi / L_2z) * (n_z - ((n_x / (2 * L_Ax)) * L_2x))
            
            k_vec = np.array([k_x, k_y, k_z])
            k_amp_cur = np.linalg.norm(k_vec)
            
            if k_amp_cur > k_max or k_amp_cur < k_min :
                continue           
            
            if k_amp_cur == 0:
                print(f"index = {cur_index} and k_vec = {k_vec} ")
                continue

            phi_null, theta_null = cart2spherical(k_vec/k_amp_cur)
            k_amp[cur_index] = k_amp_cur
            phi[cur_index] = phi_null
            theta[cur_index] = theta_null

            tilde_xi[cur_index, 0] = np.exp(-1j * k_vec @ x0)
            tilde_xi[cur_index, 1] = np.exp(-1j * k_vec @ (x0 - T_A)) 
            
            cur_index += 1

    subspace1_interval = [0, cur_index]

    # 2 Subspace

    modes_1 = [(0, ny, 0) for ny in range(1, n_y_max + 1)] # This is because now the first subspace is in the second
    modes_2 = [(nx, ny, nz) for nx in range(-n_x_max, n_x_max + 1) if nx != 0 for ny in range(1, n_y_max + 1) for nz in range(-n_z_max, n_z_max + 1)]
    modes_3 = [(nx, ny, nz) for nx in range(-n_x_max, n_x_max + 1) for ny in range(1, n_y_max + 1) for nz in range(-n_z_max, n_z_max + 1) if nz != 0]
    modes = np.concatenate((modes_1, modes_2, modes_3))

    for (n_x, n_y, n_z) in modes:
                
        k_x = (np.pi * n_x) / L_Ax
        k_y = (2 * np.pi * n_y) / L_1y
        k_z = (2 * np.pi / L_2z) * (n_z - ((n_x * L_2x) / (2 * L_Ax)))
        
        k_vec = np.array([k_x, k_y, k_z])
        k_amp_cur = np.linalg.norm(k_vec)
        
        if k_amp_cur > k_max or k_amp_cur < k_min :
            continue

        if k_amp_cur == 0:
            print(f"index = {cur_index} and k_vec = {k_vec} ")
            continue
        
        phi_null, theta_null = cart2spherical(k_vec/k_amp_cur)
        k_amp[cur_index] = k_amp_cur
        phi[cur_index] = phi_null
        theta[cur_index] = theta_null

        tilde_xi[cur_index, 0] = np.exp(-1j * k_vec @ x0)
#        tilde_xi[cur_index, 1] = np.exp(-1j * (((M_A @ k_vec) @ x0) - (k_vec @ T_A))) 
        tilde_xi[cur_index, 1] = np.exp(-1j * (k_vec @ ((M_A @ x0) - T_A)))
     
        cur_index += 1
                
    k_amp = k_amp[:cur_index]
    phi = phi[:cur_index]   
    theta = theta[:cur_index]
    tilde_xi = tilde_xi[:cur_index, :]
    subspace2_interval = [subspace1_interval[1], cur_index]
                
    print(f"Intervals: \n 1 supspace - {subspace1_interval} \n 2 subspace - {subspace2_interval}")
                  
    return k_amp, phi, theta, tilde_xi, subspace1_interval, subspace2_interval


@njit(nogil=True, parallel=False)
def get_c_lmlpmp(min_ell,
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
                 tilde_xi,
                 subspace1_interval, 
                 subspace2_interval
                  ):

    total_num_l_m = (l_max - l_min + 1) * (l_max + l_min + 2) // 2 
    total_number_of_xis = (ell_range[1] * (ell_range[1] + 1) + ell_range[1] + 1 - ell_range[0] * ell_range[0]) 
    C_lmlpmp = np.zeros((6, total_number_of_xis, total_number_of_xis), dtype=np.complex128)

    ipow = np.array([1, 1j, -1, -1j])
    m_list = np.arange(0, l_max+1)
    shortle = np.array([1, -1])
    min_k_amp = np.min(k_amp)

    oversqrt2 = 1/np.sqrt(2)

    # 1 Subspace
    for i in range(subspace1_interval[0], subspace1_interval[1]): 
        
        k_amp_cur = k_amp[i]
        wigner_d_l_m_index = theta_unique_index[i]
        k_unique_index_cur = k_amp_unique_index[i]
        phase_list_minus = np.exp(-1j * phi[i] * m_list)
        phase_list_plus  = np.exp(-1j * phi[i] * m_list)
        
        for ell in range(l_min, l_max + 1):
            
            coef_T_ell = ipow[ell%4] * sqrt(pi* (2 * ell + 1) * (ell + 2) * (ell + 1) * ell * (ell - 1) / 2)
            coef_E_B_ell = ipow[ell%4] * sqrt(pi* (2 * ell + 1)/2)
            
            for m in range(-ell, ell + 1):

                abs_m = np.abs(m)
                wigner_d_l_m_2_cur_index = lm_index[ell, abs_m]
                lm_index_cur = ell * (ell+1) + m - l_min * l_min

                if m<0:
                    wigner_d_l_m_plus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m]
                    wigner_d_l_m_minus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m]
                else:
                    wigner_d_l_m_plus2 = wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m] 
                    wigner_d_l_m_minus2 =  wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m]  

                xi_lm_plus =  tilde_xi[i, 0] * wigner_d_l_m_plus2
                xi_lm_minus =  tilde_xi[i, 1] * wigner_d_l_m_minus2
                
                for ell_p in range(l_min, l_max + 1):
                    
#                    if k_amp_cur > np.sqrt(k_max_list[ell]*k_max_list[ell_p]):
#                        continue  

                    coef_T_ell_p = ipow[ell_p%4] * sqrt(pi* (2 * ell_p + 1) * (ell_p + 2) * (ell_p + 1) * ell_p * (ell_p - 1) / 2)
                    coef_E_B_ell_p = ipow[ell_p%4] * sqrt(pi* (2 * ell_p + 1)/2)

                    for m_p in range(-ell_p, ell_p + 1):
                
                        abs_m_p = np.abs(m_p)
                        wigner_d_l_m_2_cur_index = lm_index[ell_p, abs_m_p] 
                        
                        lm_p_index_cur = ell_p * (ell_p+1) + m_p - l_min * l_min          

                        if m_p<0:
                            wigner_d_l_m_p_plus2 = shortle[abs_m_p%2] * wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m_p]
                            wigner_d_l_m_p_minus2 = shortle[abs_m_p%2] * wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m_p]
                        else:
                            wigner_d_l_m_p_plus2 = wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p] 
                            wigner_d_l_m_p_minus2 =  wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p]  
       
                        xi_lm_p_plus = tilde_xi[i, 0] * wigner_d_l_m_p_plus2
                        xi_lm_p_minus =  tilde_xi[i, 1] * wigner_d_l_m_p_minus2

                        Xi_plus = (xi_lm_plus * conjugate(xi_lm_p_plus) + xi_lm_minus * conjugate(xi_lm_p_minus))
                        Xi_minus = (xi_lm_minus * conjugate(xi_lm_p_minus) - xi_lm_plus * conjugate(xi_lm_p_plus))

                        # TT correlations
                        C_lmlpmp[0, lm_index_cur, lm_p_index_cur] += coef_T_ell * coef_T_ell_p * integrand[0, k_unique_index_cur, ell, ell_p] * Xi_plus 

                        # EE correlations
                        C_lmlpmp[1, lm_index_cur, lm_p_index_cur] += coef_E_B_ell * coef_E_B_ell_p * integrand[1, k_unique_index_cur, ell, ell_p] * Xi_plus

                        # BB correlations
                        C_lmlpmp[2, lm_index_cur, lm_p_index_cur] += coef_E_B_ell * coef_E_B_ell_p * integrand[2, k_unique_index_cur, ell, ell_p] * Xi_plus

                        # TE correlations
                        C_lmlpmp[3, lm_index_cur, lm_p_index_cur] += coef_T_ell * coef_E_B_ell_p * integrand[3, k_unique_index_cur, ell, ell_p] * Xi_plus

                        # EB correlations
                        C_lmlpmp[4, lm_index_cur, lm_p_index_cur] += coef_E_B_ell * coef_E_B_ell_p * integrand[4, k_unique_index_cur, ell, ell_p] * Xi_minus

                        # TB correlations
                        C_lmlpmp[5, lm_index_cur, lm_p_index_cur] += coef_T_ell * coef_E_B_ell_p * integrand[5, k_unique_index_cur, ell, ell_p] * Xi_minus


    # 2 Subspace
    for i in range(subspace2_interval[0], subspace2_interval[1]):
        
        k_amp_cur = k_amp[i]
        wigner_d_l_m_index = theta_unique_index[i]
        k_unique_index_cur = k_amp_unique_index[i]
        phase_list_minus = np.exp(-1j * phi[i] * m_list)
        phase_list_plus  = np.exp(1j * phi[i] * m_list)
    
        for ell in range(l_min, l_max + 1):
            
            coef_T_ell = ipow[ell%4] * sqrt(pi* (2 * ell + 1) * (ell + 2) * (ell + 1) * ell * (ell - 1) / 2)
            coef_E_B_ell = ipow[ell%4] * sqrt(pi* (2 * ell + 1) / 2)

            if ell_p_range[0] > ell:
                l_start = ell_p_range[0]
            else:
                l_start = ell
            
            for m in range(-ell, ell + 1):
                
                abs_m = np.abs(m)
                wigner_d_l_m_2_cur_index = lm_index[ell, abs_m]   
                lm_index_cur = ell * (ell+1) + m - l_min * l_min
                
                if m<0:
                    wigner_d_l_m_plus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m]
                    wigner_d_l_m_minus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m]
                else:
                    wigner_d_l_m_plus2 = wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m] 
                    wigner_d_l_m_minus2 =  wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m]  

                xi_lm_plus =  (tilde_xi[i, 0] * wigner_d_l_m_plus2 + tilde_xi[i, 1] * conjugate(wigner_d_l_m_minus2))
                xi_lm_minus =  (tilde_xi[i, 0] * wigner_d_l_m_minus2 + tilde_xi[i, 1] * conjugate(wigner_d_l_m_plus2))

                xi_lm_B_plus = (tilde_xi[i, 0] * wigner_d_l_m_plus2 - tilde_xi[i, 1] * conjugate(wigner_d_l_m_minus2))
                xi_lm_B_minus = (tilde_xi[i, 0] * wigner_d_l_m_minus2 - tilde_xi[i, 1] * conjugate(wigner_d_l_m_plus2))

                for ell_p in range(l_min, l_max + 1):
                    
                    coef_T_ell_p = ipow[ell_p%4] *  sqrt(pi* (2 * ell_p + 1) * (ell_p + 2) * (ell_p + 1) * ell_p * (ell_p - 1) / 2)
                    coef_E_B_ell_p = ipow[ell_p%4] *sqrt(pi* (2 * ell_p + 1)/2)
                    
                    for m_p in range(-ell_p, ell_p + 1):
                
                        abs_m_p = np.abs(m_p)
                        wigner_d_l_m_2_cur_index = lm_index[ell_p, abs_m_p]
                        lm_p_index_cur = ell_p * (ell_p+1) + m_p - l_min * l_min

                        if m_p<0:
                            wigner_d_l_m_p_plus2 = shortle[abs_m_p%2] * wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m_p]
                            wigner_d_l_m_p_minus2 = shortle[abs_m_p%2] * wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m_p]
                        else:
                            wigner_d_l_m_p_plus2 = wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p] 
                            wigner_d_l_m_p_minus2 =  wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p]  

                        xi_lm_p_plus = (tilde_xi[i, 0] * wigner_d_l_m_p_plus2 + tilde_xi[i, 1] * conjugate(wigner_d_l_m_p_minus2))
                        xi_lm_p_minus = (tilde_xi[i, 0] * wigner_d_l_m_p_minus2 +  tilde_xi[i, 1] * conjugate(wigner_d_l_m_p_plus2))

                        xi_lm_B_p_plus = (tilde_xi[i, 0] * wigner_d_l_m_p_plus2 - tilde_xi[i, 1] * conjugate(wigner_d_l_m_p_minus2))
                        xi_lm_B_p_minus = (tilde_xi[i, 0] * wigner_d_l_m_p_minus2 - tilde_xi[i, 1] * conjugate(wigner_d_l_m_p_plus2))
                    
                        Xi_T = (xi_lm_plus * conjugate(xi_lm_p_plus) + xi_lm_minus * conjugate(xi_lm_p_minus))
                        Xi_B = (xi_lm_B_minus * conjugate(xi_lm_B_p_minus) - xi_lm_B_plus * conjugate(xi_lm_B_p_plus))
                        Xi_TB = (xi_lm_minus * conjugate(xi_lm_B_p_minus) - xi_lm_plus * conjugate(xi_lm_B_p_plus))

                        # TT correlations
                        C_lmlpmp[0, lm_index_cur, lm_p_index_cur] += coef_T_ell * coef_T_ell_p  * integrand[0, k_unique_index_cur, ell, ell_p] * Xi_T

                        # EE correlations
                        C_lmlpmp[1, lm_index_cur, lm_p_index_cur] += coef_E_B_ell * coef_E_B_ell_p  * integrand[1, k_unique_index_cur, ell, ell_p] * Xi_T

                        # BB correlations
                        C_lmlpmp[2, lm_index_cur, lm_p_index_cur] += coef_E_B_ell * coef_E_B_ell_p  * integrand[2, k_unique_index_cur, ell, ell_p] * Xi_B 

                        # TE correlations
                        C_lmlpmp[3, lm_index_cur, lm_p_index_cur] += coef_T_ell * coef_E_B_ell_p  * integrand[3, k_unique_index_cur, ell, ell_p] * (-1) * Xi_T

                        # EB correlations
                        C_lmlpmp[4, lm_index_cur, lm_p_index_cur] += coef_E_B_ell * coef_E_B_ell_p  * integrand[4, k_unique_index_cur, ell, ell_p] * (-1) * Xi_TB

                        # TB correlations
                        C_lmlpmp[5, lm_index_cur, lm_p_index_cur] += coef_T_ell * coef_E_B_ell_p  * integrand[5, k_unique_index_cur, ell, ell_p] * Xi_TB

    C_lmlpmp *= pi*pi / (2 * V) 

    return C_lmlpmp





