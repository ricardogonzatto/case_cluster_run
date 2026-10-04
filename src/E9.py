from .tools import *
from .topology import Topology
import numpy as np
from numpy import pi, sin, tan, sqrt, conjugate, exp, dot
from numba import njit, prange, jit
from numba_progress import ProgressBar


class E9(Topology):
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
    
    print('Running - E9 l_max={}_LAx_{}_LAy_{}_L1y_{}_L2x_{}_L2z_{}_x_{}_y_{}_z_{}'.format(self.l_max, 
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
    n_z_max = int(np.ceil(((L_2z / 2 * np.pi) * (k_max + ((np.pi * L_2x * n_x_max) / (L_2z * L_Ax)) + ((np.pi * n_y_max) / L_2z)))))

    list_length = n_x_max * n_y_max * n_z_max * 8
    k_amp = np.zeros(list_length)
    phi = np.zeros(list_length)
    theta = np.zeros(list_length)

    tilde_xi = np.zeros((list_length, 2), dtype=np.complex128)
    
    cur_index = 0
    
    # 1 Subspace
    
    modes = [(nx, 0, nz) for nx in range(-n_x_max, n_x_max + 1) if nx % 2 == 0 for nz in range(-n_z_max, n_z_max + 1) if nx != 0 or nz != 0]
    k_y = 0

    for (n_x, n_y, n_z) in modes:

        k_x = (np.pi * n_x) / L_Ax
        k_z = (2 * np.pi / L_2z) * (n_z - (((n_x * L_2x) / (2 * L_Ax)) ) - (n_y / 2))
        
        k_vec = np.array([k_x, k_y, k_z])
        k_amp_cur = np.linalg.norm(k_vec)
        
        if k_amp_cur > k_max or k_amp_cur < k_min :
            continue           
        
        phi_null, theta_null = cart2spherical(k_vec/k_amp_cur)
        k_amp[cur_index] = k_amp_cur
        phi[cur_index] = phi_null
        theta[cur_index] = theta_null

        tilde_xi[cur_index, 0] = np.exp(-1j * k_vec @ x0)/np.sqrt(2)
        tilde_xi[cur_index, 1] = np.exp(-1j * k_vec @ (x0 - T_A))/np.sqrt(2)
        
        cur_index += 1

    subspace1_interval = [0, cur_index]

    # 2 Subspace

    modes = [(nx, ny, nz) for nx in range(-n_x_max, n_x_max + 1) for ny in range(1, n_y_max + 1) for nz in range(-n_z_max, n_z_max + 1) if not (nx == 0 and ny == 0 and nz == 0)]   

    for (n_x, n_y, n_z) in modes:
                
        k_x = (np.pi * n_x) / L_Ax
        k_y = (2 * np.pi * n_y) / L_1y
        k_z = (2 * np.pi / L_2z) * (n_z - ((n_x * L_2x) / (2 * L_Ax) - (n_y / 2)))

        k_vec = np.array([k_x, k_y, k_z])
        k_amp_cur = np.linalg.norm(k_vec)
        
        if k_amp_cur > k_max or k_amp_cur < k_min :
            continue

        phi_null, theta_null = cart2spherical(k_vec/k_amp_cur)
        k_amp[cur_index] = k_amp_cur
        phi[cur_index] = phi_null
        theta[cur_index] = theta_null

        tilde_xi[cur_index, 0] = np.exp(-1j * k_vec @ x0)/np.sqrt(2)
#        tilde_xi[cur_index, 1] = np.exp(-1j * (((M_A @ k_vec) @ x0) - (k_vec @ T_A))) 
        tilde_xi[cur_index, 1] = np.exp(-1j * (k_vec @ ((M_A @ x0) - T_A)))/np.sqrt(2)
     
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

    # 1 Subspace
    for i in range(subspace1_interval[0], subspace1_interval[1]): 
        
        k_amp_cur = k_amp[i]
        wigner_d_l_m_index = theta_unique_index[i]
        k_unique_index_cur = k_amp_unique_index[i]
        phase_list_minus = np.exp(-1j * phi[i] * m_list)
        phase_list_plus  = np.exp(1j * phi[i] * m_list)

        cur_tilde_xi = tilde_xi[i, :]
        
        for ell in range(min_ell, max_ell + 1):
            
            coef_T_ell = sqrt(pi* (2 * ell + 1) * (ell + 2) * (ell + 1) * ell * (ell - 1) / 2)
            coef_E_B_ell = sqrt(pi* (2 * ell + 1)/2)

            if ell_p_range[0] > ell:
                l_start = ell_p_range[0]
            else:
                l_start = ell
            
            for m in range(-ell, ell + 1):

                abs_m = np.abs(m)
                wigner_d_l_m_2_cur_index = lm_index[ell, abs_m]
                lm_index_cur = ell * (ell+1) + m - ell_range[0] * ell_range[0]

                if m<0:
                    wigner_D_l_m_plus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m]
                    wigner_D_l_m_minus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m]
                else:
                    wigner_D_l_m_plus2 = wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m] 
                    wigner_D_l_m_minus2 =  wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m]  


                xi_T_lm = cur_tilde_xi[0] * wigner_D_l_m_plus2 + cur_tilde_xi[1] * wigner_D_l_m_minus2
                xi_E_lm = - xi_T_lm
                xi_B_lm = cur_tilde_xi[0] * wigner_D_l_m_plus2 - cur_tilde_xi[1] * wigner_D_l_m_minus2

                for ell_p in range(l_start, ell_p_range[1] + 1):

                    coef_T_ell_p = ipow[(ell - ell_p)%4] * sqrt(pi* (2 * ell_p + 1) * (ell_p + 2) * (ell_p + 1) * ell_p * (ell_p - 1) / 2)
                    coef_E_B_ell_p = ipow[(ell - ell_p)%4] * sqrt(pi* (2 * ell_p + 1)/2)

                    coef_T_ell_p_low = ipow[(ell_p - ell)%4] * sqrt(pi* (2 * ell_p + 1) * (ell_p + 2) * (ell_p + 1) * ell_p * (ell_p - 1) / 2)
                    coef_E_B_ell_p_low = ipow[(ell_p - ell)%4] * sqrt(pi* (2 * ell_p + 1)/2)

                    for m_p in range(0, ell_p + 1):
                
                        abs_m_p = np.abs(m_p)
                        wigner_d_l_m_2_cur_index = lm_index[ell_p, abs_m_p] 
                        
                        lm_p_index_cur = ell_p * (ell_p+1) + m_p - ell_p_range[0] * ell_p_range[0]        

                        wigner_D_l_m_p_plus2 = wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p] 
                        wigner_D_l_m_p_minus2 =  wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p]  
    

                        xi_T_lm_p = cur_tilde_xi[0] * wigner_D_l_m_p_plus2 + cur_tilde_xi[1] * wigner_D_l_m_p_minus2
                        xi_E_lm_p = - xi_T_lm_p
                        xi_B_lm_p = cur_tilde_xi[0] * wigner_D_l_m_p_plus2 - cur_tilde_xi[1] * wigner_D_l_m_p_minus2
                                                                        
                        Xi_TT = xi_T_lm * conjugate(xi_T_lm_p)
                        Xi_EE = xi_E_lm * conjugate(xi_E_lm_p)
                        Xi_TE = xi_T_lm * conjugate(xi_E_lm_p)
                        Xi_BB = xi_B_lm * conjugate(xi_B_lm_p)
                        Xi_EB = xi_E_lm * conjugate(xi_B_lm_p)
                        Xi_TB = xi_T_lm * conjugate(xi_B_lm_p)
                        
                        # TT correlations
                        C_lmlpmp[0, lm_index_cur, lm_p_index_cur] += coef_T_ell * coef_T_ell_p * integrand[0, k_unique_index_cur, ell, ell_p] * Xi_TT 

                        # EE correlations
                        C_lmlpmp[1, lm_index_cur, lm_p_index_cur] += coef_E_B_ell * coef_E_B_ell_p * integrand[1, k_unique_index_cur, ell, ell_p] * Xi_EE

                        # BB correlations
                        C_lmlpmp[2, lm_index_cur, lm_p_index_cur] += coef_E_B_ell * coef_E_B_ell_p * integrand[2, k_unique_index_cur, ell, ell_p] * Xi_BB

                        # TE correlations
                        C_lmlpmp[3, lm_index_cur, lm_p_index_cur] += coef_T_ell * coef_E_B_ell_p * integrand[3, k_unique_index_cur, ell, ell_p] * Xi_TE

                        # EB correlations
                        C_lmlpmp[4, lm_index_cur, lm_p_index_cur] += coef_E_B_ell * coef_E_B_ell_p * integrand[4, k_unique_index_cur, ell, ell_p] * Xi_EB

                        # TB correlations
                        C_lmlpmp[5, lm_index_cur, lm_p_index_cur] += coef_T_ell * coef_E_B_ell_p * integrand[5, k_unique_index_cur, ell, ell_p] * Xi_TB

                        if ell != ell_p:

                            Xi_TE_low = xi_T_lm_p * conjugate(xi_E_lm)
                            Xi_EB_low = xi_E_lm_p * conjugate(xi_B_lm)
                            Xi_TB_low = xi_T_lm_p * conjugate(xi_B_lm)
                            # TE correlations
                            C_lmlpmp[3, lm_p_index_cur, lm_index_cur] += coef_T_ell_p_low * coef_E_B_ell * integrand[3, k_unique_index_cur, ell_p, ell] * Xi_TE_low
                            # EB correlations
                            C_lmlpmp[4, lm_p_index_cur, lm_index_cur] += coef_E_B_ell_p_low * coef_E_B_ell * integrand[4, k_unique_index_cur, ell_p, ell] * Xi_EB_low
                            # TB correlations
                            C_lmlpmp[5, lm_p_index_cur, lm_index_cur] += coef_T_ell_p_low * coef_E_B_ell * integrand[5, k_unique_index_cur, ell_p, ell] * Xi_TB_low

    # 2 Subspace
    for i in range(subspace2_interval[0], subspace2_interval[1]):
        
        k_amp_cur = k_amp[i]
        wigner_d_l_m_index = theta_unique_index[i]
        k_unique_index_cur = k_amp_unique_index[i]
        phase_list_minus = np.exp(-1j * phi[i] * m_list)
        phase_list_plus  = np.exp(1j * phi[i] * m_list)

        cur_tilde_xi = tilde_xi[i, :]
    
        for ell in range(min_ell, max_ell + 1):
            
            coef_T_ell = sqrt(pi* (2 * ell + 1) * (ell + 2) * (ell + 1) * ell * (ell - 1) / 2)
            coef_E_B_ell = sqrt(pi* (2 * ell + 1) / 2)

            if ell_p_range[0] > ell:
                l_start = ell_p_range[0]
            else:
                l_start = ell
            
            for m in range(-ell, ell + 1):
                
                abs_m = np.abs(m)
                wigner_d_l_m_2_cur_index = lm_index[ell, abs_m]   
                lm_index_cur = ell * (ell+1) + m - ell_range[0] * ell_range[0]
                
                if m<0:
                    wigner_D_l_m_plus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m]
                    wigner_D_l_m_minus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m]
                else:
                    wigner_D_l_m_plus2 = wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m] 
                    wigner_D_l_m_minus2 =  wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m]  

                xi_T_lm_plus = (cur_tilde_xi[0] * wigner_D_l_m_plus2 + cur_tilde_xi[1] * conjugate(wigner_D_l_m_minus2))
                xi_T_lm_minus = (cur_tilde_xi[0] * wigner_D_l_m_minus2 + cur_tilde_xi[1] * conjugate(wigner_D_l_m_plus2))

                xi_E_lm_plus = - xi_T_lm_plus
                xi_E_lm_minus = - xi_T_lm_minus

                xi_B_lm_plus = -(cur_tilde_xi[0] * wigner_D_l_m_plus2 - cur_tilde_xi[1] * conjugate(wigner_D_l_m_minus2))
                xi_B_lm_minus = (cur_tilde_xi[0] * wigner_D_l_m_minus2 - cur_tilde_xi[1] * conjugate(wigner_D_l_m_plus2))

                for ell_p in range(l_start, ell_p_range[1] + 1):

                    coef_T_ell_p = ipow[(ell - ell_p)%4] * sqrt(pi* (2 * ell_p + 1) * (ell_p + 2) * (ell_p + 1) * ell_p * (ell_p - 1) / 2)
                    coef_E_B_ell_p = ipow[(ell - ell_p)%4] * sqrt(pi* (2 * ell_p + 1)/2)

                    coef_T_ell_p_low = ipow[(ell_p - ell)%4] * sqrt(pi* (2 * ell_p + 1) * (ell_p + 2) * (ell_p + 1) * ell_p * (ell_p - 1) / 2)
                    coef_E_B_ell_p_low = ipow[(ell_p - ell)%4] * sqrt(pi* (2 * ell_p + 1)/2)

                    for m_p in range(0, ell_p + 1):
                
                        abs_m_p = np.abs(m_p)
                        wigner_d_l_m_2_cur_index = lm_index[ell_p, abs_m_p]
                        lm_p_index_cur = ell_p * (ell_p+1) + m_p - ell_p_range[0] * ell_p_range[0]

                        wigner_D_l_m_p_plus2 = wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p] 
                        wigner_D_l_m_p_minus2 =  wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p]  

                        xi_T_lm_p_plus = (cur_tilde_xi[0] * wigner_D_l_m_p_plus2 + cur_tilde_xi[1] * conjugate(wigner_D_l_m_p_minus2))
                        xi_T_lm_p_minus = (cur_tilde_xi[0] * wigner_D_l_m_p_minus2+ cur_tilde_xi[1] * conjugate(wigner_D_l_m_p_plus2))
                          
                        xi_E_lm_p_plus = - xi_T_lm_p_plus
                        xi_E_lm_p_minus = - xi_T_lm_p_minus
                          
                        xi_B_lm_p_plus = -(cur_tilde_xi[0] * wigner_D_l_m_p_plus2 - cur_tilde_xi[1] * conjugate(wigner_D_l_m_p_minus2))
                        xi_B_lm_p_minus = (cur_tilde_xi[0] * wigner_D_l_m_p_minus2 - cur_tilde_xi[1] * conjugate(wigner_D_l_m_p_plus2))
                                                                                      
                        Xi_TT = (xi_T_lm_plus * conjugate(xi_T_lm_p_plus) + xi_T_lm_minus * conjugate(xi_T_lm_p_minus))
                        Xi_EE = (xi_E_lm_plus * conjugate(xi_E_lm_p_plus) + xi_E_lm_minus * conjugate(xi_E_lm_p_minus))
                        Xi_TE = (xi_T_lm_plus * conjugate(xi_E_lm_p_plus) + xi_T_lm_minus * conjugate(xi_E_lm_p_minus))
                        Xi_BB = (xi_B_lm_plus * conjugate(xi_B_lm_p_plus) + xi_B_lm_minus * conjugate(xi_B_lm_p_minus))
                        Xi_EB = (xi_E_lm_plus * conjugate(xi_B_lm_p_plus) + xi_E_lm_minus * conjugate(xi_B_lm_p_minus))
                        Xi_TB = (xi_T_lm_plus * conjugate(xi_B_lm_p_plus) + xi_T_lm_minus * conjugate(xi_B_lm_p_minus))
                        
                        # TT correlations
                        C_lmlpmp[0, lm_index_cur, lm_p_index_cur] += coef_T_ell * coef_T_ell_p  * integrand[0, k_unique_index_cur, ell, ell_p] * Xi_TT

                        # EE correlations
                        C_lmlpmp[1, lm_index_cur, lm_p_index_cur] += coef_E_B_ell * coef_E_B_ell_p  * integrand[1, k_unique_index_cur, ell, ell_p] * Xi_EE

                        # BB correlations
                        C_lmlpmp[2, lm_index_cur, lm_p_index_cur] += coef_E_B_ell * coef_E_B_ell_p  * integrand[2, k_unique_index_cur, ell, ell_p] * Xi_BB

                        # TE correlations
                        C_lmlpmp[3, lm_index_cur, lm_p_index_cur] += coef_T_ell * coef_E_B_ell_p  * integrand[3, k_unique_index_cur, ell, ell_p] * Xi_TE

                        # EB correlations
                        C_lmlpmp[4, lm_index_cur, lm_p_index_cur] += coef_E_B_ell * coef_E_B_ell_p  * integrand[4, k_unique_index_cur, ell, ell_p] * Xi_EB

                        # TB correlations
                        C_lmlpmp[5, lm_index_cur, lm_p_index_cur] += coef_T_ell * coef_E_B_ell_p  * integrand[5, k_unique_index_cur, ell, ell_p] * Xi_TB

                        if ell != ell_p:

                            Xi_TE_low = (xi_T_lm_p_plus * conjugate(xi_E_lm_plus) + xi_T_lm_p_minus * conjugate(xi_E_lm_minus))
                            Xi_EB_low = (xi_E_lm_p_plus * conjugate(xi_B_lm_plus) + xi_E_lm_p_minus * conjugate(xi_B_lm_minus))
                            Xi_TB_low = (xi_T_lm_p_plus * conjugate(xi_B_lm_plus) + xi_T_lm_p_minus * conjugate(xi_B_lm_minus))
                            # TE correlations
                            C_lmlpmp[3, lm_p_index_cur, lm_index_cur] += coef_T_ell_p_low * coef_E_B_ell * integrand[3, k_unique_index_cur, ell_p, ell] * Xi_TE_low
                            # EB correlations
                            C_lmlpmp[4, lm_p_index_cur, lm_index_cur] += coef_E_B_ell_p_low * coef_E_B_ell * integrand[4, k_unique_index_cur, ell_p, ell] * Xi_EB_low
                            # TB correlations
                            C_lmlpmp[5, lm_p_index_cur, lm_index_cur] += coef_T_ell_p_low * coef_E_B_ell * integrand[5, k_unique_index_cur, ell_p, ell] * Xi_TB_low



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





