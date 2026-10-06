from .tools import *
from .topology import Topology
import numpy as np
from numpy import pi, sin, tan, sqrt, conjugate, exp, dot
from numba import njit, prange, jit
from numba_progress import ProgressBar

class E10(Topology):
  def __init__(self, param, debug=True, make_run_folder = False):
    L_LSS = 13824.9 * 2
    self.LAx = param['LAx'] * L_LSS
    self.LAy = param['LAy'] * L_LSS
    self.LBx = param['LBx'] * L_LSS
    self.LBz = param['LBz'] * L_LSS
    self.LCy = param['LCy'] * L_LSS
    self.x0 = param['x0'] * L_LSS
 
    self.V = 4 * self.LAx * self.LBz * self.LCy
    self.l_max = param['l_max']
    self.l_min = param['l_min']

    self.param = param
    
    print('Running - E10 l_max={}_LAx_{}_LAy_{}_LBx_{}_LBz_{}_LCy_{}_x_{}_y_{}_z_{}'.format(self.l_max, 
    int(self.LAx), int(self.LAy), 
    int(self.LBx), int(self.LBz), int(self.LCy), 
    "{:.2f}".format(param['x0'][0]),
    "{:.2f}".format(param['x0'][1]),
    "{:.2f}".format(param['x0'][2])))
    self.root = 'runs/{}_LAx_{}_LAy_{}_LBx_{}_LBz_{}_LCy_{}_x_{}_y_{}_z_{}_l_max_{}/'.format(
        param['topology'],
        "{:.2f}".format(param['LAx']),
        "{:.2f}".format(param['LAy']),
        "{:.2f}".format(param['LBx']),
        "{:.2f}".format(param['LBz']),
        "{:.2f}".format(param['LCy']),
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
      self.subspace1_interval,
      self.subspace2_interval,
      self.subspace3_interval,
      self.subspace4_interval  
    )



  def get_list_of_k_phi_theta(self):

    k_amp, phi, theta, self.tilde_xi, self.subspace1_interval, self.subspace2_interval, self.subspace3_interval, self.subspace4_interval = get_list_of_k_phi_theta(max(self.k_max_list), self.k_list[0], self.LAx, 
                                                                       self.LAy, self.LBx, self.LBz, self.LCy, self.x0)
    return k_amp, phi, theta 
 

#@njit(parallel = False)
def get_list_of_k_phi_theta(k_max, k_min, LAx, LAy, LBx, LBz, LCy, x0):
    # Returns list of k, phi, and theta for this topology

    M_A = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    M_B = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    n_x_max = int(np.ceil(k_max * LAx / pi))
    n_y_max = int(np.ceil((k_max * LCy) / (2 * pi)))
    n_z_max = int(np.ceil(k_max * LBz / pi))
 
    list_length = n_x_max * n_y_max * n_z_max * 8
    k_amp = np.zeros(list_length)
    phi = np.zeros(list_length)
    theta = np.zeros(list_length)

    tilde_xi = np.zeros((list_length, 4), dtype=np.complex128)

    T_A = np.array([LAx, LAy, 0])
    T_B = np.array([LBx, LCy/2 , LBz])

    cur_index = 0

    # 1 Subspace 
    modes_N1 = [(0, 0, n_z) for n_z in range(-n_z_max, n_z_max+1) if n_z != 0]

    for (n_x, n_y, n_z) in modes_N1:

        k_x = (np.pi * n_x) / LAx
        k_y = (2 *np.pi * n_y) / LCy
        k_z = (np.pi * n_z) / LBz     

        k_vec = np.array([k_x, k_y, k_z])
        k_amp_cur = np.linalg.norm(k_vec)

        if k_amp_cur > k_max or k_amp_cur < k_min :
            continue

        phi_null, theta_null = cart2spherical(k_vec/k_amp_cur)
        k_amp[cur_index] = k_amp_cur
        phi[cur_index] = phi_null
        theta[cur_index] = theta_null

        tilde_xi[cur_index, 0] = np.exp(-1j * k_vec @ x0)
        tilde_xi[cur_index, 1] = ((-1)**n_z) * np.exp(-1j * k_vec @ x0)

        cur_index += 1

    subspace1_interval = [0, cur_index]

    # 2 Subspace 
    modes_N2 = [(0, ny, nz) for ny in range(1, n_y_max+1) for nz in range(-n_z_max, n_z_max+1)]

    for (n_x, n_y, n_z) in modes_N2:

        k_x = (np.pi * n_x) / LAx
        k_y = (2 *np.pi * n_y) / LCy
        k_z = (np.pi * n_z) / LBz     

        k_vec = np.array([k_x, k_y, k_z])
        k_amp_cur = np.linalg.norm(k_vec)

        if k_amp_cur > k_max or k_amp_cur < k_min :
            continue

        phi_null, theta_null = cart2spherical(k_vec/k_amp_cur)
        k_amp[cur_index] = k_amp_cur
        phi[cur_index] = phi_null
        theta[cur_index] = theta_null

        tilde_xi[cur_index, 0] = np.exp(-1j * k_vec @ x0)
        tilde_xi[cur_index, 1] = np.exp(-1j * k_vec @ ((M_A @ x0) - T_A - T_B ))

        cur_index += 1

    subspace2_interval = [subspace1_interval[1], cur_index]

    # 3 Subspace 
    modes_N3 = [(nx, 0, nz) for nx in range(1, n_x_max+1) for nz in range(-n_z_max, n_z_max+1)]

    for (n_x, n_y, n_z) in modes_N3:

        k_x = (np.pi * n_x) / LAx
        k_y = (2 *np.pi * n_y) / LCy
        k_z = (np.pi * n_z) / LBz     

        k_vec = np.array([k_x, k_y, k_z])
        k_amp_cur = np.linalg.norm(k_vec)

        if k_amp_cur > k_max or k_amp_cur < k_min :
            continue

        phi_null, theta_null = cart2spherical(k_vec/k_amp_cur)
        k_amp[cur_index] = k_amp_cur
        phi[cur_index] = phi_null
        theta[cur_index] = theta_null

        tilde_xi[cur_index, 0] = np.exp(-1j * k_vec @ x0)
        tilde_xi[cur_index, 1] = np.exp(-1j * k_vec @ ((M_B @ x0) - T_A - T_B ))

        cur_index += 1

    subspace3_interval = [subspace2_interval[1], cur_index]

    # 4 Subspace 
    modes_1 = [(nx, ny, nz) for nx in range(1, n_x_max + 1) for ny in range(-n_y_max, n_y_max+1) for nz in range(-n_z_max, n_z_max+1)]
    modes_2 = [(0, ny, nz) for ny in range(1, n_y_max + 1) for nz in range(-n_z_max, n_z_max+1)]

    modes_N4 = np.concatenate((modes_1, modes_2))

    for (n_x, n_y, n_z) in modes_N4:

        k_x = (np.pi * n_x) / LAx
        k_y = (2 *np.pi * n_y) / LCy
        k_z = (np.pi * n_z) / LBz     

        k_vec = np.array([k_x, k_y, k_z])
        k_amp_cur = np.linalg.norm(k_vec)

        if k_amp_cur > k_max or k_amp_cur < k_min :
            continue

        phi_null, theta_null = cart2spherical(k_vec/k_amp_cur)
        k_amp[cur_index] = k_amp_cur
        phi[cur_index] = phi_null
        theta[cur_index] = theta_null

        tilde_xi[cur_index, 0] = np.exp(-1j * k_vec @ x0)
        tilde_xi[cur_index, 1] = np.exp(-1j * k_vec @ ((M_A @ x0) - T_A ))
        tilde_xi[cur_index, 2] = np.exp(-1j * k_vec @ ((M_B @ x0) - T_B ))
        tilde_xi[cur_index, 3] = np.exp(-1j * k_vec @ ((M_A @ M_B @ x0) - (M_B @ T_A) - T_B))

        cur_index += 1

    subspace4_interval = [subspace3_interval[1], cur_index]

    k_amp = k_amp[:cur_index]
    phi = phi[:cur_index]   
    theta = theta[:cur_index]
    tilde_xi = tilde_xi[:cur_index, :]

    print('E7 Final num of elements:', k_amp.size, 'Minimum k_amp', np.amin(k_amp), 'n_x_max', n_x_max, 'n_z_max', n_z_max)
    return k_amp, phi, theta, tilde_xi, subspace1_interval, subspace2_interval, subspace3_interval, subspace4_interval


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
                 subspace2_interval,
                 subspace3_interval,
                 subspace4_interval
                  ):

    total_num_l_m = (l_max - l_min + 1) * (l_max + l_min + 2) // 2 
    total_number_of_xis = (ell_range[1] * (ell_range[1] + 1) + ell_range[1] + 1 - ell_range[0] * ell_range[0]) 

    ipow = np.array([1, 1j, -1, -1j])
    C_lmlpmp = np.zeros((6, total_number_of_xis, total_number_of_xis), dtype=np.complex128)
    m_list = np.arange(0, l_max+1)
    shortle = np.array([1, -1])
    min_k_amp = np.min(k_amp)

    oversqrt2 = 1/np.sqrt(2)
    oversqrt4 = 1/np.sqrt(4)

    # 1 Subspace
    for i in range(subspace1_interval[0], subspace1_interval[1]): 
        
        k_amp_cur = k_amp[i]
        wigner_d_l_m_index = theta_unique_index[i]
        k_unique_index_cur = k_amp_unique_index[i]
        phase_list_minus = np.exp(-1j * phi[i] * m_list)
        phase_list_plus  = np.exp(1j * phi[i] * m_list)

        for ell in range(min_ell, max_ell + 1):
            
            coef_T_ell = ipow[ell%4] * sqrt(pi* (2 * ell + 1) * (ell + 2) * (ell + 1) * ell * (ell - 1) / 2)
            coef_E_B_ell = ipow[ell%4] * sqrt(pi* (2 * ell + 1) / 2)

            for m in range(-ell, ell + 1):

                abs_m = np.abs(m)
                wigner_d_l_m_2_cur_index = lm_index[ell, abs_m]
                lm_index_cur = ell * (ell+1) + m - ell_range[0] * ell_range[0]

                if m<0:
                    wigner_d_l_m_plus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m]
                    wigner_d_l_m_minus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m]
                else:
                    wigner_d_l_m_plus2 = wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m] 
                    wigner_d_l_m_minus2 =  wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m]  

                xi_lm_plus = oversqrt2 * tilde_xi[i, 0] * wigner_d_l_m_plus2
                xi_lm_minus = oversqrt2 * tilde_xi[i, 1] * wigner_d_l_m_minus2

                for ell_p in range(ell_p_range[0], ell_p_range[1] + 1):
                    
                    if k_amp_cur > np.sqrt(k_max_list[ell]*k_max_list[ell_p]) and k_amp_cur > min_k_amp:
                        continue 

                    coef_T_ell_p = ipow[ell_p%4] * sqrt(pi* (2 * ell_p + 1) * (ell_p + 2) * (ell_p + 1) * ell_p * (ell_p - 1) / 2)
                    coef_E_B_ell_p = ipow[ell_p%4] * sqrt(pi* (2 * ell_p + 1)/2)

                    for m_p in range(-ell_p, ell_p + 1):
                
                        abs_m_p = np.abs(m_p)
                        wigner_d_l_m_2ner_p_cur_index = lm_index[ell_p, abs_m_p]

                        lm_p_index_cur = ell_p * (ell_p+1) + m_p - ell_range[0] * ell_range[0] 

                        if m_p < 0:
                            wigner_d_l_m_p_plus2 = shortle[abs_m_p%2] * wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2ner_p_cur_index] * phase_list_plus[abs_m_p]
                            wigner_d_l_m_p_minus2 = shortle[abs_m_p%2] * wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2ner_p_cur_index] * phase_list_plus[abs_m_p]
                        else:
                            wigner_d_l_m_p_plus2 = wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2ner_p_cur_index] * phase_list_minus[abs_m_p] 
                            wigner_d_l_m_p_minus2 =  wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2ner_p_cur_index] * phase_list_minus[abs_m_p]

                        xi_lm_p_plus = oversqrt2 * tilde_xi[i, 0] * wigner_d_l_m_p_plus2
                        xi_lm_p_minus =  oversqrt2 * tilde_xi[i, 1] * wigner_d_l_m_p_minus2

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

        for ell in range(min_ell, max_ell + 1):
            
            coef_T_ell = ipow[ell%4] * sqrt(pi* (2 * ell + 1) * (ell + 2) * (ell + 1) * ell * (ell - 1) / 2)
            coef_E_B_ell = ipow[ell%4] * sqrt(pi* (2 * ell + 1) / 2)

            for m in range(-ell, ell + 1):

                abs_m = np.abs(m)
                wigner_d_l_m_2_cur_index = lm_index[ell, abs_m]
                lm_index_cur = ell * (ell+1) + m - ell_range[0] * ell_range[0]

                if m<0:
                    wigner_d_l_m_plus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m]
                    wigner_d_l_m_minus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m]
                else:
                    wigner_d_l_m_plus2 = wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m] 
                    wigner_d_l_m_minus2 =  wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m]  

                xi_lm_plus = oversqrt2 * (tilde_xi[i, 0] * wigner_d_l_m_plus2 + tilde_xi[i, 1] * wigner_d_l_m_minus2)
                xi_lm_minus = oversqrt2 * (tilde_xi[i, 0] * wigner_d_l_m_minus2 + tilde_xi[i, 1] * wigner_d_l_m_plus2)

                for ell_p in range(ell_p_range[0], ell_p_range[1] + 1):
                    
                    if k_amp_cur > np.sqrt(k_max_list[ell]*k_max_list[ell_p]) and k_amp_cur > min_k_amp:
                        continue 

                    coef_T_ell_p = ipow[ell_p%4] * sqrt(pi* (2 * ell_p + 1) * (ell_p + 2) * (ell_p + 1) * ell_p * (ell_p - 1) / 2)
                    coef_E_B_ell_p = ipow[ell_p%4] * sqrt(pi* (2 * ell_p + 1)/2)

                    for m_p in range(-ell_p, ell_p + 1):

                        if k_amp_cur > np.sqrt(k_max_list[ell]*k_max_list[ell_p]) and k_amp_cur > min_k_amp:
                            continue 
                
                        abs_m_p = np.abs(m_p)
                        wigner_d_l_m_2_cur_index = lm_index[ell_p, abs_m_p]
                        lm_p_index_cur = ell_p * (ell_p+1) + m_p - ell_range[0] * ell_range[0]

                        if m_p<0:
                            wigner_d_l_m_p_plus2 = shortle[abs_m_p%2] * wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m_p]
                            wigner_d_l_m_p_minus2 = shortle[abs_m_p%2] * wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m_p]
                        else:
                            wigner_d_l_m_p_plus2 = wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p] 
                            wigner_d_l_m_p_minus2 =  wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p]  

                        xi_lm_p_plus = oversqrt2 * (tilde_xi[i, 0] * wigner_d_l_m_p_plus2 + tilde_xi[i, 1] * wigner_d_l_m_p_minus2)
                        xi_lm_p_minus = oversqrt2 * (tilde_xi[i, 0] * wigner_d_l_m_p_minus2 + tilde_xi[i, 1] * wigner_d_l_m_p_plus2)
                        
                        Xi_plus = (xi_lm_plus * conjugate(xi_lm_p_plus) + xi_lm_minus * conjugate(xi_lm_p_minus))
                        Xi_minus = (xi_lm_minus * conjugate(xi_lm_p_minus) - xi_lm_plus * conjugate(xi_lm_p_plus))
            #            Xi_BB = (xi_lm_B_plus * conjugate(xi_lm_p_B_plus) + xi_lm_B_minus * conjugate(xi_lm_p_B_minus))

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

    # 3 Subspace 
    for i in range(subspace3_interval[0], subspace3_interval[1]):
        
        k_amp_cur = k_amp[i]
        wigner_d_l_m_index = theta_unique_index[i]
        k_unique_index_cur = k_amp_unique_index[i]
        phase_list_minus = np.exp(-1j * phi[i] * m_list)
        phase_list_plus  = np.exp(1j * phi[i] * m_list)

        for ell in range(min_ell, max_ell + 1):
            
            coef_T_ell = ipow[ell%4] * sqrt(pi* (2 * ell + 1) * (ell + 2) * (ell + 1) * ell * (ell - 1) / 2)
            coef_E_B_ell = ipow[ell%4] * sqrt(pi* (2 * ell + 1) / 2)

            for m in range(-ell, ell + 1):

                abs_m = np.abs(m)
                wigner_d_l_m_2_cur_index = lm_index[ell, abs_m]
                lm_index_cur = ell * (ell+1) + m - ell_range[0] * ell_range[0]

                if m<0:
                    wigner_d_l_m_plus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m]
                    wigner_d_l_m_minus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m]
                else:
                    wigner_d_l_m_plus2 = wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m] 
                    wigner_d_l_m_minus2 =  wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m]  

                xi_lm_plus = oversqrt2 * (tilde_xi[i, 0] * wigner_d_l_m_plus2 + tilde_xi[i, 1] * wigner_d_l_m_minus2)
                xi_lm_minus = oversqrt2 * (tilde_xi[i, 0] * wigner_d_l_m_minus2 + tilde_xi[i, 1] * wigner_d_l_m_plus2)

                for ell_p in range(ell_p_range[0], ell_p_range[1] + 1):
                    
                    if k_amp_cur > np.sqrt(k_max_list[ell]*k_max_list[ell_p]) and k_amp_cur > min_k_amp:
                        continue 

                    coef_T_ell_p = ipow[ell_p%4] * sqrt(pi* (2 * ell_p + 1) * (ell_p + 2) * (ell_p + 1) * ell_p * (ell_p - 1) / 2)
                    coef_E_B_ell_p = ipow[ell_p%4] * sqrt(pi* (2 * ell_p + 1)/2)

                    for m_p in range(-ell_p, ell_p + 1):

                        if k_amp_cur > np.sqrt(k_max_list[ell]*k_max_list[ell_p]) and k_amp_cur > min_k_amp:
                            continue 
                
                        abs_m_p = np.abs(m_p)
                        wigner_d_l_m_2_cur_index = lm_index[ell_p, abs_m_p]
                        lm_p_index_cur = ell_p * (ell_p+1) + m_p - ell_range[0] * ell_range[0]

                        if m_p<0:
                            wigner_d_l_m_p_plus2 = shortle[abs_m_p%2] * wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m_p]
                            wigner_d_l_m_p_minus2 = shortle[abs_m_p%2] * wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m_p]
                        else:
                            wigner_d_l_m_p_plus2 = wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p] 
                            wigner_d_l_m_p_minus2 =  wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p]  

                        xi_lm_p_plus = oversqrt2 * (tilde_xi[i, 0] * wigner_d_l_m_p_plus2 + tilde_xi[i, 1] * wigner_d_l_m_p_minus2)
                        xi_lm_p_minus = oversqrt2 * (tilde_xi[i, 0] * wigner_d_l_m_p_minus2 + tilde_xi[i, 1] * wigner_d_l_m_p_plus2)

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


    # 4 Subspace 
    for i in range(subspace4_interval[0], subspace4_interval[1]):
        
        k_amp_cur = k_amp[i]
        wigner_d_l_m_index = theta_unique_index[i]
        k_unique_index_cur = k_amp_unique_index[i]
        phase_list_minus = np.exp(-1j * phi[i] * m_list)
        phase_list_plus  = np.exp(1j * phi[i] * m_list)

        for ell in range(min_ell, max_ell + 1):
            
            coef_T_ell = ipow[ell%4] * sqrt(pi* (2 * ell + 1) * (ell + 2) * (ell + 1) * ell * (ell - 1) / 2)
            coef_E_B_ell = ipow[ell%4] * sqrt(pi* (2 * ell + 1) / 2)

            for m in range(-ell, ell + 1):

                abs_m = np.abs(m)
                wigner_d_l_m_2_cur_index = lm_index[ell, abs_m]
                lm_index_cur = ell * (ell+1) + m - ell_range[0] * ell_range[0]

                if m<0:
                    wigner_d_l_m_plus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m]
                    wigner_d_l_m_minus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m]
                else:
                    wigner_d_l_m_plus2 = wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m] 
                    wigner_d_l_m_minus2 =  wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m]  

                xi_lm_plus = oversqrt4 * (tilde_xi[i, 0] * wigner_d_l_m_plus2 + shortle[abs_m%2] * tilde_xi[i, 1] * wigner_d_l_m_minus2 + tilde_xi[i, 2] * wigner_d_l_m_minus2 + shortle[abs_m%2] * tilde_xi[i, 3] * wigner_d_l_m_plus2)
                xi_lm_minus = oversqrt4 * (tilde_xi[i, 0] * wigner_d_l_m_minus2 + shortle[abs_m%2] * tilde_xi[i, 1] * wigner_d_l_m_plus2 + tilde_xi[i, 2] * wigner_d_l_m_plus2 + shortle[abs_m%2] * tilde_xi[i, 3] * wigner_d_l_m_minus2)

                for ell_p in range(l_min, l_max + 1):
                    
                    coef_T_ell_p = ipow[ell_p%4] * sqrt(pi* (2 * ell_p + 1) * (ell_p + 2) * (ell_p + 1) * ell_p * (ell_p - 1) / 2)
                    coef_E_B_ell_p = ipow[ell_p%4] * sqrt(pi* (2 * ell_p + 1)/2)
                    
                    for m_p in range(-ell_p, ell_p + 1):

                        if k_amp_cur > np.sqrt(k_max_list[ell]*k_max_list[ell_p]) and k_amp_cur > min_k_amp:
                            continue 
                
                        abs_m_p = np.abs(m_p)
                        wigner_d_l_m_2_cur_index = lm_index[ell_p, abs_m_p]
                        lm_p_index_cur = ell_p * (ell_p+1) + m_p - ell_range[0] * ell_range[0]

                        if m_p<0:
                            wigner_d_l_m_p_plus2 = shortle[abs_m_p%2] * wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m_p]
                            wigner_d_l_m_p_minus2 = shortle[abs_m_p%2] * wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_plus[abs_m_p]
                        else:
                            wigner_d_l_m_p_plus2 = wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p] 
                            wigner_d_l_m_p_minus2 =  wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p]  

                        xi_lm_p_plus = oversqrt4 * (tilde_xi[i, 0] * wigner_d_l_m_p_plus2 + shortle[abs_m%2] * tilde_xi[i, 1] * wigner_d_l_m_p_minus2 + tilde_xi[i, 2] * wigner_d_l_m_p_minus2 + shortle[abs_m%2] * tilde_xi[i, 3] * wigner_d_l_m_p_plus2)
                        xi_lm_p_minus = oversqrt4 * (tilde_xi[i, 0] * wigner_d_l_m_p_minus2 + shortle[abs_m%2] * tilde_xi[i, 1] * wigner_d_l_m_p_plus2 + tilde_xi[i, 2] * wigner_d_l_m_p_plus2 + shortle[abs_m%2] * tilde_xi[i, 3] * wigner_d_l_m_p_minus2)

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

    C_lmlpmp *= pi*pi / ((2 * V))

    return C_lmlpmp


