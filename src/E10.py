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
 
    self.V = 2 * self.LAx * self.LBz * self.LCy
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
    n_y_max = int(np.ceil(k_max * LCy / (2 * pi)))
    n_z_max = int(np.ceil(k_max * LBz / pi))
 
    list_length = n_x_max * n_y_max * n_z_max * 8
    k_amp = np.zeros(list_length)
    phi = np.zeros(list_length)
    theta = np.zeros(list_length)

    tilde_xi = np.zeros((list_length, 4), dtype=np.complex128)

    T_A = np.array([LAx, LAy, 0])
    T_B = np.array([LBx, LCy/2, LBz])

    cur_index = 0

    # 1 Subspace 
    modes_N1 = [(0, 0, n_z) for n_z in range(-n_z_max, n_z_max+1) if n_z != 0 and n_z % 2 == 0]

    for (n_x, n_y, n_z) in modes_N1:

        k_x = (np.pi * n_x) / LAx
        k_y = 2 * (np.pi * n_y) / LCy
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
        tilde_xi[cur_index, 1] = np.exp(1j * k_vec @ (T_A - T_B - x0)) 

        cur_index += 1

    subspace1_interval = [0, cur_index]

    # 2 Subspace 
    modes_N2 = [(n_x, 0, n_z) for n_x in range(-n_x_max, n_x_max+1) if n_x % 2 == 0 and n_x != 0 for n_z in range(-n_z_max, n_z_max+1) if n_z % 2 == 0]

    for (n_x, n_y, n_z) in modes_N2:

        k_x = (np.pi * n_x) / LAx
        k_y = 2 * (np.pi * n_y) / LCy
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
        tilde_xi[cur_index, 1] = np.exp(1j * k_vec @ (T_A + T_B - (M_B @ x0))) 

        cur_index += 1

    subspace2_interval = [subspace1_interval[1], cur_index]

    # 3 Subspace 
    modes_N3 = [(0, n_y, n_z) for n_y in range(-n_y_max, n_y_max+1) if n_y != 0 for n_z in range(-n_z_max, n_z_max+1) if (n_y + n_z) % 2 == 0]

    for (n_x, n_y, n_z) in modes_N3:

        k_x = (np.pi * n_x) / LAx
        k_y = 2 * (np.pi * n_y) / LCy
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
        tilde_xi[cur_index, 1] = np.exp(1j * k_vec @ (T_A + T_B - (M_A @ x0))) 

        cur_index += 1

    subspace3_interval = [subspace2_interval[1], cur_index]

    # 4 Subspace
    modes_N4 = [(n_x, n_y, n_z) for n_x in range(1, n_x_max+1) for n_y in range(1, n_y_max+1) for n_z in range(-n_z_max, n_z_max+1)]

    for (n_x, n_y, n_z) in modes_N4:

        k_x = (np.pi * n_x) / LAx
        k_y = 2 * (np.pi * n_y) / LCy
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
        tilde_xi[cur_index, 1] = np.exp(1j * k_vec @ (T_A - (M_A @ x0))) 
        tilde_xi[cur_index, 2] = np.exp(1j * k_vec @ (T_B - (M_B @ x0))) 
        tilde_xi[cur_index, 3] = np.exp(1j * k_vec @ ((M_B @ T_A) + T_B - (M_A @ M_B @ x0)))

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

                    # coefficients for the lower block (row ell_p, column ell)
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

                            # Lower block: row (ell_p, m_p), column (ell, m). The first field sits on the row,
                            # so e.g. EB needs xi_E(ell_p m_p) * conj(xi_B(ell m)), not conj(Xi_EB) (which is BE)
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

                xi_T_lm_plus = (cur_tilde_xi[0] * wigner_D_l_m_plus2 + shortle[abs_m%2] * cur_tilde_xi[1] * conjugate(wigner_D_l_m_minus2))
                xi_T_lm_minus = (cur_tilde_xi[0] * wigner_D_l_m_minus2 + shortle[abs_m%2] * cur_tilde_xi[1] * conjugate(wigner_D_l_m_plus2))

                xi_E_lm_plus = - xi_T_lm_plus
                xi_E_lm_minus = - xi_T_lm_minus

                xi_B_lm_plus = -(cur_tilde_xi[0] * wigner_D_l_m_plus2 - shortle[abs_m%2] * cur_tilde_xi[1] * conjugate(wigner_D_l_m_minus2))
                xi_B_lm_minus = (cur_tilde_xi[0] * wigner_D_l_m_minus2 - shortle[abs_m%2] * cur_tilde_xi[1] * conjugate(wigner_D_l_m_plus2))

                for ell_p in range(l_start, ell_p_range[1] + 1):

                    coef_T_ell_p = ipow[(ell - ell_p)%4] * sqrt(pi* (2 * ell_p + 1) * (ell_p + 2) * (ell_p + 1) * ell_p * (ell_p - 1) / 2)
                    coef_E_B_ell_p = ipow[(ell - ell_p)%4] * sqrt(pi* (2 * ell_p + 1)/2)

                    # coefficients for the lower block (row ell_p, column ell)
                    coef_T_ell_p_low = ipow[(ell_p - ell)%4] * sqrt(pi* (2 * ell_p + 1) * (ell_p + 2) * (ell_p + 1) * ell_p * (ell_p - 1) / 2)
                    coef_E_B_ell_p_low = ipow[(ell_p - ell)%4] * sqrt(pi* (2 * ell_p + 1)/2)

                    for m_p in range(0, ell_p + 1):
                
                        abs_m_p = np.abs(m_p)
                        wigner_d_l_m_2_cur_index = lm_index[ell_p, abs_m_p]
                        lm_p_index_cur = ell_p * (ell_p+1) + m_p - ell_p_range[0] * ell_p_range[0]

                        wigner_D_l_m_p_plus2 = wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p] 
                        wigner_D_l_m_p_minus2 =  wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p]  

                        xi_T_lm_p_plus = (cur_tilde_xi[0] * wigner_D_l_m_p_plus2 + shortle[abs_m_p%2] * cur_tilde_xi[1] * conjugate(wigner_D_l_m_p_minus2))
                        xi_T_lm_p_minus = (cur_tilde_xi[0] * wigner_D_l_m_p_minus2 + shortle[abs_m_p%2] * cur_tilde_xi[1] * conjugate(wigner_D_l_m_p_plus2))
                            
                        xi_E_lm_p_plus = - xi_T_lm_p_plus
                        xi_E_lm_p_minus = - xi_T_lm_p_minus
                            
                        xi_B_lm_p_plus = -(cur_tilde_xi[0] * wigner_D_l_m_p_plus2 - shortle[abs_m_p%2] * cur_tilde_xi[1] * conjugate(wigner_D_l_m_p_minus2))
                        xi_B_lm_p_minus = (cur_tilde_xi[0] * wigner_D_l_m_p_minus2 - shortle[abs_m_p%2] * cur_tilde_xi[1] * conjugate(wigner_D_l_m_p_plus2))
                                                                                        
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
                            
                            # Lower block: row (ell_p, m_p), column (ell, m), first field on the row
                            Xi_TE_low = (xi_T_lm_p_plus * conjugate(xi_E_lm_plus) + xi_T_lm_p_minus * conjugate(xi_E_lm_minus))
                            Xi_EB_low = (xi_E_lm_p_plus * conjugate(xi_B_lm_plus) + xi_E_lm_p_minus * conjugate(xi_B_lm_minus))
                            Xi_TB_low = (xi_T_lm_p_plus * conjugate(xi_B_lm_plus) + xi_T_lm_p_minus * conjugate(xi_B_lm_minus))
                            # TE correlations
                            C_lmlpmp[3, lm_p_index_cur, lm_index_cur] += coef_T_ell_p_low * coef_E_B_ell * integrand[3, k_unique_index_cur, ell_p, ell] * Xi_TE_low
                            # EB correlations
                            C_lmlpmp[4, lm_p_index_cur, lm_index_cur] += coef_E_B_ell_p_low * coef_E_B_ell * integrand[4, k_unique_index_cur, ell_p, ell] * Xi_EB_low
                            # TB correlations
                            C_lmlpmp[5, lm_p_index_cur, lm_index_cur] += coef_T_ell_p_low * coef_E_B_ell * integrand[5, k_unique_index_cur, ell_p, ell] * Xi_TB_low

    
    # 3 Subspace 
    for i in range(subspace3_interval[0], subspace3_interval[1]):
            
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

                    # coefficients for the lower block (row ell_p, column ell)
                    coef_T_ell_p_low = ipow[(ell_p - ell)%4] * sqrt(pi* (2 * ell_p + 1) * (ell_p + 2) * (ell_p + 1) * ell_p * (ell_p - 1) / 2)
                    coef_E_B_ell_p_low = ipow[(ell_p - ell)%4] * sqrt(pi* (2 * ell_p + 1)/2)

                    for m_p in range(0, ell_p + 1):
                
                        abs_m_p = np.abs(m_p)
                        wigner_d_l_m_2_cur_index = lm_index[ell_p, abs_m_p]
                        lm_p_index_cur = ell_p * (ell_p+1) + m_p - ell_p_range[0] * ell_p_range[0]

                        wigner_D_l_m_p_plus2 = wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p] 
                        wigner_D_l_m_p_minus2 =  wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p]  

                        xi_T_lm_p_plus = (cur_tilde_xi[0] * wigner_D_l_m_p_plus2 + cur_tilde_xi[1] * conjugate(wigner_D_l_m_p_minus2))
                        xi_T_lm_p_minus = (cur_tilde_xi[0] * wigner_D_l_m_p_minus2 + cur_tilde_xi[1] * conjugate(wigner_D_l_m_p_plus2))
                            
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
                            
                            # Lower block: row (ell_p, m_p), column (ell, m), first field on the row
                            Xi_TE_low = (xi_T_lm_p_plus * conjugate(xi_E_lm_plus) + xi_T_lm_p_minus * conjugate(xi_E_lm_minus))
                            Xi_EB_low = (xi_E_lm_p_plus * conjugate(xi_B_lm_plus) + xi_E_lm_p_minus * conjugate(xi_B_lm_minus))
                            Xi_TB_low = (xi_T_lm_p_plus * conjugate(xi_B_lm_plus) + xi_T_lm_p_minus * conjugate(xi_B_lm_minus))
                            # TE correlations
                            C_lmlpmp[3, lm_p_index_cur, lm_index_cur] += coef_T_ell_p_low * coef_E_B_ell * integrand[3, k_unique_index_cur, ell_p, ell] * Xi_TE_low
                            # EB correlations
                            C_lmlpmp[4, lm_p_index_cur, lm_index_cur] += coef_E_B_ell_p_low * coef_E_B_ell * integrand[4, k_unique_index_cur, ell_p, ell] * Xi_EB_low
                            # TB correlations
                            C_lmlpmp[5, lm_p_index_cur, lm_index_cur] += coef_T_ell_p_low * coef_E_B_ell * integrand[5, k_unique_index_cur, ell_p, ell] * Xi_TB_low


   # 4 Subspace 
    for i in range(subspace4_interval[0], subspace4_interval[1]):
            
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

                # (-1)^m D_{-m,mu} = conj(D_{m,-mu}): terms 1,4 keep the helicity, terms 2,3 flip it
                # oversqrt2: A_l/sqrt(4V) here vs the global 1/(2V)
                xi_T_lm_plus = oversqrt2 * ((cur_tilde_xi[0] + shortle[abs_m%2] * cur_tilde_xi[3]) * wigner_D_l_m_plus2
                                            + (cur_tilde_xi[1] + shortle[abs_m%2] * cur_tilde_xi[2]) * conjugate(wigner_D_l_m_minus2))
                xi_T_lm_minus = oversqrt2 * ((cur_tilde_xi[0] + shortle[abs_m%2] * cur_tilde_xi[3]) * wigner_D_l_m_minus2
                                             + (cur_tilde_xi[1] + shortle[abs_m%2] * cur_tilde_xi[2]) * conjugate(wigner_D_l_m_plus2))

                xi_E_lm_plus = - xi_T_lm_plus
                xi_E_lm_minus = - xi_T_lm_minus

                xi_B_lm_plus = - oversqrt2 * ((cur_tilde_xi[0] + shortle[abs_m%2] * cur_tilde_xi[3]) * wigner_D_l_m_plus2
                                              - (cur_tilde_xi[1] + shortle[abs_m%2] * cur_tilde_xi[2]) * conjugate(wigner_D_l_m_minus2))
                xi_B_lm_minus = oversqrt2 * ((cur_tilde_xi[0] + shortle[abs_m%2] * cur_tilde_xi[3]) * wigner_D_l_m_minus2
                                             - (cur_tilde_xi[1] + shortle[abs_m%2] * cur_tilde_xi[2]) * conjugate(wigner_D_l_m_plus2))

                for ell_p in range(l_start, ell_p_range[1] + 1):

                    coef_T_ell_p = ipow[(ell - ell_p)%4] * sqrt(pi* (2 * ell_p + 1) * (ell_p + 2) * (ell_p + 1) * ell_p * (ell_p - 1) / 2)
                    coef_E_B_ell_p = ipow[(ell - ell_p)%4] * sqrt(pi* (2 * ell_p + 1)/2)

                    # coefficients for the lower block (row ell_p, column ell)
                    coef_T_ell_p_low = ipow[(ell_p - ell)%4] * sqrt(pi* (2 * ell_p + 1) * (ell_p + 2) * (ell_p + 1) * ell_p * (ell_p - 1) / 2)
                    coef_E_B_ell_p_low = ipow[(ell_p - ell)%4] * sqrt(pi* (2 * ell_p + 1)/2)

                    for m_p in range(0, ell_p + 1):
                
                        abs_m_p = np.abs(m_p)
                        wigner_d_l_m_2_cur_index = lm_index[ell_p, abs_m_p]
                        lm_p_index_cur = ell_p * (ell_p+1) + m_p - ell_p_range[0] * ell_p_range[0]

                        wigner_D_l_m_p_plus2 = wigner_d_l_m_2[wigner_d_l_m_index, wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p] 
                        wigner_D_l_m_p_minus2 =  wigner_d_l_m_2[wigner_d_l_m_index, total_num_l_m + wigner_d_l_m_2_cur_index] * phase_list_minus[abs_m_p]  

                        xi_T_lm_p_plus = oversqrt2 * ((cur_tilde_xi[0] + shortle[abs_m_p%2] * cur_tilde_xi[3]) * wigner_D_l_m_p_plus2
                                                      + (cur_tilde_xi[1] + shortle[abs_m_p%2] * cur_tilde_xi[2]) * conjugate(wigner_D_l_m_p_minus2))
                        xi_T_lm_p_minus = oversqrt2 * ((cur_tilde_xi[0] + shortle[abs_m_p%2] * cur_tilde_xi[3]) * wigner_D_l_m_p_minus2
                                                       + (cur_tilde_xi[1] + shortle[abs_m_p%2] * cur_tilde_xi[2]) * conjugate(wigner_D_l_m_p_plus2))
                            
                        xi_E_lm_p_plus = - xi_T_lm_p_plus
                        xi_E_lm_p_minus = - xi_T_lm_p_minus
                            
                        xi_B_lm_p_plus = - oversqrt2 * ((cur_tilde_xi[0] + shortle[abs_m_p%2] * cur_tilde_xi[3]) * wigner_D_l_m_p_plus2
                                                        - (cur_tilde_xi[1] + shortle[abs_m_p%2] * cur_tilde_xi[2]) * conjugate(wigner_D_l_m_p_minus2))
                        xi_B_lm_p_minus = oversqrt2 * ((cur_tilde_xi[0] + shortle[abs_m_p%2] * cur_tilde_xi[3]) * wigner_D_l_m_p_minus2
                                                       - (cur_tilde_xi[1] + shortle[abs_m_p%2] * cur_tilde_xi[2]) * conjugate(wigner_D_l_m_p_plus2))
                                                                                        
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
                            
                            # Lower block: row (ell_p, m_p), column (ell, m), first field on the row
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

    C_lmlpmp *= pi*pi / ((2 * V))

    return C_lmlpmp

