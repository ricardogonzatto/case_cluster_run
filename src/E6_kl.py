from .topology_kl import Topology
from .tools_kl import *
import numpy as np
from numpy import pi, sin, cos, exp, sqrt, tan, conjugate, abs, dot
from numba import njit, prange
from numba_progress import ProgressBar

class E6(Topology):
  def __init__(self, param, debug=True, make_run_folder = False):
    L_LSS = 13824.9 * 2
    self.LAx = param['Lx'] * L_LSS
    self.LBy = param['Ly'] * L_LSS
    self.LCz = param['Lz'] * L_LSS
    self.alpha_x = param['alpha_x']
    self.alpha_y = param['alpha_y']
    self.alpha_z = param['alpha_z']

    self.V = self.LAx * self.LBy * self.LCz * 2

    self.x0 = param['x0'] * L_LSS
    if np.linalg.norm(param['x0']) < 1e-6 and np.abs(param['beta'] - 90) < 1e-6  and np.abs(param['alpha'] - 90) < 1e-6:
      self.no_shift = True
    else:
      self.no_shift = False
    print('No shift:', self.no_shift)
    self.l_max = param['l_max']
    self.param = param

    self.root = 'runs/{}_LAx_{}_LBy_{}_LCz_{}_alphax_{}_alphay_{}_alphaz_{}_x_{}_y_{}_z_{}_l_max_{}_accuracy_{}_percent/'.format(
        param['topology'],
        "{:.2f}".format(param['Lx']),
        "{:.2f}".format(param['Ly']),
        "{:.2f}".format(param['Lz']),
        "{:.2f}".format(param['alpha_x']),
        "{:.2f}".format(param['alpha_y']),
        "{:.2f}".format(param['alpha_z']),
        "{:.2f}".format(param['x0'][0]),
        "{:.2f}".format(param['x0'][1]),
        "{:.2f}".format(param['x0'][2]),
        self.l_max,
        int(param['c_l_accuracy']*100)
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
      self.second_eigenmode_index_split,
    )

  def get_list_of_k_phi_theta(self):
    k_amp, phi, theta, tilde_xi, first_eigenmode_index_split, second_eigenmode_index_split = get_list_of_k_phi_theta(
      max(self.k_max_list),
      self.k_list[0],
      self.LAx,
      self.LBy,
      self.LCz,
      self.alpha_x,
      self.alpha_y,
      self.alpha_z,
      self.x0,
    )
    print('Size of tilde xi: {} MB.'.format(round(tilde_xi.size * tilde_xi.itemsize / 1024 / 1024, 2)))
    print('Shape tilde xi:', tilde_xi.shape, '\n')
    self.tilde_xi = tilde_xi
    self.first_eigenmode_index_split = first_eigenmode_index_split
    self.second_eigenmode_index_split = second_eigenmode_index_split
    return k_amp, phi, theta


@njit(parallel = False)
def get_list_of_k_phi_theta(k_max, k_min, L_Ax, L_By, L_Cz, alpha_x, alpha_y, alpha_z, x0):
    # Returns list of k, phi, and theta for this topology

    n_x_max = int(np.ceil(k_max * L_Ax / (pi)))
    n_y_max = int(np.ceil(k_max * L_By / (pi)))
    n_z_max = int(np.ceil(k_max * L_Cz / (pi)))

    k_amp = np.zeros(n_x_max * n_y_max * n_z_max * 8)
    phi = np.zeros(n_x_max * n_y_max * n_z_max * 8)
    theta = np.zeros(n_x_max * n_y_max * n_z_max * 8)

    # First index is i, second is l%2, third is m%2. The fourth index splits the fourth eigenmode into two pieces
    tilde_xi = np.zeros((n_x_max * n_y_max * n_z_max * 8, 2, 2, 2), dtype=np.complex128)

    T_A = np.array([L_Ax,                     (alpha_y + 1/2)*L_By,     (alpha_z - 1/2) * L_Cz])
    T_B = np.array([(alpha_x - 1/2) * L_Ax,   L_By,                     (alpha_z + 1/2)*L_Cz])
    T_C = np.array([(alpha_x + 1/2) * L_Ax,   (alpha_y - 1/2) * L_By,   L_Cz])

    M_A = np.array([
      [1, 0, 0],
      [0, -1, 0],
      [0, 0, -1]
    ], dtype=np.float64)

    M_B = np.array([
      [-1, 0, 0],
      [0, 1, 0],
      [0, 0, -1]
    ], dtype=np.float64)

    M_C = np.array([
      [-1, 0, 0],
      [0, -1, 0],
      [0, 0, 1]
    ], dtype=np.float64)

    cur_index = 0

    # Eigenmode 3
    k_x = 0
    k_y = 0
    for n_z in range(2, n_z_max+1, 2):
      k_z = pi * n_z / L_Cz
      k_xyz = abs(k_z)
      if k_xyz > k_max or k_xyz < k_min:
        continue
      
      k_amp[cur_index] = k_xyz
      cur_phi, cur_theta = 0.0, 0.0
      phi[cur_index] = cur_phi
      theta[cur_index] = cur_theta
      tilde_xi[cur_index, :, :, 0] = exp(- 1j * k_z * (x0[2] + T_A[2]/2)) 
      cur_index += 1
    print(f"Eigenmodes 3: {cur_index}")
    first_eigenmode_index_split = cur_index

    # Eigenmode 1
    k_y = 0
    k_z = 0
    for n_x in range(2, n_x_max+1, 2):
      k_x = pi * n_x / L_Ax
      k_xyz = abs(k_x)
      if k_xyz > k_max or k_xyz < k_min:
        continue
      
      k_amp[cur_index] = k_xyz
      cur_phi, cur_theta = cart2spherical(np.array([k_x, k_y, k_z])/k_xyz)
      phi[cur_index] = cur_phi
      theta[cur_index] = cur_theta
      
      for m_mod_2 in range(2):
        tilde_xi[cur_index, :, m_mod_2, 0] = (exp(- 1j * k_x * x0[0]) 
                                                    + (-1)**m_mod_2 * exp(1j * k_x * (x0[0] + T_B[0])))
      cur_index += 1
    print(f"Eigenmodes 1: {cur_index}")

    # Eigenmode 2
    k_x = 0
    k_z = 0
    for n_y in range(2, n_y_max+1, 2):
      k_y = pi * n_y / L_By
      k_xyz = abs(k_y)
      if k_xyz > k_max or k_xyz < k_min:
        continue
      
      k_amp[cur_index] = k_xyz
      cur_phi, cur_theta = cart2spherical(np.array([k_x, k_y, k_z])/k_xyz)
      phi[cur_index] = cur_phi
      theta[cur_index] = cur_theta
      
      for m_mod_2 in range(2):
        tilde_xi[cur_index, :, m_mod_2, 0] = (exp(- 1j * k_y * x0[1]) 
                                                  + (-1)**m_mod_2 * exp(1j * k_y * (x0[1] + T_C[1])))
      cur_index += 1
    print(f"Eigenmodes 2: {cur_index}")
    second_eigenmode_index_split = cur_index
    tilde_xi /= sqrt(2)

    # Eigenmode 4
    for n_z in range(-n_z_max, n_z_max+1):
      k_z = pi * n_z / L_Cz

      if n_z <= 0:
        n_x_start = 1
        n_x_end = n_x_max
        
        n_y_start = 1
        n_y_end = n_y_max
      else:
        n_x_start = 0
        n_x_end = n_x_max
        
        n_y_start = 0
        n_y_end = n_y_max

      for n_x in range(n_x_start, n_x_end+1):
        k_x = pi * n_x / L_Ax
        
        k_zx_squared = k_z * k_z + k_x * k_x
        if k_zx_squared > k_max * k_max:
          continue

        for n_y in range(n_y_start, n_y_end+1):
          if n_x == 0 and n_y == 0:
            continue

          k_y = pi * n_y / L_By
          
          k_xyz = sqrt(k_zx_squared + k_y * k_y)

          if k_xyz > k_max or k_xyz < k_min :
            continue

          k_vec = np.array([k_x, k_y, k_z])

          k_amp[cur_index] = k_xyz
          cur_phi, cur_theta = cart2spherical(k_vec/k_xyz)
          phi[cur_index] = cur_phi
          theta[cur_index] = cur_theta
         
          for l_mod_2 in range(2):
            for m_mod_2 in range(2):
              tilde_xi[cur_index, l_mod_2, m_mod_2, 0] = ( exp(-1j * np.dot(k_vec, x0) ) 
                                                          + (-1)**m_mod_2 * exp(-1j * dot(k_vec, dot(M_C, x0) - T_C ) ) )/2
              tilde_xi[cur_index, l_mod_2, m_mod_2, 1] = ( (-1)**(l_mod_2 + m_mod_2)  * exp(-1j * dot(k_vec, dot(M_A, x0)) ) * exp(1j * dot(k_vec, T_A)) 
                                                          + (-1)**l_mod_2 * exp(-1j * dot(k_vec, dot(M_B, x0))) * exp(1j * dot(k_vec, T_B)) )/2
          cur_index += 1

    k_amp = k_amp[:cur_index]
    phi = phi[:cur_index]   
    theta = theta[:cur_index]
    tilde_xi = tilde_xi[:cur_index, :, :, :]

    print(cur_index, first_eigenmode_index_split, second_eigenmode_index_split, 'split')
    print('Final num of elements:', k_amp.size, 'Minimum k_amp', np.amin(k_amp), 'n_x_max', n_x_max, 'n_z_max', n_z_max)
    return k_amp, phi, theta, tilde_xi, first_eigenmode_index_split, second_eigenmode_index_split




@njit(nogil=True, parallel=False)
def get_c_lmlpmp(
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
    tilde_xi,
    first_eigenmode_index_split,
    second_eigenmode_index_split,
    ):  

    total_num_l_m = (l_max - l_min + 1)*(l_max + l_min + 2) // 2 # the total number of wigner_d matrices for \lambda =+/-2 
    num_l_m = ell_range[1] * (ell_range[1] + 1) + ell_range[1] + 1 - ell_range[0] * ell_range[0]      # = Sum[2 l+1, {l , l_min, l_max}]
    C_lmlpmp = np.zeros((6, num_l_m, num_l_m), dtype=np.complex128)
    eig_num = k_amp.size
    ipow = np.array([1, 1j, -1, -1j])
    m_list = np.arange(0, l_max+1)
    shortle = np.array([1, -1])
    
    for i in prange(first_eigenmode_index_split):
      k_amp_cur = k_amp[i]
      k_unique_index_cur = k_amp_unique_index[i]
      cur_tilde_xi = tilde_xi[i, :, :, :]    

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
          coeff_E_B_lp = sqrt(pi* (2 * l_p + 1)/2)
          coeff_T_lp = sqrt(pi* (2 * l_p + 1) * (l_p + 2) * (l_p + 1) * l_p * (l_p - 1) / 2)

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
          for m in [-2, 2]:
            lm_index_cur = l * (l+1) + m - ell_range[0] * ell_range[0]

            if m == -2:
              xi_lm_plus = shortle[l%2] * conjugate(cur_tilde_xi[0, 0, 0])
              xi_lm_minus = cur_tilde_xi[0, 0, 0]
            else:
              xi_lm_plus = cur_tilde_xi[0, 0, 0]
              xi_lm_minus = shortle[l%2] * conjugate(cur_tilde_xi[0, 0, 0])

            m_p = 2
            lm_p_index_cur = l_p * (l_p+1) + m_p  - ell_p_range[0]*ell_p_range[0]

            xi_lm_p_plus = cur_tilde_xi[0, 0, 0]        
            xi_lm_p_minus = shortle[l_p%2] * conjugate(cur_tilde_xi[0, 0, 0])
            
            Xi_plus = ( xi_lm_plus* conjugate(xi_lm_p_plus) 
                        + xi_lm_minus * conjugate(xi_lm_p_minus) 
                        )
            Xi_minus = ( xi_lm_minus * conjugate(xi_lm_p_minus)
                          - xi_lm_plus* conjugate(xi_lm_p_plus) 
                        )

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


    for i in prange(first_eigenmode_index_split, second_eigenmode_index_split):
      k_amp_cur = k_amp[i]
      k_unique_index_cur = k_amp_unique_index[i]
      wigner_d_index = theta_unique_index[i]
      phase_list_minus = np.exp(-1j * phi[i] * m_list)
      phase_list_plus  = np.exp(1j * phi[i] * m_list)
      cur_tilde_xi = tilde_xi[i, :, :, :]    

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
          coeff_E_B_lp = sqrt(pi* (2 * l_p + 1)/2)
          coeff_T_lp = sqrt(pi* (2 * l_p + 1) * (l_p + 2) * (l_p + 1) * l_p * (l_p - 1) / 2)

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
          for m in range(-l, l + 1):
            lm_index_cur = l * (l+1) + m - ell_range[0] * ell_range[0]
            abs_m = np.abs(m)
            wigner_cur_index = lm_index[l, abs_m]

            if m<0:
              wigner_D_l_m_plus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_cur_index] * phase_list_plus[abs_m]
              wigner_D_l_m_minus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_index, wigner_cur_index] * phase_list_plus[abs_m]
            else:
              wigner_D_l_m_plus2 = wigner_d_l_m_2[wigner_d_index, wigner_cur_index] * phase_list_minus[abs_m] 
              wigner_D_l_m_minus2 =  wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_cur_index] * phase_list_minus[abs_m]    

            xi_lm_plus = cur_tilde_xi[l%2, m%2, 0] * wigner_D_l_m_plus2
            xi_lm_minus = cur_tilde_xi[l%2, m%2, 0] * wigner_D_l_m_minus2
            
            for m_p in range(0, l_p + 1):
              lm_p_index_cur = l_p * (l_p+1) + m_p  - ell_p_range[0]*ell_p_range[0]
              wigner_p_cur_index = lm_index[l_p, m_p]

              wigner_D_l_m_p_plus2 = wigner_d_l_m_2[wigner_d_index, wigner_p_cur_index] * phase_list_minus[m_p] 
              wigner_D_l_m_p_minus2 =  wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_p_cur_index] * phase_list_minus[m_p]  

              xi_lm_p_plus = cur_tilde_xi[l_p%2, m_p%2, 0] * wigner_D_l_m_p_plus2
              xi_lm_p_minus = cur_tilde_xi[l_p%2, m_p%2, 0] * wigner_D_l_m_p_minus2
                                                              
              
              Xi_plus = ( xi_lm_plus* conjugate(xi_lm_p_plus) 
                          + xi_lm_minus * conjugate(xi_lm_p_minus) 
                          )
              Xi_minus = ( xi_lm_minus * conjugate(xi_lm_p_minus)
                            - xi_lm_plus* conjugate(xi_lm_p_plus) 
                          )

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
    
    for i in prange(second_eigenmode_index_split, eig_num):
      k_amp_cur = k_amp[i]
      k_unique_index_cur = k_amp_unique_index[i]
      wigner_d_index = theta_unique_index[i]
      phase_list_minus = np.exp(-1j * phi[i] * m_list)
      phase_list_plus  = np.exp(1j * phi[i] * m_list)
      cur_tilde_xi = tilde_xi[i, :, :, :]    

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
          coeff_E_B_lp = sqrt(pi* (2 * l_p + 1)/2)
          coeff_T_lp = sqrt(pi* (2 * l_p + 1) * (l_p + 2) * (l_p + 1) * l_p * (l_p - 1) / 2)

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
          for m in range(-l, l + 1):
            lm_index_cur = l * (l+1) + m - ell_range[0] * ell_range[0]
            abs_m = np.abs(m)
            wigner_cur_index = lm_index[l, abs_m]

            if m<0:
              wigner_D_l_m_plus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_cur_index] * phase_list_plus[abs_m]
              wigner_D_l_m_minus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_index, wigner_cur_index] * phase_list_plus[abs_m]
            else:
              wigner_D_l_m_plus2 = wigner_d_l_m_2[wigner_d_index, wigner_cur_index] * phase_list_minus[abs_m] 
              wigner_D_l_m_minus2 =  wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_cur_index] * phase_list_minus[abs_m]    

            xi_lm_plus = (
                          cur_tilde_xi[l%2, m%2, 0] * wigner_D_l_m_plus2
                          + cur_tilde_xi[l%2, m%2, 1] * conjugate(wigner_D_l_m_minus2)
                        )
            xi_lm_minus = (
                          cur_tilde_xi[l%2, m%2, 0] * wigner_D_l_m_minus2
                          + cur_tilde_xi[l%2, m%2, 1] * conjugate(wigner_D_l_m_plus2)
                        )
            
            for m_p in range(0, l_p + 1):
              lm_p_index_cur = l_p * (l_p+1) + m_p  - ell_p_range[0]*ell_p_range[0]
              wigner_p_cur_index = lm_index[l_p, m_p]

              wigner_D_l_m_p_plus2 = wigner_d_l_m_2[wigner_d_index, wigner_p_cur_index] * phase_list_minus[m_p] 
              wigner_D_l_m_p_minus2 =  wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_p_cur_index] * phase_list_minus[m_p]  

              xi_lm_p_plus = (
                              cur_tilde_xi[l_p%2, m_p%2, 0] * wigner_D_l_m_p_plus2
                              + cur_tilde_xi[l_p%2, m_p%2, 1] * conjugate(wigner_D_l_m_p_minus2)
                            )
              xi_lm_p_minus = (
                              cur_tilde_xi[l_p%2, m_p%2, 0] * wigner_D_l_m_p_minus2
                              + cur_tilde_xi[l_p%2, m_p%2, 1] * conjugate(wigner_D_l_m_p_plus2)
                            )
                                                              
              
              Xi_plus = ( xi_lm_plus* conjugate(xi_lm_p_plus) 
                          + xi_lm_minus * conjugate(xi_lm_p_minus) 
                          )
              Xi_minus = ( xi_lm_minus * conjugate(xi_lm_p_minus)
                            - xi_lm_plus* conjugate(xi_lm_p_plus) 
                          )

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

