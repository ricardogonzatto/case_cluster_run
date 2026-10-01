import numpy as np
from numpy import pi, sin, cos, exp, sqrt, tan, conjugate
from numba import njit, prange
from numba_progress import ProgressBar



@njit(nogil=True, parallel=False)
def E2_E3_get_full_c_lmlpmp(
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
  tilde_xi_delta_m,
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
    cur_tilde_xi = tilde_xi[i, :]
    
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
          xi_m = cur_tilde_xi[m % tilde_xi_delta_m]

          if m<0:
            wigner_D_l_m_plus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_cur_index] * phase_list_plus[abs_m]
            wigner_D_l_m_minus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_index, wigner_cur_index] * phase_list_plus[abs_m]
          else:
            wigner_D_l_m_plus2 = wigner_d_l_m_2[wigner_d_index, wigner_cur_index] * phase_list_minus[abs_m] 
            wigner_D_l_m_minus2 =  wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_cur_index] * phase_list_minus[abs_m]

          
          # Only do m-mp = 0 mod 2  
          for m_p in range(0, l_p + 1):
            conj_xi_m_p = conjugate(cur_tilde_xi[m_p % tilde_xi_delta_m])
            lm_p_index_cur = l_p * (l_p+1) + m_p  - ell_range[0] * ell_range[0]
            wigner_p_cur_index = lm_index[l_p, m_p]

            wigner_D_l_m_p_plus2 = wigner_d_l_m_2[wigner_d_index, wigner_p_cur_index] * phase_list_minus[m_p] 
            wigner_D_l_m_p_minus2 =  wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_p_cur_index] * phase_list_minus[m_p] 

            Xi_plus = (
                    wigner_D_l_m_plus2* np.conjugate(wigner_D_l_m_p_plus2) 
                    + wigner_D_l_m_minus2 *np.conjugate(wigner_D_l_m_p_minus2)
                    ) * xi_m * conj_xi_m_p
            Xi_minus = (
                    wigner_D_l_m_minus2 *np.conjugate(wigner_D_l_m_p_minus2) 
                    - wigner_D_l_m_plus2* np.conjugate(wigner_D_l_m_p_plus2)
                    ) * xi_m * conj_xi_m_p
          
            
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




@njit(nogil=True, parallel=False)
def E2_E3_get_c_lmlpmp_diag(
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
    tilde_xi,
    tilde_xi_delta_m,
    progress
    ):

    total_num_l_m = (l_max - l_min + 1)*(l_max + l_min + 2) // 2 # the total number of wigner_d matrices for \lambda =+/-2 
    num_l = l_max + 1 - l_min
    C_lmlpmp = np.zeros((6, num_l), dtype=np.complex128)  
    m_list = np.arange(0, l_max+1)
    min_k = min(k_amp)
    shortle = np.array([1, -1])
    
    for i in range(min_index, max_index):
      k_amp_cur = k_amp[i]
      k_unique_index_cur = k_amp_unique_index[i]
      wigner_d_index = theta_unique_index[i]
      phase_list_minus = np.exp(-1j * phi[i] * m_list)
      phase_list_plus  = np.exp(1j * phi[i] * m_list)
      cur_tilde_xi = tilde_xi[i, :]

      for l in range(l_min, l_max + 1):
        if k_amp_cur > k_max_list[l] or k_amp_cur < min_k :
          continue

        l_index_cur = l - l_min # l - 2
        coeff_EE_BB_ell = pi / 2
        coeff_TT_ell = pi * (l + 2) * (l + 1) * l * (l - 1) / 2 
        coeff_TE_ell = pi * sqrt((l + 2) * (l + 1) * l * (l - 1) / 4)

        for m in range(-l, l + 1):
            abs_m = np.abs(m)
            wigner_cur_index = lm_index[l, abs_m]

            if m<0:
              wigner_D_l_m_plus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_cur_index] * phase_list_plus[abs_m]
              wigner_D_l_m_minus2 = shortle[abs_m%2] * wigner_d_l_m_2[wigner_d_index, wigner_cur_index] * phase_list_plus[abs_m]
            else:
              wigner_D_l_m_plus2 = wigner_d_l_m_2[wigner_d_index, wigner_cur_index] * phase_list_minus[abs_m] 
              wigner_D_l_m_minus2 =  wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_cur_index] * phase_list_minus[abs_m]
            # Then tilde xi
            xi_m = cur_tilde_xi[m % tilde_xi_delta_m]
              

            Xi_plus = (wigner_D_l_m_plus2* conjugate(wigner_D_l_m_plus2) 
                        + wigner_D_l_m_minus2 * conjugate(wigner_D_l_m_minus2))* xi_m * conjugate(xi_m)
            Xi_minus = (wigner_D_l_m_minus2 * conjugate(wigner_D_l_m_minus2) 
                        - wigner_D_l_m_plus2* conjugate(wigner_D_l_m_plus2))* xi_m * conjugate(xi_m)

            C_lmlpmp[0, l_index_cur] += coeff_TT_ell* integrand[0, k_unique_index_cur, l_index_cur] * Xi_plus
            C_lmlpmp[1, l_index_cur] += coeff_EE_BB_ell * integrand[1, k_unique_index_cur, l_index_cur] * Xi_plus
            C_lmlpmp[2, l_index_cur] += coeff_EE_BB_ell* integrand[2, k_unique_index_cur, l_index_cur] * Xi_plus
            C_lmlpmp[3, l_index_cur] += coeff_TE_ell * integrand[3, k_unique_index_cur, l_index_cur] * Xi_plus
            C_lmlpmp[4, l_index_cur] += coeff_EE_BB_ell * integrand[4, k_unique_index_cur, l_index_cur] * Xi_minus
            C_lmlpmp[5, l_index_cur] += coeff_TE_ell * integrand[5, k_unique_index_cur, l_index_cur] * Xi_minus     

      if progress != None: progress.update(1)
    

    C_lmlpmp *= pi*pi / (2 * V)
    print('This chunk has been finished')

    return C_lmlpmp
