import numpy as np
import matplotlib.pyplot as plt
import camb

l_max = 70

pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0.1)
pars.set_for_lmax(l_max)
pars.WantTensors = True

# More accurate transfer functions. Takes longer time to run
pars.set_accuracy(AccuracyBoost = 3, lAccuracyBoost = 3, lSampleBoost = 100)
pars.Accuracy.IntkAccuracyBoost = 3
pars.Accuracy.SourcekAccuracyBoost = 3

# Get the CAMB functions and save them
data = camb.get_transfer_functions(pars)
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, raw_cl=True, CMB_unit='muK')['tensor']
transfer_tensor = data.get_cmb_transfer_data(tp='tensor')

# To get C_\ell in units of umK, we multiply by 1e6 (K to micro K) and the temperature of the CMB in K
transfer_data = np.array(transfer_tensor.delta_p_l_k) * 1e6 * 2.7255
print('Shape of transfer function from CAMB:', transfer_data.shape)

# CAMB gives the transfer data for a set of k and ell. Store these values
# and later we will use interpolation for other k/ell values.
#k is in Mpc^{-1}
k_list = np.array(transfer_tensor.q)
ell_list = np.array(transfer_tensor.L)

kl_list = np.zeros((3,2), dtype = np.float64)
## KL Divergence
def calculate_kl_divergence(matrix):
  w, _ = np.linalg.eig(matrix)
  kl_P_assuming_Q = 0
  kl_Q_assuming_P = 0

  for eig in w:
    kl_P_assuming_Q += (np.log(np.abs(eig)) + 1/eig - 1)/2
    kl_Q_assuming_P += (-np.log(np.abs(eig)) + eig - 1)/2

  return np.real(kl_P_assuming_Q), np.real(kl_Q_assuming_P)



num_l_m = l_max * (l_max + 1) + l_max + 1 - 4     # = Sum[2 l+1, {l , l_min, l_max}], l_min = 2
C_lmlpmp_TEB_E18 = np.zeros((3 * num_l_m, 3 * num_l_m), dtype=np.complex128) 

# TT correlations
for ell in range(2, l_max + 1):
    for m in range(-ell, ell + 1):
        lm_p_index = ell * (ell+1) + m  - 4
        C_lmlpmp_TEB_E18[lm_p_index, lm_p_index] = powers[ell, 0]
# TE correlations
for ell in range(2, l_max + 1):
    for m in range(-ell, ell + 1):
        lm_p_index = ell * (ell+1) + m  - 4
        C_lmlpmp_TEB_E18[lm_p_index + num_l_m, lm_p_index] = powers[ell, 3]
# ET correlations
for ell in range(2, l_max + 1):
    for m in range(-ell, ell + 1):
        lm_p_index = ell * (ell+1) + m  - 4
        C_lmlpmp_TEB_E18[lm_p_index, lm_p_index + num_l_m] = np.conjugate(powers[ell, 3])
# EE correlations
for ell in range(2, l_max + 1):
    for m in range(-ell, ell + 1):
        lm_p_index = ell * (ell+1) + m  - 4
        C_lmlpmp_TEB_E18[lm_p_index + num_l_m, lm_p_index + num_l_m] = powers[ell, 1]
# BB correlations
for ell in range(2, l_max + 1):
    for m in range(-ell, ell + 1):
        lm_p_index = ell * (ell+1) + m  - 4
        C_lmlpmp_TEB_E18[lm_p_index + 2 * num_l_m, lm_p_index + 2 * num_l_m] = powers[ell, 2]

C_lmlpmp_TEB_E_i = np.zeros((3 * num_l_m, 3 * num_l_m), dtype=np.complex128) 

### $L_{A} = 1.4$, $L_{B} = 0.9 L_{circle}$ l_max = 70 $x_0 = (0.35, -0.2, 0)$

topology = 'E4'

directory_path = "runs/E1_Lx_1.40_Ly_1.40_Lz_0.90_beta_90_alpha_90_l_max_70_accuracy_99_percent/"

C_lmlpmp = np.load(directory_path+'full_corr_matrix_1_l_2_70_lp_2_70.npy')

# TT correlations
C_lmlpmp_TEB_E_i[:num_l_m, :num_l_m] = C_lmlpmp[0]
# TE correlations
C_lmlpmp_TEB_E_i[num_l_m:2 * num_l_m, :num_l_m] = C_lmlpmp[3].conj().T
# TB correlations
C_lmlpmp_TEB_E_i[2* num_l_m:3 * num_l_m, :num_l_m] = C_lmlpmp[5].conj().T
# ET correlations
C_lmlpmp_TEB_E_i[:num_l_m, num_l_m:2 * num_l_m] = C_lmlpmp[3]
# EE correlations
C_lmlpmp_TEB_E_i[num_l_m:2 * num_l_m, num_l_m:2 * num_l_m] = C_lmlpmp[1]
# EB correlations
C_lmlpmp_TEB_E_i[2 * num_l_m:3 * num_l_m, num_l_m:2 * num_l_m] = C_lmlpmp[4].conj().T
# BT correlations
C_lmlpmp_TEB_E_i[:num_l_m, 2*num_l_m:3 * num_l_m] = C_lmlpmp[5]
# BE correlations
C_lmlpmp_TEB_E_i[num_l_m:2 * num_l_m, 2*num_l_m:3 * num_l_m] = C_lmlpmp[4]
# BB correlations
C_lmlpmp_TEB_E_i[2*num_l_m:3 * num_l_m, 2*num_l_m:3 * num_l_m] = C_lmlpmp[2]

C_lmlpmp_TEB_E_i = np.linalg.inv(C_lmlpmp_TEB_E18) @ C_lmlpmp_TEB_E_i 
kl_list[0] = calculate_kl_divergence(C_lmlpmp_TEB_E_i)
print(f"kl_P_Q, kl_Q_P = {kl_list[0]}")


### $L_{A} = 1.4$, $L_{B} = L_{circle}$ l_max = 70 $x_0 = (0.35, -0.2, 0)$

directory_path = "runs/E1_Lx_1.40_Ly_1.40_Lz_1.00_beta_90_alpha_90_l_max_70_accuracy_99_percent/"

C_lmlpmp = np.load(directory_path+'full_corr_matrix_1_l_2_70_lp_2_70.npy')

# TT correlations
C_lmlpmp_TEB_E_i[:num_l_m, :num_l_m] = C_lmlpmp[0]
# TE correlations
C_lmlpmp_TEB_E_i[num_l_m:2 * num_l_m, :num_l_m] = C_lmlpmp[3]
# TB correlations
C_lmlpmp_TEB_E_i[2* num_l_m:3 * num_l_m, :num_l_m] = C_lmlpmp[5].conj().T
# ET correlations
C_lmlpmp_TEB_E_i[:num_l_m, num_l_m:2 * num_l_m] = C_lmlpmp[3].conj().T
# EE correlations
C_lmlpmp_TEB_E_i[num_l_m:2 * num_l_m, num_l_m:2 * num_l_m] = C_lmlpmp[1]
# EB correlations
C_lmlpmp_TEB_E_i[2 * num_l_m:3 * num_l_m, num_l_m:2 * num_l_m] = C_lmlpmp[4]
# BT correlations
C_lmlpmp_TEB_E_i[:num_l_m, 2*num_l_m:3 * num_l_m] = C_lmlpmp[5]
# BE correlations
C_lmlpmp_TEB_E_i[num_l_m:2 * num_l_m, 2*num_l_m:3 * num_l_m] = C_lmlpmp[4].conj().T
# BB correlations
C_lmlpmp_TEB_E_i[2*num_l_m:3 * num_l_m, 2*num_l_m:3 * num_l_m] = C_lmlpmp[2]

C_lmlpmp_TEB_E_i = np.linalg.inv(C_lmlpmp_TEB_E18) @ C_lmlpmp_TEB_E_i 
kl_list[1] = calculate_kl_divergence(C_lmlpmp_TEB_E_i)
print(f"kl_P_Q, kl_Q_P = {kl_list[1]}")


### $L_{A} = 1.4$, $L_{B} = 1.1 L_{circle}$ l_max = 70 $x_0 = (0.35, -0.2, 0)

directory_path = "runs/E1_Lx_1.40_Ly_1.40_Lz_1.05_beta_90_alpha_90_l_max_70_accuracy_99_percent/"

C_lmlpmp = np.load(directory_path+'full_corr_matrix_1_l_2_70_lp_2_70.npy')

# TT correlations
C_lmlpmp_TEB_E_i[:num_l_m, :num_l_m] = C_lmlpmp[0]
# TE correlations
C_lmlpmp_TEB_E_i[num_l_m:2 * num_l_m, :num_l_m] = C_lmlpmp[3]
# TB correlations
C_lmlpmp_TEB_E_i[2* num_l_m:3 * num_l_m, :num_l_m] = C_lmlpmp[5].conj().T
# ET correlations
C_lmlpmp_TEB_E_i[:num_l_m, num_l_m:2 * num_l_m] = C_lmlpmp[3].conj().T
# EE correlations
C_lmlpmp_TEB_E_i[num_l_m:2 * num_l_m, num_l_m:2 * num_l_m] = C_lmlpmp[1]
# EB correlations
C_lmlpmp_TEB_E_i[2 * num_l_m:3 * num_l_m, num_l_m:2 * num_l_m] = C_lmlpmp[4]
# BT correlations
C_lmlpmp_TEB_E_i[:num_l_m, 2*num_l_m:3 * num_l_m] = C_lmlpmp[5]
# BE correlations
C_lmlpmp_TEB_E_i[num_l_m:2 * num_l_m, 2*num_l_m:3 * num_l_m] = C_lmlpmp[4].conj().T
# BB correlations
C_lmlpmp_TEB_E_i[2*num_l_m:3 * num_l_m, 2*num_l_m:3 * num_l_m] = C_lmlpmp[2]

C_lmlpmp_TEB_E_i = np.linalg.inv(C_lmlpmp_TEB_E18) @ C_lmlpmp_TEB_E_i 
kl_list[2] = calculate_kl_divergence(C_lmlpmp_TEB_E_i)
print(f"kl_P_Q, kl_Q_P = {kl_list[2]}")

np.save('TEB_kl_list_E1_lmax_70.npy', kl_list)