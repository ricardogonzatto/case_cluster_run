import numpy as np
import camb

l_max = 80

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

num_l_m = l_max * (l_max + 1) + l_max + 1 - 4     # = Sum[2 l+1, {l , l_min, l_max}], l_min = 2
c_lmlpmp_TEB = np.zeros((3 * num_l_m, 3 * num_l_m), dtype=np.complex128) 

# TT correlations
for ell in range(2, l_max + 1):
    for m in range(-ell, ell + 1):
        lm_p_index = ell * (ell+1) + m  - 4
        c_lmlpmp_TEB[lm_p_index, lm_p_index] = powers[ell, 0]
# TE correlations
for ell in range(2, l_max + 1):
    for m in range(-ell, ell + 1):
        lm_p_index = ell * (ell+1) + m  - 4
        c_lmlpmp_TEB[lm_p_index + num_l_m, lm_p_index] = powers[ell, 3]
# ET correlations
for ell in range(2, l_max + 1):
    for m in range(-ell, ell + 1):
        lm_p_index = ell * (ell+1) + m  - 4
        c_lmlpmp_TEB[lm_p_index, lm_p_index + num_l_m] = np.conjugate(powers[ell, 3])
# EE correlations
for ell in range(2, l_max + 1):
    for m in range(-ell, ell + 1):
        lm_p_index = ell * (ell+1) + m  - 4
        c_lmlpmp_TEB[lm_p_index + num_l_m, lm_p_index + num_l_m] = powers[ell, 1]
# BB correlations
for ell in range(2, l_max + 1):
    for m in range(-ell, ell + 1):
        lm_p_index = ell * (ell+1) + m  - 4
        c_lmlpmp_TEB[lm_p_index + 2 * num_l_m, lm_p_index + 2 * num_l_m] = powers[ell, 2]

eigvals, eigvecs = np.linalg.eig(c_lmlpmp_TEB)
c_diag_E18_inv = np.zeros_like(c_lmlpmp_TEB)
for i in range(c_lmlpmp_TEB.shape[0]):
    c_diag_E18_inv[i, i] = 1 / eigvals[i]
np.save('unitary_matrix_lmax_{}.npy'.format(l_max), eigvecs)
np.save('diagonal_inv_TEB_E18_lmax_{}.npy'.format(l_max), c_diag_E18_inv)