import numpy as np
import matplotlib.pyplot as plt
import camb

l_max = 60
top_list = ["E7", "E9"]
directory_path = "./"   # folder with the c_lmlpmp_*.npy files (e.g. ~/Documents/MikelMatrices/)
LAx_slash_Lcircle_list = [0.9, 1.0, 1.1, 1.2, 1.3] 
LAx_list = [0.9, 1.0, 1.1, 1.2, 1.3]
cases = {
    "untilted": dict(L_circle=1.0,  x0="0_0_0"),
    "tilted":   dict(L_circle=0.73, x0="-0.1_0.34_0.0"),
}
LAy_dict = {"E7": "0", "E9": "0.6"}   # LAy as it appears in the file names
fname = "c_lmlpmp_{case}_{topology}_LA{LA}_LAy{LAy}_Ly{Ly}_L2x{L2x}_L2z{L2z}_x={x0}_lmax60.npy"

## To compute the normalized matrix
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


n = num_l_m
for case, p in cases.items():
    for topology in top_list:
        kl_list = np.zeros((len(LAx_list), 2))
        for i, r in enumerate(LAx_list):
            path = directory_path + fname.format(case=case, topology=topology, LA=round(r * p["L_circle"], 3),
                                                 LAy=LAy_dict[topology], Ly=1.2, L2x=0.7, L2z=1, x0=p["x0"])
            C = np.load(path)
            # TEMPORARY CORRECTIONS for the current matrices:
            #  - TE has the opposite sign convention to CAMB (diag(C[3])/powers[:,3] ~ -0.98), so flip it
            #  - TB was computed with ell<->ell' swapped (code before commit c0c17be) and cannot be
            #    recovered from the saved matrix, so it is dropped (KL is then a bit underestimated)
            TE = -C[3]
            TB = np.zeros_like(C[5])
            M = np.block([
                [C[0],          TE,            TB],
                [TE.conj().T,   C[1],          C[4]],
                [TB.conj().T,   C[4].conj().T, C[2]],
            ])
            del C, TE, TB
            # sanity check: TE diagonal must now have the same sign as CAMB
            te_ratio = np.real(np.diag(M[:n, n:2*n]))[-100:].mean() / powers[l_max, 3]
            print(f"{case} {topology} LAx/Lcircle={r}: TE diag ratio at high ell (should be ~ +0.98): {te_ratio:.3f}")
            rel = np.linalg.solve(C_lmlpmp_TEB_E18, M)
            del M
            w = np.linalg.eigvals(rel)
            del rel
            # a valid covariance gives only positive eigenvalues
            if w.real.min() <= 0:
                raise ValueError(f"Non-positive eigenvalue {w.real.min():.3g}: matrix is not a valid covariance")
            kl_list[i] = (np.real(np.sum(np.log(np.abs(w)) + 1/w - 1)) / 2,
                          np.real(np.sum(-np.log(np.abs(w)) + w - 1)) / 2)
            print(f"{case} {topology} LAx/Lcircle={r}: kl_P_Q, kl_Q_P = {kl_list[i]}")
        np.save(f"TEB_{case}_kl_list_{topology}_lmax_{l_max}_correctionET.npy", kl_list)