import os
import numpy as np
import camb

# KL divergence using ONLY the auto-correlation blocks TT, EE and BB.
# The cross blocks (TE, EB, TB) of the current matrices are not used: TE/EB have the wrong
# E sign and TB still has the old ell<->ell' bug. TT, EE and BB are not affected by either.
#
#   1) KL of TT only
#   2) KL of TT + EE + BB, with no cross-correlations.
#      Without cross blocks T, E and B are independent, so KL = KL_TT + KL_EE + KL_BB.
#      IMPORTANT: the covering-space reference must also have no TE, otherwise the missing
#      TE in the topology matrix alone would produce a large fake KL.
#
# Each block is read with mmap and treated separately (~0.2 GB per block at l_max = 60).

l_max = 60          # can be lowered (e.g. 30) to cut the matrices at a smaller ell_max
top_list = ["E7", "E9"]
directory_path = os.path.expanduser("~/Documents/MikelMatrices/")
LAx_list = [0.9, 1.0, 1.1, 1.2, 1.3]   # LAx / L_circle
cases = {
    "untilted": dict(L_circle=1.0,  x0="0_0_0"),
    "tilted":   dict(L_circle=0.73, x0="-0.1_0.34_0.0"),
}
fname = "c_lmlpmp_{case}_{topology}_LA{LA:.2f}_LAy{LAy}_Ly{Ly}_L2x{L2x}_L2z{L2z}_x={x0}_lmax60.npy"

## Covering-space (E18) power spectra, same CAMB settings as src/topology.py
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0.1)
pars.set_for_lmax(60)
pars.WantTensors = True
pars.set_accuracy(AccuracyBoost=3, lAccuracyBoost=3, lSampleBoost=100)
pars.Accuracy.IntkAccuracyBoost = 3
pars.Accuracy.SourcekAccuracyBoost = 3
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, raw_cl=True, CMB_unit='muK')['tensor']   # columns: TT, EE, BB, TE

num_l_m = (l_max + 1)**2 - 4                     # number of (ell, m) with 2 <= ell <= l_max
ell_of_index = np.concatenate([[ell] * (2 * ell + 1) for ell in range(2, l_max + 1)])


def kl_one_block(C_block, cl):
    """KL between the topology block C_block and the diagonal covering-space block diag(C_ell).
    Returns (D_KL(p||q), D_KL(q||p)) and the smallest eigenvalue."""
    ref = cl[ell_of_index]                         # C_ell for each (ell, m)
    A = C_block / np.sqrt(np.outer(ref, ref))      # whitened: eigenvalues of C_ref^-1 C_topo
    A = (A + A.conj().T) / 2                       # remove round-off non-Hermiticity
    w = np.linalg.eigvalsh(A)
    if w.min() <= 0:
        raise ValueError(f"Non-positive eigenvalue {w.min():.3g}: block is not a valid covariance")
    kl_pq = np.sum(np.log(w) + 1 / w - 1) / 2
    kl_qp = np.sum(-np.log(w) + w - 1) / 2
    return np.array([kl_pq, kl_qp]), w.min()


names = ["TT", "EE", "BB"]
for case, p in cases.items():
    for topology in top_list:
        kl_TT = np.zeros((len(LAx_list), 2))
        kl_diag = np.zeros((len(LAx_list), 2))
        for i, r in enumerate(LAx_list):
            path = directory_path + fname.format(case=case, topology=topology, LA=r * p["L_circle"],
                                                 LAy="0.0", Ly=1.4, L2x=0.7, L2z=1, x0=p["x0"])
            C = np.load(path, mmap_mode='r')
            kl_blocks = {}
            for k, name in enumerate(names):
                block = np.array(C[k][:num_l_m, :num_l_m])
                kl_blocks[name], wmin = kl_one_block(block, powers[:, k])
                del block
                print(f"{case} {topology} LAx/Lcircle={r}: {name}  KL = {kl_blocks[name]}  (min eig {wmin:.3f})")
            del C
            kl_TT[i] = kl_blocks["TT"]
            kl_diag[i] = kl_blocks["TT"] + kl_blocks["EE"] + kl_blocks["BB"]
            print(f"{case} {topology} LAx/Lcircle={r}: TT only = {kl_TT[i]},  TT+EE+BB = {kl_diag[i]}\n")
        np.save(f"TT_{case}_kl_list_{topology}_lmax_{l_max}.npy", kl_TT)
        np.save(f"TT_EE_BB_{case}_kl_list_{topology}_lmax_{l_max}.npy", kl_diag)
