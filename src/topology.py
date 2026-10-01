from locale import normalize
import camb
import os
import numpy as np
from numpy import pi, sqrt, conjugate
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import multiprocessing
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time
import pickle
from .tools import *
from sys import getsizeof
from itertools import product
from random import sample
import multiprocessing
from multiprocessing import Manager, shared_memory
from concurrent.futures import ProcessPoolExecutor
from memory_profiler import profile

class Topology:
    def __init__(self, param, debug=True, make_run_folder = True):
        print('Running', param)
        self.param = param
        self.topology = param['topology']
        self.l_max = param['l_max']
        self.l_min = param['l_min']
        self.c_l_accuracy = param['c_l_accuracy']
        self.node = param['node'] - 1 # To make it compatible because arrays start with index 0
        self.C_l_type = 6

        self.fig_name = 'l_max_{}'.format(self.l_max)
        self.debug = debug
        self.make_run_folder = make_run_folder
        if make_run_folder and os.path.exists(self.root) == False:
            print('Making run folder:', self.root)
            os.makedirs(self.root)
            os.makedirs(self.root+'figs/')
            os.makedirs(self.root+'realizations/') 

        time_start = time.time()
        self.do_pre_processing()
        print('Time to pre-process with l_max={} and accuracy={}:'.format(self.l_max, self.c_l_accuracy), time.time()-time_start, 'seconds')

    def do_pre_processing(self):
        # This function does all the preprocessing

        #Set up a new set of parameters for CAMB
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
        pars.InitPower.set_params(As=2e-9, ns=0.965, r=0.1)
        pars.set_for_lmax(self.l_max)
        pars.WantTensors = True

        # More accurate transfer functions. Takes longer time to run
        pars.set_accuracy(AccuracyBoost = 3, lAccuracyBoost = 3, lSampleBoost = 100)
        pars.Accuracy.IntkAccuracyBoost = 3
        pars.Accuracy.SourcekAccuracyBoost = 3

        # Get the CAMB functions and save them
        data = camb.get_transfer_functions(pars)
        results = camb.get_results(pars)
        self.powers = results.get_cmb_power_spectra(pars, raw_cl=True, CMB_unit='muK')['tensor']
        transfer_tensor = data.get_cmb_transfer_data(tp='tensor')

        # To get C_\ell in units of umK, we multiply by 1e6 (K to micro K) and the temperature of the CMB in K
        transfer_data = np.array(transfer_tensor.delta_p_l_k) * 1e6 * 2.7255
        print('Shape of transfer function from CAMB:', transfer_data.shape)

        # CAMB gives the transfer data for a set of k and ell. Store these values
        # and later we will use interpolation for other k/ell values.
        #k is in Mpc^{-1}
        self.k_list = np.array(transfer_tensor.q)
        self.ell_list = np.array(transfer_tensor.L)

        # Calculates the transfer functions, primordial power spectrum,
        # and spherical harmonics and stores them with as little memory as possible

        l_max = self.l_max
        l_min = self.l_min

        # Get the transfer functions and put them into lists of interpolate objects.
        # We can therefore evaluate the transfer function for all possibel |k| later
        assert(self.ell_list[0] == 2 and self.ell_list[l_max-2] == l_max)
        self.transfer_T_interpolate_k_l_list ={}
        self.transfer_E_interpolate_k_l_list = {}
        self.transfer_B_interpolate_k_l_list = {}
        for l in range(2, l_max+1):
            self.transfer_T_interpolate_k_l_list[l] = scipy.interpolate.interp1d(self.k_list, transfer_data[0, l-2, :], kind='cubic') 
            self.transfer_E_interpolate_k_l_list[l] = scipy.interpolate.interp1d(self.k_list, transfer_data[1, l-2, :], kind='cubic') 
            self.transfer_B_interpolate_k_l_list[l] = scipy.interpolate.interp1d(self.k_list, transfer_data[2, l-2, :], kind='cubic') 
        

        # We have a list of k_max as a function of ell. We need to make sure this is large enough
        #############################################################################
        self.get_kmax_as_function_of_ell(pars.tensor_power)
        
        # We find all allowed |k|, phi, theta and put them in big lists
        # The function get_list_of_k_phi_theta is specific to each topology
        start_time = time.time()
        k_amp, phi, theta = self.get_list_of_k_phi_theta()
        print('Time to get list of k, phi, theta:', time.time()-start_time, 'seconds')


        #Splitting the eigenmodes to several nodes!
        num_nodes = 1
        num_eigenmode = k_amp.size
        index_thread_split = np.arange(0, num_eigenmode, int(np.ceil(num_eigenmode/num_nodes)))
        index_thread_split = np.append(index_thread_split, num_eigenmode)
        min_index = index_thread_split[self.node] # minimum index in the k_amp list not unique 
        max_index = index_thread_split[self.node + 1]


        self.k_amp = k_amp[min_index: max_index]
        self.phi = phi[min_index: max_index]
        self.theta = theta[min_index: max_index]

        # |k|, phi and theta often repeats themselves. We do not want to recalculate spherical harmonics
        # twice or more so we store a list of all unique thetas. Same for |k| to quickly find transfer functions later
        start_time = time.time()
        self.k_amp_unique, self.k_amp_unique_index, self.theta_unique, self.theta_unique_index = get_k_theta_index_repeat(self.k_amp, self.theta)
        print('Time to get unique k and theta:', time.time()-start_time, 'seconds')

        # Get P(k) / k^3 for all unique |k| values
        self.tensor_pk_k3 = pars.tensor_power(self.k_amp_unique) / self.k_amp_unique**3

        # Get the transfer function for all unique |k| values
        self.transfer_T_delta_kl = self.get_transfer_functions_multi(self.transfer_T_interpolate_k_l_list)
        self.transfer_E_delta_kl = self.get_transfer_functions_multi(self.transfer_E_interpolate_k_l_list)
        self.transfer_B_delta_kl = self.get_transfer_functions_multi(self.transfer_B_interpolate_k_l_list)

        # Store the Healpy ordering of a_lm
        self.lm_2_index = np.zeros((l_max +1, l_max +1), dtype=np.int32)
        for l in range(l_min, l_max+1):
            for m in range(l+1):
                self.lm_2_index[l, m] = get_lm_idx(l_min, l, m)
        
        self.get_wigner_d_multiprocessing()

        print('\n**************')
        print('Done with all preprocessing')
        print('**************\n')

        # Get the spherical harmonics without the phase (without exp(i*m*phi))
        # We store these in an array of size (all lm, all n_z, all n*x^2+n_y^2)
        # This is because theta can be found from nz and nx^2+ny^2, and we do not
        # care about phi since we can add the phase in the sum
        #start_time = time.time()

        

   
    def calculate_c_lmlpmp(self, plot_param={}):
        # Calculaget_c_lmlpmp, the off-diagonal and on-diagonal power spectrum

        l_max = self.l_max
        print('\nCalculating covariance matrix') 

        
        integrand_TT = do_integrand_pre_processing(self.k_amp_unique, self.tensor_pk_k3, self.transfer_T_delta_kl, self.transfer_T_delta_kl, self.l_max)
        integrand_EE = do_integrand_pre_processing(self.k_amp_unique, self.tensor_pk_k3, self.transfer_E_delta_kl, self.transfer_E_delta_kl, self.l_max)
        integrand_BB = do_integrand_pre_processing(self.k_amp_unique, self.tensor_pk_k3, self.transfer_B_delta_kl, self.transfer_B_delta_kl, self.l_max)
        integrand_TE = do_integrand_pre_processing(self.k_amp_unique, self.tensor_pk_k3, self.transfer_T_delta_kl, self.transfer_E_delta_kl, self.l_max)
        integrand_BE = do_integrand_pre_processing(self.k_amp_unique, self.tensor_pk_k3, self.transfer_E_delta_kl, self.transfer_B_delta_kl, self.l_max)
        integrand_BT = do_integrand_pre_processing(self.k_amp_unique, self.tensor_pk_k3, self.transfer_B_delta_kl, self.transfer_T_delta_kl, self.l_max)
        self.integrand = np.stack((integrand_TT, integrand_EE, integrand_BB, integrand_TE, integrand_BE, integrand_BT))

        print('Size of integrand: {} MB.'.format(round(self.integrand.size * self.integrand.itemsize / 1024 / 1024, 2)))
        l_min = plot_param['l_ranges'][0, 0]
        l_max = plot_param['l_ranges'][0, 1]
        lp_min = plot_param['lp_ranges'][0, 0]
        lp_max = plot_param['lp_ranges'][0, 1]
        # Make sure the l_ranges do not overlap!
        ell_range = np.array(plot_param['l_ranges'][0, :])
        ell_p_range = np.array(plot_param['lp_ranges'][0, :])
        self.ell_range = ell_range
        self.ell_p_range = ell_p_range

        start_time = time.time()
        C_lmlpmp = self.get_c_lmlpmp_multiprocessing(
            ell_range = ell_range,
            ell_p_range = ell_p_range)
        total_time_seconds = time.time() - start_time
        hours, remainder = divmod(total_time_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Time to get Correlation functions: {int(hours)}:{int(minutes)}:{int(seconds)}")

        np.save(self.root+'full_corr_matrix_{}_l_{}_{}_lp_{}_{}.npy'.format(self.node + 1,
            l_min, l_max,
            lp_min, lp_max,), C_lmlpmp)
        
        # normalized_clmlpmp = normalize_c_lmlpmp(
        #                 C_lmlpmp[4], 
        #                 self.powers[:, 1],
        #                 self.powers[:, 2],
        #                 l_min=l_min, 
        #                 l_max=l_max, 
        #                 lp_min = lp_min,
        #                 lp_max = lp_max,
        #                 cl_accuracy = self.c_l_accuracy)
        # np.save(self.root+'norm_corr_matrix_EB_l_{}_{}_lp_{}_{}.npy'.format(
        # l_min, l_max,
        # lp_min, lp_max,), normalized_clmlpmp)
        return C_lmlpmp

                


    
    def do_cov_sub_plot(self, ax, normalize, ax_index, C_order, ell_range, ell_p_range):
        
        l_min = ell_range[0]
        l_max = ell_range[1]
        lp_min = ell_p_range[0]
        lp_max = ell_p_range[1]
        C_order = np.where(np.abs(C_order) < 1e-12, 1e-12, np.abs(C_order))

        ell_to_s_map = np.array([l * (l+1) - l - l_min**2  for l in range(l_min, l_max+1)])
        ellp_to_s_map = np.array([l * (l+1) - l - lp_min**2  for l in range(lp_min, lp_max+1)])

        axim = ax.imshow(C_order.T, cmap='inferno', norm=LogNorm(), origin='lower',  interpolation = 'nearest')


        
        if l_max-l_min > 20:
            jump = np.array([5, 10, 15, 20])-2
            ax.set_xticks(ell_to_s_map[jump]- 0.5)
            ax.set_xticklabels(np.arange(l_min, l_max+1)[jump])
        else:
            ax.set_xticks(ell_to_s_map-0.5)
            ax.set_xticklabels(np.arange(l_min, l_max+1))

        if lp_max-lp_min > 20:
            jump = np.array([5, 10, 15, 20])-2
            ax.set_yticks(ellp_to_s_map[jump]- 0.5)
            ax.set_yticklabels(np.arange(lp_min, lp_max+1)[jump])
        else:
            ax.set_yticks(ellp_to_s_map-0.5)
            ax.set_yticklabels(np.arange(lp_min, lp_max+1))
        
        if lp_max > 50 or l_max > 50:
            ax.set_title(str(ax_index+5), weight='bold', fontsize='20')
        else:
            ax.set_title(str(ax_index+1), weight='bold', fontsize='20')
        
        if ax_index == 3 or ax_index == 2:
            ax.set_xlabel(r"$\ell $")
        if ax_index == 0 or ax_index == 2: 
            ax.set_xlabel(r"$\ell $") 
            ax.set_ylabel(r"$\ell'$")
        if normalize:
            axim.set_clim(1e-8, 1e0)
        else:
            axim.set_clim(1e-14, 1e2)

        # ax.text(0.000, 0.0001, 'Additional information', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)

        ax.set_title('{} Lx={} Ly={} Lz={} beta ={} alpha = {}'.format(
            self.param['topology'],
            "{:.2f}".format(self.param['Lx']),
            "{:.2f}".format(self.param['Ly']),
            "{:.2f}".format(self.param['Lz']),
            int(self.param['beta']),
            int(self.param['alpha'])))
        
        return axim

     
    def get_kmax_as_function_of_ell(self, tensor_power):
        # Get k_max as a function of multipole ell
        # We use cumulative trapezoid to find the k_value where we reach
        # the wanted accuracy

        l_max = self.l_max

        # Do the integration up to k=0.08. This should be fine for ell=<250 and accuracy<=0.99
        print('\nFinding k_max as a function of ell')
        self.k_max_list = np.zeros(l_max+1)

        for l in tqdm(range(2, l_max+1)):            
            k_list = np.linspace(self.k_list[0], self.k_list[-1], 200000)
            integrand_TT = 1/4 *pi * (l+1)*(l+2)*l*(l-1)* tensor_power(k_list) * self.transfer_T_interpolate_k_l_list[l](k_list)**2 / k_list
            integrand_EE = 1/4 *pi * tensor_power(k_list) * self.transfer_E_interpolate_k_l_list[l](k_list)*self.transfer_E_interpolate_k_l_list[l](k_list) / k_list
            integrand_BB = 1/4 *pi * tensor_power(k_list) * self.transfer_B_interpolate_k_l_list[l](k_list)*self.transfer_B_interpolate_k_l_list[l](k_list) / k_list
           
            cumulative_c_l_TT_ratio = scipy.integrate.cumulative_trapezoid(y=integrand_TT, x=k_list) / self.powers[l, 0]
            cumulative_c_l_EE_ratio = scipy.integrate.cumulative_trapezoid(y=integrand_EE, x=k_list) / self.powers[l, 1]
            cumulative_c_l_BB_ratio = scipy.integrate.cumulative_trapezoid(y=integrand_BB, x=k_list) / self.powers[l, 2]
            
            index_closest_to_accuracy_target_TT = (np.abs(cumulative_c_l_TT_ratio -  self.c_l_accuracy)).argmin()
            index_closest_to_accuracy_target_EE = (np.abs(cumulative_c_l_EE_ratio -  self.c_l_accuracy)).argmin()
            index_closest_to_accuracy_target_BB = (np.abs(cumulative_c_l_BB_ratio -  self.c_l_accuracy)).argmin()
            
            self.k_max_list[l] = max( k_list[index_closest_to_accuracy_target_TT], k_list[index_closest_to_accuracy_target_EE], k_list[index_closest_to_accuracy_target_BB])
            # self.k_max_list[l] = self.k_list[-3000]

        if self.make_run_folder: np.save(self.root+'k_max_list.npy', self.k_max_list)

        print('Done. k_max for ell_max =', self.k_max_list[l_max])
     
    
    def get_transfer_functions_multi(self, transfer_interpolate_k_l_list):
        # Get all the transfer functions
        
        num_k_amp_unique = self.k_amp_unique.size
        transfer_delta_kl = np.zeros((num_k_amp_unique, self.l_max+1))

        ncpus = multiprocessing.cpu_count()
        os.environ['OMP_NUM_THREADS'] = '1'
        pool = multiprocessing.Pool(processes=ncpus)
        print('\nGetting transfer functions')
        
        args = zip(np.arange(num_k_amp_unique), repeat(self.l_max), repeat(self.k_amp_unique), repeat(transfer_interpolate_k_l_list))
        
        with multiprocessing.Pool(processes=ncpus) as pool:
            transfer_delta_kl = np.array(pool.starmap(transfer_parallel, tqdm(args, total=num_k_amp_unique)))
            print('Size of transfer function: {} MB.'.format(round(getsizeof(transfer_delta_kl) / 1024 / 1024, 2)), '\n')
        
        pool.close() 

        return transfer_delta_kl 
    
    
    def get_wigner_d_multiprocessing(self):
        # Get all the spherical harmonics without phase (phi=0)
        # We use multiprocessing to make this fast
        num_l_m_2 = (self.l_max - self.l_min + 1)*(self.l_max + self.l_min + 2) // 2 # the total number of wigner_d matrices for \lambda =+/-2 

        # We only find d^l_{m,mp}(\theta) for unique theta elements. We don't want to recalculate d^l_{m,mp} unnecessarily
        unique_theta_length = self.theta_unique.size
        ncpus = multiprocessing.cpu_count()
        os.environ['OMP_NUM_THREADS'] = '1'
        pool = multiprocessing.Pool(processes=ncpus)
        print('\nGetting Wigner_d matrices!')
        args = zip(np.arange(unique_theta_length), repeat(self.l_max), repeat(self.l_min), 
                   repeat(self.theta_unique), repeat(self.lm_2_index), repeat(num_l_m_2))
        with multiprocessing.Pool(processes=ncpus) as pool:
            self.wigner_d_l_m_2 = np.array(pool.starmap(get_wigner_d, tqdm(args, total=unique_theta_length)), dtype=np.float64)
            print('The Wigner_d metrices array is', round(getsizeof(self.wigner_d_l_m_2) / 1024 / 1024,2), 'MB \n')
            pool.close()

    def get_wigner_d_multiprocessing_revised(self):
        # Get all the spherical harmonics without phase (phi=0)
        # We use multiprocessing to make this fast
        # We only find d^l_{m,mp}(\theta) for unique theta elements. We don't want to recalculate d^l_{m,mp} unnecessarily
        unique_theta_length = self.theta_unique.size
        ncpus = multiprocessing.cpu_count()
        os.environ['OMP_NUM_THREADS'] = '1'
        pool = multiprocessing.Pool(processes=ncpus)
        print('\nGetting Wigner_d matrices!')
        args = zip(np.arange(unique_theta_length), repeat(self.l_max), repeat(self.l_min), repeat(self.theta_unique))
        with multiprocessing.Pool(processes=ncpus) as pool:
            results = pool.starmap(get_wigner_d_revised, tqdm(args, total=unique_theta_length))
            pool.close()
        # result.shape = (unique_theta_length, 2) => (arg_size, 2) list of arguments
        #                                         => (2, arg_size) list of the wigner_d's
        
        self.arg_list = [result[0] for _, result in enumerate(results)]
        self.wigner_list = [result[1] for _, result in enumerate(results)]
        print(f"The Wigner_d metrices array is: {np.round(getsizeof(self.wigner_list) / 1024 / 1024,2)} MB\n")

    def get_c_lmlpmp_multiprocessing(self, ell_range, ell_p_range):
        l_max = self.l_max
        l_min = self.l_min
        num_l_m = ell_range[1] * (ell_range[1] + 1) + ell_range[1] + 1 - ell_range[0] * ell_range[0]      # = Sum[2 l+1, {l , l_min, l_max}]
        c_lmlpmp = np.zeros((6, num_l_m, num_l_m), dtype=np.complex128) 
        
        ncpus = multiprocessing.cpu_count()
        semi_jumps = int(np.ceil(np.sum(2 * np.arange(l_min, l_max + 1) + 1)**2/2 / ncpus))
        index_thread_split = np.array([l_min], dtype= np.int16)
        i = 0
        for ell in range(index_thread_split[i] + 1, l_max + 1):
            if np.sum(2 * np.arange(index_thread_split[i], ell) + 1)* np.sum(2 * np.arange(index_thread_split[i], l_max+1) + 1)>= semi_jumps:
                index_thread_split = np.append(index_thread_split, ell)
                i += 1
                continue
        size = index_thread_split.size
        index_thread_split = np.append(index_thread_split, l_max + 1)
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for i in range(size):
            # Spawn a process for each cpu that goes through parts of the summation each
            min_ell = index_thread_split[i] # minimum index in the k_amp list not unique 
            max_ell = index_thread_split[i+1] - 1
            
            args = (
                i,
                return_dict,
                min_ell,
                max_ell,
                self.V,
                self.k_amp, 
                self.phi, 
                self.theta_unique_index,
                self.k_amp_unique_index,
                self.k_max_list,
                l_max,
                l_min,
                self.lm_2_index,
                self.wigner_d_l_m_2,
                self.integrand,
                ell_range,
                ell_p_range
            )

            p = multiprocessing.Process(target=self.get_c_lmlpmp_per_process_multi, args=args)
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join() #Waits for each process to complete before moving on.
            proc.close()
            
        # The final c_lmlpmp that is the sum of contribution from each process  
        for array in return_dict.values():
            c_lmlpmp += array 
        for ell in range(2, l_max+1):
            for ell_p in range(ell + 1, l_max+1):
                for m in range(-ell, ell+1):
                    for m_p in range(-ell_p, ell_p + 1):
                        lm_p_index = ell_p * (ell_p+1) + m_p - l_min * l_min
                        lm_index = ell * (ell+1) + m - l_min * l_min
                        c_lmlpmp[:3, lm_p_index, lm_index] = np.conjugate(c_lmlpmp[:3, lm_index, lm_p_index])

        return c_lmlpmp
    
"""     @profile
    def get_c_lmlpmp_multiprocessing(self, ell_range, ell_p_range):
        l_max = self.l_max
        l_min = self.l_min

        ncpus = multiprocessing.cpu_count()

        semi_jumps = int(np.ceil(np.sum(2 * np.arange(l_min, l_max + 1) + 1)**2/2 / ncpus))
        index_thread_split = np.array([l_min], dtype= np.int16)
        i = 0
        for ell in range(index_thread_split[i] + 1, l_max + 1):
            if np.sum(2 * np.arange(index_thread_split[i], ell) + 1)* np.sum(2 * np.arange(index_thread_split[i], l_max+1) + 1)>= semi_jumps:
                index_thread_split = np.append(index_thread_split, ell)
                i += 1
                continue
        size = index_thread_split.size
        index_thread_split = np.append(index_thread_split, l_max + 1)

        num_l_m = ell_range[1] * (ell_range[1] + 1) + ell_range[1] + 1 - ell_range[0] * ell_range[0]     # = Sum[2 l+1, {l , l_min, l_max}]
        C_lmlpmp = np.zeros((6, num_l_m, num_l_m), dtype=np.complex128)  
        futures = []

    
        #turning wignerD to shared_memory_wignerD
        wigner_d_l_m_2_shm = shared_memory.SharedMemory(create=True, size=self.wigner_d_l_m_2.nbytes)
        shared_wigner_d_l_m_2 = np.ndarray(self.wigner_d_l_m_2.shape, dtype=self.wigner_d_l_m_2.dtype, buffer=wigner_d_l_m_2_shm.buf)
        np.copyto(shared_wigner_d_l_m_2, self.wigner_d_l_m_2)
        #turning integrand to shared_memory_integrand
        integrand_shm = shared_memory.SharedMemory(create=True, size=self.integrand.nbytes)
        shared_integrand = np.ndarray(self.integrand.shape, dtype=self.integrand.dtype, buffer=integrand_shm.buf)
        np.copyto(shared_integrand, self.integrand)

        # k_amp_shm = shared_memory.SharedMemory(create=True, size=self.k_amp.nbytes)
        # k_amp_shared = np.ndarray(self.k_amp.shape, dtype=self.k_amp.dtype, buffer=k_amp_shm.buf)
        # np.copyto(k_amp_shared, self.k_amp)

        # phi_shm = shared_memory.SharedMemory(create=True, size=self.phi.nbytes)
        # phi_shared = np.ndarray(self.phi.shape, dtype=self.phi.dtype, buffer=phi_shm.buf)
        # np.copyto(phi_shared, self.phi)

        # theta_unique_index_shm = shared_memory.SharedMemory(create=True, size=self.theta_unique_index.nbytes)
        # theta_unique_index_shared = np.ndarray(self.theta_unique_index.shape, dtype=self.theta_unique_index.dtype, buffer=theta_unique_index_shm.buf)
        # np.copyto(theta_unique_index_shared, self.theta_unique_index)

        # k_amp_unique_index_shm = shared_memory.SharedMemory(create=True, size=self.k_amp_unique_index.nbytes)
        # k_amp_unique_index_shared = np.ndarray(self.k_amp_unique_index.shape, dtype=self.k_amp_unique_index.dtype, buffer=k_amp_unique_index_shm.buf)
        # np.copyto(k_amp_unique_index_shared, self.k_amp_unique_index)

        with ProcessPoolExecutor(max_workers= ncpus) as executor:
        # Prepare the arguments for each process
            for i in range(size):
                min_ell = index_thread_split[i] # minimum index in the k_amp list not unique 
                max_ell = index_thread_split[i+1] - 1
                
                # args = (
                # min_ell,
                # max_ell,
                # self.V,
                # k_amp_shared, 
                # phi_shared, 
                # theta_unique_index_shared,
                # k_amp_unique_index_shared,
                # self.k_max_list,
                # l_max,
                # l_min,
                # self.lm_2_index,
                # shared_wigner_d_l_m_2, # np.asarray(self.wigner_d_l_m_2, dtype=np.float64),
                # shared_integrand,
                # ell_p_range,
                # )

                args = (
                        min_ell,
                        max_ell,
                        self.V,
                        self.k_amp, 
                        self.phi, 
                        self.theta_unique_index,
                        self.k_amp_unique_index,
                        self.k_max_list,
                        l_max,
                        l_min,
                        self.lm_2_index,
                        shared_wigner_d_l_m_2, 
                        wigner_d_l_m_2_shm,
                        shared_integrand,
                        integrand_shm,
                        ell_p_range
                        )

                futures.append(executor.submit(get_full_c_lmlpmp, *args))
            
            # Collect the results as they complete
            for i, future in enumerate(futures):
                min_ell = index_thread_split[i] # minimum index in the k_amp list not unique 
                max_ell = index_thread_split[i+1] - 1

                min_ell_index = min_ell*min_ell - 4
                max_ell_index = max_ell * max_ell + 2 * max_ell - 3

                result1, result2 = future.result()  # Assuming future.result() returns a tuple or list
                C_lmlpmp[:, min_ell_index: max_ell_index, min_ell_index:] += result1
                C_lmlpmp[3:, min_ell_index:, min_ell_index: max_ell_index] += result2

        wigner_d_l_m_2_shm.close()
        wigner_d_l_m_2_shm.unlink()
        integrand_shm.close()
        integrand_shm.unlink()  

        for ell in range(2, l_max+1):
            for ell_p in range(ell + 1, l_max+1):
                for m in range(-ell, ell+1):
                    for m_p in range(-ell_p, ell_p + 1):
                        lm_p_index = ell_p * (ell_p+1) + m_p - l_min * l_min
                        lm_index = ell * (ell+1) + m - l_min * l_min
                        C_lmlpmp[:3, lm_p_index, lm_index] = np.conjugate(C_lmlpmp[:3, lm_index, lm_p_index])

        return C_lmlpmp """
    

    
    



@njit(nogil=True, parallel = False)
def get_full_c_lmlpmp(
    min_ell,
    max_ell,
    V,
    k_amp, 
    phi, 
    theta_unique_index,
    k_amp_unique_index,
    k_max_list, 
    l_max,
    lm_index,
    wigner_d_l_m_2,
    integrand,
    ell_p_range,
    ):

    total_num_l_m = int((l_max - min_ell + 1)*(l_max + min_ell + 2)/2) # the total number of wigner_d matrices for \lambda =+/-2 
    num_l_m = max_ell * (max_ell + 1) + max_ell + 1 - min_ell * min_ell     # = Sum[2 l+1, {l , l_min, l_max}]
    num_l_m_p = ell_p_range[1] * (ell_p_range[1] + 1) + ell_p_range[1] + 1 - min_ell * min_ell 
    C_lmlpmp = np.zeros((6, num_l_m, num_l_m_p), dtype=np.complex128) 
    C_lmlpmp_cross = np.zeros((3, num_l_m_p, num_l_m), dtype=np.complex128)  
 

    eig_num = k_amp.size
    ipow = np.array([1, 1j, -1, -1j])
    m_list = np.arange(0, l_max+1)
    shortle = np.array([1, -1])

    min_k_amp = np.min(k_amp)
    print("It starts!")
    for i in prange(eig_num):
        k_amp_cur = k_amp[i]
        k_unique_index_cur = k_amp_unique_index[i]
        wigner_d_index = theta_unique_index[i] # wigner_d for especific theta[i]
        phase_list_minus = np.exp(-1j * phi[i] * m_list)
        phase_list_plus  = np.exp(1j * phi[i] * m_list)
        
        for l in range(min_ell, max_ell + 1):

            coeff_E_B_ell = sqrt(pi* (2 * l + 1) / 2)
            coeff_T_ell = sqrt(pi* (2 * l + 1) * (l + 2) * (l + 1) * l * (l - 1) / 2 ) 
            coeff_E_B_ell_pow2 = coeff_E_B_ell * coeff_E_B_ell
            coeff_T_ell_pow2 = coeff_T_ell * coeff_T_ell
            coeff_E_B_T_ell = coeff_E_B_ell * coeff_T_ell

            lm_index_cur_zero = l * (l+1) - min_ell * min_ell # for m = 0 
            if ell_p_range[0] >= l:
                l_p_start = ell_p_range[0]
            else:
                l_p_start = l

            for l_p in range(l_p_start , ell_p_range[1]+1):
                if k_amp_cur > np.sqrt(k_max_list[l]*k_max_list[l_p]) and k_amp_cur > min_k_amp:
                    continue

                coeff_E_B_lp = sqrt(pi* (2 * l_p + 1) / 2)
                coeff_T_lp = sqrt(pi* (2 * l_p + 1) * (l_p + 2) * (l_p + 1) * l_p * (l_p - 1) / 2 ) 

                lm_p_index_cur_zero = l_p * (l_p+1) - min_ell * min_ell # for m_p = 0 

                wigner_cur_index = lm_index[l, 0]
                wigner_p_cur_index = lm_index[l_p, 0]

                wigner_D_l_m_plus2 = wigner_d_l_m_2[wigner_d_index, wigner_cur_index]
                wigner_D_l_m_minus2 =  wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_cur_index] 
                wigner_D_l_m_p_plus2 = wigner_d_l_m_2[wigner_d_index, wigner_p_cur_index] 
                wigner_D_l_m_p_minus2 =  wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_p_cur_index]

                # for m = m' = 0
                if l == l_p:
                    Xi_plus = wigner_D_l_m_plus2* np.conjugate(wigner_D_l_m_plus2) + wigner_D_l_m_minus2 *np.conjugate(wigner_D_l_m_minus2)
                    Xi_minus = wigner_D_l_m_minus2 *np.conjugate(wigner_D_l_m_minus2) - wigner_D_l_m_plus2* np.conjugate(wigner_D_l_m_plus2)

                    C_lmlpmp[0, lm_index_cur_zero, lm_index_cur_zero] += coeff_T_ell_pow2 * integrand[0, k_unique_index_cur, l-min_ell, l-min_ell] * Xi_plus
                    C_lmlpmp[1, lm_index_cur_zero, lm_index_cur_zero] += coeff_E_B_ell_pow2 * integrand[1, k_unique_index_cur, l-min_ell, l-min_ell] * Xi_plus
                    C_lmlpmp[2, lm_index_cur_zero, lm_index_cur_zero] += coeff_E_B_ell_pow2 * integrand[2, k_unique_index_cur, l-min_ell, l-min_ell] * Xi_plus
                    C_lmlpmp[3, lm_index_cur_zero, lm_index_cur_zero] += coeff_E_B_T_ell * integrand[3, k_unique_index_cur, l-min_ell, l-min_ell] * Xi_plus
                    C_lmlpmp[4, lm_index_cur_zero, lm_index_cur_zero] += coeff_E_B_ell_pow2* integrand[4, k_unique_index_cur, l-min_ell, l-min_ell] * Xi_minus
                    C_lmlpmp[5, lm_index_cur_zero, lm_index_cur_zero] += coeff_E_B_T_ell * integrand[5, k_unique_index_cur, l-min_ell, l-min_ell]* Xi_minus 
                else:

                    Xi_plus = wigner_D_l_m_plus2* conjugate(wigner_D_l_m_p_plus2) + wigner_D_l_m_minus2 *conjugate(wigner_D_l_m_p_minus2)
                    Xi_minus = wigner_D_l_m_minus2 *conjugate(wigner_D_l_m_p_minus2) - wigner_D_l_m_plus2* conjugate(wigner_D_l_m_p_plus2)

                    C_lmlpmp[0, lm_index_cur_zero, lm_p_index_cur_zero] += (coeff_T_ell * coeff_T_lp * integrand[0, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                    * ipow[(l-l_p)%4] * Xi_plus)


                    C_lmlpmp[1, lm_index_cur_zero, lm_p_index_cur_zero] += (coeff_E_B_ell * coeff_E_B_lp * integrand[1, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                    * ipow[(l-l_p)%4] * Xi_plus)


                    C_lmlpmp[2, lm_index_cur_zero, lm_p_index_cur_zero] += (coeff_E_B_ell * coeff_E_B_lp * integrand[2, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                    * ipow[(l-l_p)%4] * Xi_plus)
              
                    

                    C_lmlpmp[3, lm_index_cur_zero, lm_p_index_cur_zero] += (coeff_T_ell * coeff_E_B_lp * integrand[3, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                    * ipow[(l-l_p)%4] * Xi_plus)
                    C_lmlpmp_cross[0, lm_p_index_cur_zero, lm_index_cur_zero] += (coeff_T_lp * coeff_E_B_ell * integrand[3, k_unique_index_cur, l_p-min_ell, l-min_ell] 
                                                                    * ipow[(l_p-l)%4] * conjugate(Xi_plus))

                    C_lmlpmp[4, lm_index_cur_zero, lm_p_index_cur_zero] += (coeff_E_B_ell * coeff_E_B_lp * integrand[4, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                    * ipow[(l-l_p)%4] * Xi_minus)
                    C_lmlpmp_cross[1, lm_p_index_cur_zero, lm_index_cur_zero] += (coeff_E_B_ell * coeff_E_B_lp * integrand[4, k_unique_index_cur, l_p-min_ell, l-min_ell] 
                                                                    * ipow[(l_p-l)%4] * conjugate(Xi_minus))
                    
                    C_lmlpmp[5, lm_index_cur_zero, lm_p_index_cur_zero] += (coeff_T_ell * coeff_E_B_lp * integrand[5, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                    * ipow[(l-l_p)%4] * Xi_minus) 
                    C_lmlpmp_cross[2, lm_p_index_cur_zero, lm_index_cur_zero] += (coeff_E_B_ell * coeff_T_lp * integrand[5, k_unique_index_cur, l_p-min_ell, l-min_ell] 
                                                                    * ipow[(l_p-l)%4] * conjugate(Xi_minus)) 
                    

                # for m = 0 and m_p != 0
                for m_p in range(1, l_p + 1):
                    if m_p%2 ==1:
                        continue
                    lm_p_index_cur_pos_m_p = l_p * (l_p+1) + m_p - min_ell*min_ell # for positive m
                    lm_p_index_cur_neg_m_p = l_p * (l_p+1) - m_p - min_ell*min_ell # for negative m
                    
                    wigner_p_cur_index_pos_m_p = lm_index[l_p, m_p]

                    wigner_D_l_pos_m_p_plus2 = ( wigner_d_l_m_2[wigner_d_index, wigner_p_cur_index_pos_m_p] 
                                                * phase_list_minus[m_p])
                    wigner_D_l_pos_m_p_minus2 = ( wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_p_cur_index_pos_m_p] 
                                                * phase_list_minus[m_p] )

                    Xi_plus_zero_m_pos_m_p = ( wigner_D_l_m_plus2* conjugate(wigner_D_l_pos_m_p_plus2) 
                                            + wigner_D_l_m_minus2 * conjugate(wigner_D_l_pos_m_p_minus2) )
                  
                    Xi_minus_zero_m_pos_m_p = ( wigner_D_l_m_minus2 * conjugate(wigner_D_l_pos_m_p_minus2)
                                                - wigner_D_l_m_plus2* conjugate(wigner_D_l_pos_m_p_plus2) )
                    
                    # TT
                    C_lmlpmp[0, lm_index_cur_zero, lm_p_index_cur_pos_m_p] += (coeff_T_ell * coeff_T_lp * 
                                                                                integrand[0, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_plus_zero_m_pos_m_p)
                    C_lmlpmp[0, lm_index_cur_zero, lm_p_index_cur_neg_m_p] = ( shortle[m_p %2]*
                                                                                    conjugate(C_lmlpmp[0, lm_index_cur_zero, lm_p_index_cur_pos_m_p]) )
                    
                    # EE
                    C_lmlpmp[1, lm_index_cur_zero, lm_p_index_cur_pos_m_p] += (coeff_E_B_ell * coeff_E_B_lp * 
                                                                                integrand[1, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_plus_zero_m_pos_m_p)
                    C_lmlpmp[1, lm_index_cur_zero, lm_p_index_cur_neg_m_p] = ( shortle[m_p %2]*
                                                                                    conjugate(C_lmlpmp[1, lm_index_cur_zero, lm_p_index_cur_pos_m_p]) )
                    
                    # BB
                    C_lmlpmp[2, lm_index_cur_zero, lm_p_index_cur_pos_m_p] += (coeff_E_B_ell * coeff_E_B_lp * 
                                                                                integrand[2, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_plus_zero_m_pos_m_p)
                    C_lmlpmp[2, lm_index_cur_zero, lm_p_index_cur_neg_m_p] = ( shortle[m_p %2]*
                                                                                    conjugate(C_lmlpmp[2, lm_index_cur_zero, lm_p_index_cur_pos_m_p]) )
                    
                    # TE
                    C_lmlpmp[3, lm_index_cur_zero, lm_p_index_cur_pos_m_p] += (coeff_T_ell * coeff_E_B_lp * 
                                                                                integrand[3, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_plus_zero_m_pos_m_p)
                    C_lmlpmp[3, lm_index_cur_zero, lm_p_index_cur_neg_m_p] = ( shortle[m_p %2]*
                                                                                    conjugate(C_lmlpmp[3, lm_index_cur_zero, lm_p_index_cur_pos_m_p]) )
                    
                    #EB
                    C_lmlpmp[4, lm_index_cur_zero, lm_p_index_cur_pos_m_p] += (coeff_E_B_ell * coeff_E_B_lp * 
                                                                                integrand[4, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_minus_zero_m_pos_m_p)
                    C_lmlpmp[4, lm_index_cur_zero, lm_p_index_cur_neg_m_p] = ( shortle[m_p %2]*
                                                                                    conjugate(C_lmlpmp[4, lm_index_cur_zero, lm_p_index_cur_pos_m_p]) )
                    
                    # TB
                    C_lmlpmp[5, lm_index_cur_zero, lm_p_index_cur_pos_m_p] += (coeff_T_ell * coeff_E_B_lp * 
                                                                                integrand[5, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_minus_zero_m_pos_m_p)
                    C_lmlpmp[5, lm_index_cur_zero, lm_p_index_cur_neg_m_p] = ( shortle[m_p %2]*
                                                                                    conjugate(C_lmlpmp[5, lm_index_cur_zero, lm_p_index_cur_pos_m_p]) )
                    
                    if l != l_p:
                        # TE correlations
                        
                        C_lmlpmp_cross[0, lm_p_index_cur_pos_m_p, lm_index_cur_zero] += (coeff_T_lp * coeff_E_B_ell * 
                                                                                integrand[3, k_unique_index_cur, l_p-min_ell, l-min_ell] 
                                                                                * ipow[(l_p-l)%4] * conjugate(Xi_plus_zero_m_pos_m_p))

                        C_lmlpmp_cross[0, lm_p_index_cur_neg_m_p, lm_index_cur_zero] = ( shortle[m_p %2]*
                                                                                    conjugate(C_lmlpmp_cross[0, lm_p_index_cur_pos_m_p, lm_index_cur_zero]) )

                        # EB correlations

                        C_lmlpmp_cross[1, lm_p_index_cur_pos_m_p, lm_index_cur_zero] += (coeff_E_B_lp * coeff_E_B_ell * 
                                                                                integrand[4, k_unique_index_cur, l_p-min_ell, l-min_ell] 
                                                                                * ipow[(l_p-l)%4] * conjugate(Xi_minus_zero_m_pos_m_p))
                        C_lmlpmp_cross[1, lm_p_index_cur_neg_m_p, lm_index_cur_zero] = ( shortle[m_p %2]*
                                                                                    conjugate(C_lmlpmp_cross[1, lm_p_index_cur_pos_m_p, lm_index_cur_zero]) )

                        # TB correlations
                        C_lmlpmp_cross[2, lm_p_index_cur_pos_m_p, lm_index_cur_zero] += (coeff_T_lp * coeff_E_B_ell * 
                                                                                integrand[5, k_unique_index_cur, l_p-min_ell, l-min_ell] 
                                                                                * ipow[(l_p-l)%4] * conjugate(Xi_minus_zero_m_pos_m_p))
                        C_lmlpmp_cross[2, lm_p_index_cur_neg_m_p, lm_index_cur_zero] = ( shortle[m_p %2]*
                                                                                    conjugate(C_lmlpmp_cross[2, lm_p_index_cur_pos_m_p, lm_index_cur_zero]) )

                
                # for m_p = 0 and m != 0
                for m in range(1, l + 1):
                    if m%2 ==1:
                        continue
                    lm_index_cur_pos_m = l * (l+1) + m - min_ell*min_ell # for positive m
                    lm_index_cur_neg_m = l * (l +1) - m - min_ell*min_ell # for negative m
                    
                    wigner_cur_index_pos_m = lm_index[l, m]

                    wigner_D_l_pos_m_plus2 = ( wigner_d_l_m_2[wigner_d_index, wigner_cur_index_pos_m] 
                                                * phase_list_minus[m])
                    wigner_D_l_pos_m_minus2 = ( wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_cur_index_pos_m] 
                                                * phase_list_minus[m] )
                    Xi_plus_pos_m_zero_m_p = ( wigner_D_l_pos_m_plus2* conjugate(wigner_D_l_m_p_plus2) 
                                                + wigner_D_l_pos_m_minus2 * conjugate(wigner_D_l_m_p_minus2) )
                    Xi_minus_pos_m_zero_m_p = ( wigner_D_l_pos_m_minus2 * conjugate(wigner_D_l_m_p_minus2) 
                                                -  wigner_D_l_pos_m_plus2* conjugate(wigner_D_l_m_p_plus2) )


                    # TT
                    C_lmlpmp[0, lm_index_cur_pos_m, lm_p_index_cur_zero] += (coeff_T_ell * coeff_T_lp * 
                                                                                integrand[0, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_plus_pos_m_zero_m_p)
                    C_lmlpmp[0, lm_index_cur_neg_m, lm_p_index_cur_zero] = ( shortle[m %2]*
                                                                                conjugate(C_lmlpmp[0, lm_index_cur_pos_m, lm_p_index_cur_zero]) )
                        
                        
                    # EE
                    C_lmlpmp[1, lm_index_cur_pos_m, lm_p_index_cur_zero] += (coeff_E_B_ell * coeff_E_B_lp * 
                                                                            integrand[1, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                            * ipow[(l-l_p)%4] * Xi_plus_pos_m_zero_m_p)
                    C_lmlpmp[1, lm_index_cur_neg_m, lm_p_index_cur_zero] = ( shortle[m %2]*
                                                                                conjugate(C_lmlpmp[1, lm_index_cur_pos_m, lm_p_index_cur_zero]) )
                    
                    # BB

                    C_lmlpmp[2, lm_index_cur_pos_m, lm_p_index_cur_zero] += (coeff_E_B_ell * coeff_E_B_lp * 
                                                                            integrand[2, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                            * ipow[(l-l_p)%4] * Xi_plus_pos_m_zero_m_p)
                    C_lmlpmp[2, lm_index_cur_neg_m, lm_p_index_cur_zero] = ( shortle[m %2]*
                                                                                conjugate(C_lmlpmp[2, lm_index_cur_pos_m, lm_p_index_cur_zero]) )
                    
                    # TE
                    C_lmlpmp[3, lm_index_cur_pos_m, lm_p_index_cur_zero] += (coeff_T_ell * coeff_E_B_lp * 
                                                                                integrand[3, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_plus_pos_m_zero_m_p)
                    C_lmlpmp[3, lm_index_cur_neg_m, lm_p_index_cur_zero] = ( shortle[m %2]*
                                                                                conjugate(C_lmlpmp[3, lm_index_cur_pos_m, lm_p_index_cur_zero]) )
                      
                       
                    #EB
                    C_lmlpmp[4, lm_index_cur_pos_m, lm_p_index_cur_zero] += (coeff_E_B_ell * coeff_E_B_lp * 
                                                                                integrand[4, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_minus_pos_m_zero_m_p)
                    C_lmlpmp[4, lm_index_cur_neg_m, lm_p_index_cur_zero] = ( shortle[m %2]*
                                                                                conjugate(C_lmlpmp[4, lm_index_cur_pos_m, lm_p_index_cur_zero]) )
                    
                        

                    # TB
                    C_lmlpmp[5, lm_index_cur_pos_m, lm_p_index_cur_zero] += (coeff_T_ell * coeff_E_B_lp * 
                                                                                integrand[5, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * conjugate(Xi_minus_pos_m_zero_m_p))
                    C_lmlpmp[5, lm_index_cur_neg_m, lm_p_index_cur_zero] = ( shortle[m %2]*
                                                                                conjugate(C_lmlpmp[5, lm_index_cur_pos_m, lm_p_index_cur_zero]) )

                
                    if l != l_p:
                        # TE correlations
                        
                        C_lmlpmp_cross[0, lm_p_index_cur_zero, lm_index_cur_pos_m] += (coeff_T_lp * coeff_E_B_ell * 
                                                                                    integrand[3, k_unique_index_cur, l_p-min_ell, l-min_ell] 
                                                                                    * ipow[(l_p-l)%4] * conjugate(Xi_plus_pos_m_zero_m_p))
                        C_lmlpmp_cross[0, lm_p_index_cur_zero, lm_index_cur_neg_m] = ( shortle[m %2]*
                                                                                    conjugate(C_lmlpmp_cross[0, lm_p_index_cur_zero, lm_index_cur_pos_m]) )
                          
                        # EB correlations

                        C_lmlpmp_cross[1, lm_p_index_cur_zero, lm_index_cur_pos_m] += (coeff_E_B_lp * coeff_E_B_ell * 
                                                                                    integrand[4, k_unique_index_cur, l_p-min_ell, l-min_ell] 
                                                                                    * ipow[(l_p-l)%4] * conjugate(Xi_minus_pos_m_zero_m_p))
                        C_lmlpmp_cross[1, lm_p_index_cur_zero, lm_index_cur_neg_m] = ( shortle[m %2]*
                                                                                    conjugate(C_lmlpmp_cross[1, lm_p_index_cur_zero, lm_index_cur_pos_m]) )
                            

                        # TB correlations
                        C_lmlpmp_cross[2, lm_p_index_cur_zero, lm_index_cur_pos_m] += (coeff_T_lp * coeff_E_B_ell * 
                                                                                    integrand[5, k_unique_index_cur, l_p-min_ell, l-min_ell] 
                                                                                    * ipow[(l_p-l)%4] * conjugate(Xi_minus_pos_m_zero_m_p))
                        C_lmlpmp_cross[2, lm_p_index_cur_zero, lm_index_cur_neg_m] = ( shortle[m %2]*
                                                                                    conjugate(C_lmlpmp_cross[2, lm_p_index_cur_zero, lm_index_cur_pos_m]) )


                # for non-zero m and m'
                for m in range(1, l + 1):
                    lm_index_cur_pos_m = l * (l+1) + m - min_ell * min_ell # for positive m
                    lm_index_cur_neg_m = l * (l+1) - m - min_ell * min_ell # for negative m
                    
                    

                    wigner_cur_index_pos_m = lm_index[l, m]
             
                    wigner_D_l_pos_m_plus2 = ( wigner_d_l_m_2[wigner_d_index, wigner_cur_index_pos_m] 
                                                * phase_list_minus[m])
                    wigner_D_l_pos_m_minus2 = ( wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_cur_index_pos_m] 
                                                * phase_list_minus[m] )

                    
                    # Only do m-mp = 0 mod 2  
                    for m_p in range(1, l_p + 1):
                        if (m_p-m)%2 ==1:
                            continue

                        lm_p_index_cur_pos_m_p = l_p * (l_p+1) + m_p - min_ell*min_ell # for positive m
                        lm_p_index_cur_neg_m_p = l_p * (l_p+1) - m_p - min_ell*min_ell # for negative m
                        
                        wigner_p_cur_index_pos_m_p = lm_index[l_p, m_p]

                        wigner_D_l_neg_m_p_plus2 = ( shortle[m_p%2] * wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_p_cur_index_pos_m_p] 
                                                    * phase_list_plus[m_p] )
                        wigner_D_l_neg_m_p_minus2 = ( shortle[m_p%2] * wigner_d_l_m_2[wigner_d_index, wigner_p_cur_index_pos_m_p] 
                                                        * phase_list_plus[m_p] )

                        wigner_D_l_pos_m_p_plus2 = ( wigner_d_l_m_2[wigner_d_index, wigner_p_cur_index_pos_m_p] 
                                                    * phase_list_minus[m_p])
                        wigner_D_l_pos_m_p_minus2 = ( wigner_d_l_m_2[wigner_d_index, total_num_l_m + wigner_p_cur_index_pos_m_p] 
                                                    * phase_list_minus[m_p] )

                        Xi_plus_pos_m_pos_m_p = ( wigner_D_l_pos_m_plus2* conjugate(wigner_D_l_pos_m_p_plus2) 
                                                + wigner_D_l_pos_m_minus2 * conjugate(wigner_D_l_pos_m_p_minus2) )
                        Xi_plus_pos_m_neg_m_p = ( wigner_D_l_pos_m_plus2* conjugate(wigner_D_l_neg_m_p_plus2) 
                                                + wigner_D_l_pos_m_minus2 * conjugate(wigner_D_l_neg_m_p_minus2) )
                      
                        Xi_minus_pos_m_pos_m_p = ( wigner_D_l_pos_m_minus2 * conjugate(wigner_D_l_pos_m_p_minus2)
                                                    - wigner_D_l_pos_m_plus2* conjugate(wigner_D_l_pos_m_p_plus2) )
                        Xi_minus_pos_m_neg_m_p = ( wigner_D_l_pos_m_minus2 * conjugate(wigner_D_l_neg_m_p_minus2)
                                                    - wigner_D_l_pos_m_plus2* conjugate(wigner_D_l_neg_m_p_plus2) )
                      
                        
                        # TT correlations
                        C_lmlpmp[0, lm_index_cur_pos_m, lm_p_index_cur_pos_m_p] += (coeff_T_ell * coeff_T_lp * 
                                                                                integrand[0, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_plus_pos_m_pos_m_p)
                        C_lmlpmp[0, lm_index_cur_pos_m, lm_p_index_cur_neg_m_p] += (coeff_T_ell * coeff_T_lp * 
                                                                                integrand[0, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_plus_pos_m_neg_m_p)
                    
                        
                        C_lmlpmp[0, lm_index_cur_neg_m, lm_p_index_cur_neg_m_p] = ( shortle[(m + m_p)%2] *
                                                                                    conjugate(C_lmlpmp[0, lm_index_cur_pos_m, lm_p_index_cur_pos_m_p]) )
                        C_lmlpmp[0, lm_index_cur_neg_m, lm_p_index_cur_pos_m_p] = ( shortle[(m + m_p)%2] *
                                                                                    conjugate(C_lmlpmp[0, lm_index_cur_pos_m, lm_p_index_cur_neg_m_p]) )
                      
                        # EE correlations
                        C_lmlpmp[1, lm_index_cur_pos_m, lm_p_index_cur_pos_m_p] += (coeff_E_B_ell * coeff_E_B_lp * 
                                                                                integrand[1, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_plus_pos_m_pos_m_p)
                        C_lmlpmp[1, lm_index_cur_pos_m, lm_p_index_cur_neg_m_p] += (coeff_E_B_ell * coeff_E_B_lp * 
                                                                                integrand[1, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_plus_pos_m_neg_m_p)
               
                        C_lmlpmp[1, lm_index_cur_neg_m, lm_p_index_cur_neg_m_p] = ( shortle[(m + m_p)%2] *
                                                                                    conjugate(C_lmlpmp[1, lm_index_cur_pos_m, lm_p_index_cur_pos_m_p]) )
                        C_lmlpmp[1, lm_index_cur_neg_m, lm_p_index_cur_pos_m_p] = ( shortle[(m + m_p)%2] *
                                                                                    conjugate(C_lmlpmp[1, lm_index_cur_pos_m, lm_p_index_cur_neg_m_p]) )
                        
                        
                        # BB correlations
                        C_lmlpmp[2, lm_index_cur_pos_m, lm_p_index_cur_pos_m_p] += (coeff_E_B_ell * coeff_E_B_lp * 
                                                                                integrand[2, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_plus_pos_m_pos_m_p)
                        C_lmlpmp[2, lm_index_cur_pos_m, lm_p_index_cur_neg_m_p] += (coeff_E_B_ell * coeff_E_B_lp * 
                                                                                integrand[2, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_plus_pos_m_neg_m_p)
               
                        C_lmlpmp[2, lm_index_cur_neg_m, lm_p_index_cur_neg_m_p] = ( shortle[(m + m_p)%2] *
                                                                                    conjugate(C_lmlpmp[2, lm_index_cur_pos_m, lm_p_index_cur_pos_m_p]) )
                        C_lmlpmp[2, lm_index_cur_neg_m, lm_p_index_cur_pos_m_p] = ( shortle[(m + m_p)%2] *
                                                                                    conjugate(C_lmlpmp[2, lm_index_cur_pos_m, lm_p_index_cur_neg_m_p]) )
                        
                        # TE correlations
                        C_lmlpmp[3, lm_index_cur_pos_m, lm_p_index_cur_pos_m_p] += (coeff_T_ell * coeff_E_B_lp * 
                                                                                integrand[3, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_plus_pos_m_pos_m_p)
                       
                        C_lmlpmp[3, lm_index_cur_pos_m, lm_p_index_cur_neg_m_p] += (coeff_T_ell * coeff_E_B_lp * 
                                                                                integrand[3, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_plus_pos_m_neg_m_p)
                       
                        
                        C_lmlpmp[3, lm_index_cur_neg_m, lm_p_index_cur_neg_m_p] = ( shortle[(m + m_p)%2] *
                                                                                    conjugate(C_lmlpmp[3, lm_index_cur_pos_m, lm_p_index_cur_pos_m_p]) )
                      
                        C_lmlpmp[3, lm_index_cur_neg_m, lm_p_index_cur_pos_m_p] = ( shortle[(m + m_p)%2] *
                                                                                    conjugate(C_lmlpmp[3, lm_index_cur_pos_m, lm_p_index_cur_neg_m_p]) )
                      
                        # EB correlations
                        C_lmlpmp[4, lm_index_cur_pos_m, lm_p_index_cur_pos_m_p] += (coeff_E_B_ell * coeff_E_B_lp * 
                                                                                integrand[4, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_minus_pos_m_pos_m_p)
                     
                        C_lmlpmp[4, lm_index_cur_pos_m, lm_p_index_cur_neg_m_p] += (coeff_E_B_ell * coeff_E_B_lp * 
                                                                                integrand[4, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_minus_pos_m_neg_m_p)
            
                        C_lmlpmp[4, lm_index_cur_neg_m, lm_p_index_cur_neg_m_p] = ( shortle[(m + m_p)%2] *
                                                                                    conjugate(C_lmlpmp[4, lm_index_cur_pos_m, lm_p_index_cur_pos_m_p]) )
                      
                        C_lmlpmp[4, lm_index_cur_neg_m, lm_p_index_cur_pos_m_p] = ( shortle[(m + m_p)%2] *
                                                                                    conjugate(C_lmlpmp[4, lm_index_cur_pos_m, lm_p_index_cur_neg_m_p]) )
                      
                        # TB correlations
                        C_lmlpmp[5, lm_index_cur_pos_m, lm_p_index_cur_pos_m_p] += (coeff_T_ell * coeff_E_B_lp * 
                                                                                integrand[5, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_minus_pos_m_pos_m_p)
                    
                        C_lmlpmp[5, lm_index_cur_pos_m, lm_p_index_cur_neg_m_p] += (coeff_T_ell * coeff_E_B_lp * 
                                                                                integrand[5, k_unique_index_cur, l-min_ell, l_p-min_ell] 
                                                                                * ipow[(l-l_p)%4] * Xi_minus_pos_m_neg_m_p)
                     
                        C_lmlpmp[5, lm_index_cur_neg_m, lm_p_index_cur_neg_m_p] = ( shortle[(m + m_p)%2] *
                                                                                    conjugate(C_lmlpmp[5, lm_index_cur_pos_m, lm_p_index_cur_pos_m_p]) )
                    
                        C_lmlpmp[5, lm_index_cur_neg_m, lm_p_index_cur_pos_m_p] = ( shortle[(m + m_p)%2] *
                                                                                    conjugate(C_lmlpmp[5, lm_index_cur_pos_m, lm_p_index_cur_neg_m_p]) )
                        if l != l_p:
                            # TE correlations

                            C_lmlpmp_cross[0, lm_p_index_cur_pos_m_p, lm_index_cur_pos_m] += (coeff_T_lp * coeff_E_B_ell * 
                                                                                    integrand[3, k_unique_index_cur, l_p-min_ell, l-min_ell] 
                                                                                    * ipow[(l_p-l)%4] * conjugate(Xi_plus_pos_m_pos_m_p))

                            C_lmlpmp_cross[0, lm_p_index_cur_neg_m_p, lm_index_cur_pos_m] += (coeff_T_lp * coeff_E_B_ell * 
                                                                                    integrand[3, k_unique_index_cur, l_p-min_ell, l-min_ell] 
                                                                                    * ipow[(l_p-l)%4] * conjugate(Xi_plus_pos_m_neg_m_p))
                            
                            C_lmlpmp_cross[0, lm_p_index_cur_neg_m_p, lm_index_cur_neg_m] = ( shortle[(m + m_p)%2] *
                                                                                        conjugate(C_lmlpmp_cross[0, lm_p_index_cur_pos_m_p, lm_index_cur_pos_m]) )
                            
                            C_lmlpmp_cross[0, lm_p_index_cur_pos_m_p, lm_index_cur_neg_m] = ( shortle[(m + m_p)%2] *
                                                                                        conjugate(C_lmlpmp_cross[0, lm_p_index_cur_neg_m_p, lm_index_cur_pos_m]) )
                            
                            
                            # EB correlations

                            C_lmlpmp_cross[1, lm_p_index_cur_pos_m_p, lm_index_cur_pos_m] += (coeff_E_B_lp * coeff_E_B_ell * 
                                                                                    integrand[4, k_unique_index_cur, l_p-min_ell, l-min_ell] 
                                                                                    * ipow[(l_p-l)%4] * conjugate(Xi_minus_pos_m_pos_m_p))
                            
                            C_lmlpmp_cross[1, lm_p_index_cur_neg_m_p, lm_index_cur_pos_m] += (coeff_E_B_lp * coeff_E_B_ell * 
                                                                                    integrand[4, k_unique_index_cur, l_p-min_ell, l-min_ell] 
                                                                                    * ipow[(l_p-l)%4] * conjugate(Xi_minus_pos_m_neg_m_p))
          
                            C_lmlpmp_cross[1, lm_p_index_cur_neg_m_p, lm_index_cur_neg_m] = ( shortle[(m + m_p)%2] *
                                                                                        conjugate(C_lmlpmp_cross[1, lm_p_index_cur_pos_m_p, lm_index_cur_pos_m]) )

                            C_lmlpmp_cross[1, lm_p_index_cur_pos_m_p, lm_index_cur_neg_m] = ( shortle[(m + m_p)%2] *
                                                                                        conjugate(C_lmlpmp_cross[1, lm_p_index_cur_neg_m_p, lm_index_cur_pos_m]) )
     
                            # TB correlations
                            C_lmlpmp_cross[2, lm_p_index_cur_pos_m_p, lm_index_cur_pos_m] += (coeff_T_lp * coeff_E_B_ell * 
                                                                                    integrand[5, k_unique_index_cur, l_p-min_ell, l-min_ell] 
                                                                                    * ipow[(l_p-l)%4] * conjugate(Xi_minus_pos_m_pos_m_p))

                            C_lmlpmp_cross[2, lm_p_index_cur_neg_m_p, lm_index_cur_pos_m] += (coeff_T_lp * coeff_E_B_ell * 
                                                                                    integrand[5, k_unique_index_cur, l_p-min_ell, l-min_ell] 
                                                                                    * ipow[(l_p-l)%4] * conjugate(Xi_minus_pos_m_neg_m_p))
                 
                            C_lmlpmp_cross[2, lm_p_index_cur_neg_m_p, lm_index_cur_neg_m] = ( shortle[(m + m_p)%2] *
                                                                                        conjugate(C_lmlpmp_cross[2, lm_p_index_cur_pos_m_p, lm_index_cur_pos_m]) )

                            C_lmlpmp_cross[2, lm_p_index_cur_pos_m_p, lm_index_cur_neg_m] = ( shortle[(m + m_p)%2] *
                                                                                        conjugate(C_lmlpmp_cross[2, lm_p_index_cur_neg_m_p, lm_index_cur_pos_m]) )
       
                

    C_lmlpmp *= pi*pi / (2 * V)
    C_lmlpmp_cross *= pi*pi / (2 * V)

    
    return C_lmlpmp, C_lmlpmp_cross


