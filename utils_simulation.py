'''
Script contains the definition of laser pulse (we use 'gauss_spectral') and
func beta_scan_classic which calculates the COmpton spectra for a range of
linear chirp parameters
'''

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from ComptonSpec_classic import Trajectory, Spectrum


@np.vectorize
def envelope(x, mode='gauss', tau=2*np.pi, beta=0):
    if mode == 'rectangle':
        idx = (x >= -tau/2) & (x <= tau/2)
        g = np.zeros_like(x)
        g[idx] = 1
    elif mode == 'super gauss':
        g = np.exp(-(x/tau)**60)
    elif mode == 'gauss':
        g = np.exp(-x**2/tau**2)
    elif mode == 'gauss_spectral':
        g = np.exp(-x**2/(2*tau**2*(1 + beta**2)))
    return g


@np.vectorize
def phase(x, mode='gauss', tau=2*np.pi, beta=0):
    if mode != 'gauss_spectral':
        return x
    else:
        return x + x**2*beta / (2*tau**2*(1 + beta**2)) + np.arctan(beta/(1+np.sqrt(1+beta**2)))


def calc_A(x, mode='gauss', tau=2*np.pi, a0=1, beta=0, polarization='circular'):
    '''
    Calculate laser vector potential
    '''
    ph = phase(x, mode, tau, beta)
    if mode == 'gauss_spectral':
        a0_loc = a0 / np.power(1+beta**2, 0.25)
    else:
        a0_loc = a0
    if polarization == 'circular':
        return a0_loc*envelope(x,mode,tau,beta)*np.array([np.cos(ph), np.sin(ph)])
    elif polarization == 'linear':
        return a0_loc*envelope(x,mode,tau,beta)*np.array([np.cos(ph), np.zeros_like(ph)])


def beta_scan_classic(beta_arr, a0=1, tau=2*np.pi, theta=np.pi, phi=0, wb=[0.,1.4],
                      mode='gauss_spectral', polarization='circular'):
    '''
    Calculate Compton emission spectra for an array of linear chirp parameters
    beta_arr for spectral gaussian laser pulse (a0, tau, beta) at given
    angles (theta, phi) and return frequencies and spectra in the interval
    specified by wb
    beta_arr - array of chirp parameter beta
    a0 - dimensional laser pulse amplitude
    tau - pulse duration
    theta, phi - angles
    wb - interval of frequencies for which the spectra will be saved
    mode - type of temporal envelope
    polarization - [linear, circular] laser pulse polarization
    '''
    u0, r0 = np.array([0.,0.,0.]), np.array([0.,0.,0.])
    traj = Trajectory(u0, r0)
    n = beta_arr.shape[0]
    w_arr, N_ph_arr = [], []
    for i in range(n):
        beta = beta_arr[i]

        # Define grid over laser pulse phase
        eta_b = 4.4*tau*np.sqrt(1+beta**2)
        eta = np.linspace(-eta_b, eta_b, int(2*eta_b*100))
        # Calculate laser pulse vector potential and electron trajectory
        A = calc_A(eta, mode, tau=tau, a0=a0, beta=beta, polarization=polarization)
        u, r = traj.calc_u_x(A, eta)
        # Calculate Compton spectra
        spec = Spectrum(eta, u, r)
        I, w = spec.calc_spectrum_I_w(theta=theta, phi=phi)
        idx = (w > 0)
        # Transform d2 I/dw dOmega -> d2 N_ph / dw dOmega
        N_ph = I[idx] / w[idx] / 137
        w = w[idx]

        # Save the spectrum in the frequency interval wb
        idx = (w > wb[0]) & (w < wb[1])
        w_arr.append(w[idx])
        N_ph_arr.append(N_ph[idx])
    return w_arr, N_ph_arr


def a0_beta_scan_classic(beta_arr, a0_arr, tau=2*np.pi, theta=np.pi, phi=0, wb=[0.,1.4],
                         mode='gauss_spectral', polarization='circular'):
    '''
    Perform beta_scan function for a given a0_arr and return the spectra (w, N_ph),
    N_ph_max values and optimal beta
    '''
    n_a0, n_beta = a0_arr.shape[0], beta_arr.shape[0]
    w_a0_list, N_ph_a0_list, N_max_a0_list = [], [], []
    beta_optimal = np.zeros_like(a0_arr)

    for i in range(n_a0):
        w_arr, N_ph_arr = beta_scan_classic(beta_arr, a0=a0_arr[i], tau=tau, theta=theta, phi=phi, wb=wb,
                                            mode=mode, polarization=polarization)
        w_a0_list.append(w_arr)
        N_ph_a0_list.append(N_ph_arr)
        print('i = {}, Calculation is finished!'.format(i))

        # Calculate the maximum of photon spectrum and the optimal beta at which
        # it occurs
        N_max = np.zeros(n_beta)
        for j in range(n_beta):
            N_max[j] = np.max(N_ph_arr[j])
        idx_max = np.argmax(N_max)
        N_max_a0_list.append(N_max)
        beta_optimal[i] = beta_arr[idx_max]
    return w_a0_list, N_ph_a0_list, N_max_a0_list, beta_optimal


def transform_N_ph_list_to_arr(N_ph_a0_list):
    '''
    Transform N_ph_a0 list into an array of max values over (beta, a0) grid
    for plotting
    '''
    n_a0, n_beta = len(N_ph_a0_list), len(N_ph_a0_list[0])
    N_max_beta_a0 = np.zeros((n_beta, n_a0))
    for i in range(n_a0):
        for j in range(n_beta):
            N_max_beta_a0[j,i] = np.max(N_ph_a0_list[i][j])
    return N_max_beta_a0


def interpolate_spectra_on_same_grid(w_list, N_ph_list, wb=[.01,1.5]):
    '''
    For various laser pulse parameters, spectra are calculated on different
    frequency grids. This function interpolates all spectra on the frequency
    grid obtained in a non-chirped case.
    w_list, N_ph_list - outputs of beta_scan_classic
    '''
    n_beta = len(w_list)
    idx_w = (w_list[0] >= wb[0]) & (w_list[0] <= wb[1])
    w_grid = w_list[0][idx_w]
    N_ph_arr = np.zeros((n_beta, w_grid.shape[0]))
    for i in range(n_beta):
        idx = (w_list[i] >= wb[0]) & (w_list[i] <= wb[1])
        N_ph_interpolator = interp1d(w_list[i][idx], N_ph_list[i][idx], fill_value='extrapolate')
        N_ph_arr[i] = N_ph_interpolator(w_grid)
    return w_grid, N_ph_arr


def interpolate_spectra_for_a0_arr(w_a0_list, N_ph_a0_list, a0_arr, wb=[.01,1.5]):
    '''
    Do the interpolation procedure for the results for a0_arr
    '''
    n_a0 = a0_arr.shape[0]
    w_a0_list_interp, N_ph_a0_list_interp = [], []
    for i in range(n_a0):
        w_grid, N_ph_arr = interpolate_spectra_on_same_grid(w_a0_list[i], N_ph_a0_list[i], wb=wb)
        w_a0_list_interp.append(w_grid)
        N_ph_a0_list_interp.append(N_ph_arr)
    return w_a0_list_interp, N_ph_a0_list_interp


def save_data_fixed_tau(w_a0_list, N_ph_a0_list, beta_arr, a0_arr, N_max_beta_a0,
                        beta_optimal, folder='tau_2'):
    n_a0 = a0_arr.shape[0]
    # Save spectra
    for i in range(n_a0):
        w_data, N_ph_data = w_a0_list[i], N_ph_a0_list[i]
        np.save(folder + '/w_{:.2f}.npy'.format(a0_arr[i]), w_data)
        np.save(folder + '/N_ph_{:.2f}.npy'.format(a0_arr[i]), N_ph_data)
    # Save beta and a0 arrays
    np.save(folder + '/a0_arr.npy', a0_arr)
    np.save(folder + '/beta_arr.npy', beta_arr)
    # Save N_max
    np.save(folder + '/N_max_beta_a0.npy', N_max_beta_a0)
    np.save(folder + '/beta_optimal.npy', beta_optimal)
    print('Saving is finished')


def load_data_fixed_tau(folder='tau_2'):
    a0_arr = np.load(folder + '/a0_arr.npy')
    beta_arr = np.load(folder + '/beta_arr.npy')
    N_max_beta_a0 = np.load(folder + '/N_max_beta_a0.npy')
    beta_optimal = np.load(folder + '/beta_optimal.npy')
    n_a0 = a0_arr.shape[0]
    w_a0_list, N_ph_a0_list = [], []
    for i in range(n_a0):
        w_data = np.load(folder + '/w_{:.2f}.npy'.format(a0_arr[i]))
        N_ph_data = np.load(folder + '/N_ph_{:.2f}.npy'.format(a0_arr[i]))
        w_a0_list.append(w_data)
        N_ph_a0_list.append(N_ph_data)
    return w_a0_list, N_ph_a0_list, beta_arr, a0_arr, N_max_beta_a0, beta_optimal
