'''
Script contains analytical function definitions for comparison with numerics
    - solve_cusp_equation()
    - solve_Pearcey_max_equation()
    - calculate_Taylor_correction_to_Pearcey_max()
'''

import numpy as np


def solve_cusp_equation(beta_arr, a0, tau, theta_c=np.pi):
    '''
    Find beta_c - solution of the cusp equation which is the result of
    the system of 3 equations: Phi'(phi) = Phi''(phi) = Phi(3)(phi) = 0
    beta_arr - array over beta
    theta_c - angle at which cusp is situated
    '''
    angle = 0.5*(1 - np.cos(theta_c))
    exp = np.exp(-0.5)
    a_sq = 2 * a0**2 * exp / np.sqrt(1 + beta_arr**2)
    lhs = beta_arr * (1 + angle*a_sq)
    rhs = np.sqrt(2) * angle * a0**2 * tau * exp
    err = np.abs(lhs - rhs)
    idx = np.argmin(err)
    return beta_arr[idx], err[idx]


def solve_cusp_equation_for_a0_arr(beta_arr, a0_arr, tau, theta_c=np.pi):
    '''
    Do solve_cusp_equation() for an array of a0
    '''
    n_a0 = a0_arr.shape[0]
    beta_solution, err_solution = np.zeros(n_a0), np.zeros(n_a0)
    for i in range(n_a0):
        beta_solution[i], err_solution[i] = solve_cusp_equation(beta_arr, a0_arr[i], tau, theta_c)
    return beta_solution, err_solution


def solve_Pearcey_max_equation(beta_arr, a0, tau, theta=np.pi, Cmax=2.16):
    '''
    Pe (x, y) - Pearcey integral
    Find beta_Pearcey - solution of the system of two equations specifying the
    location of Pearcey maximum
    beta_arr - array of beta values
    theta - angle at which Pearcey maximum is observed
    Cmax (x_max=-2.16) - the value of x argument in Pe(x,y) at which Pearcey maximum
                         is observed
    '''
    angle = 1 - np.cos(theta)
    a_sq = 2*a0**2/np.sqrt(1+beta_arr**2) * np.exp(-0.5)
    tau_eff = tau * np.sqrt(1+beta_arr**2)
    lhs_1 = angle*a_sq / (2*np.sqrt(2))
    lhs_2 = beta_arr/tau_eff * (1 + 0.5*angle*a_sq)
    lhs = lhs_1 - lhs_2
    w = (1 - beta_arr/(np.sqrt(2)*tau_eff)) / (1 + 0.25*angle*a_sq)
    prefactor = np.sqrt(w * angle * a_sq / (3*np.sqrt(2)*tau_eff))
    rhs = Cmax * prefactor * (1 + 0.25*angle*a_sq)
    err = np.abs(lhs - rhs)
    idx = np.argmin(err)
    return beta_arr[idx], w[idx], err[idx]


def solve_Pearcey_max_equation_for_a0_arr(beta_arr, a0_arr, tau, theta=np.pi, Cmax=2.16):
    '''
    Do solve_Pearcey_max_equation() for an array of a0
    '''
    n_a0 = a0_arr.shape[0]
    beta_solution, w_solution, err_solution = [np.zeros(n_a0) for i in range(3)]
    for i in range(n_a0):
        beta_solution[i], w_solution[i], err_solution[i] = solve_Pearcey_max_equation(beta_arr, a0_arr[i], tau, theta, Cmax)
    return beta_solution, w_solution, err_solution


def solve_fold_equation(phi_arr, beta, a0=1, tau=2*np.pi):
    '''
    Calculate error for the fold equation for an array of laser phase phi_arr
    '''
    a_sq = 2*a0**2 / np.sqrt(1 + beta**2)
    tau_eff = tau * np.sqrt(1 + beta**2)
    prefactor = a_sq * np.exp(-(phi_arr/tau_eff)**2)
    lhs = (2*phi_arr*(1 + beta*phi_arr/tau_eff**2) + beta) * prefactor
    rhs = -2*beta
    err = np.abs(lhs - rhs)
    return err


def calculate_w_on_fold(phi_fold, beta, a0=1, tau=2*np.pi):
    '''
    Calculate frequency on fold (for spesified array phi_fold)
    '''
    a_sq = 2*a0**2 / np.sqrt(1 + beta**2)
    tau_eff = tau * np.sqrt(1 + beta**2)
    w_L = 1 + beta*phi_fold/tau_eff**2
    w = w_L / (1 + 0.5*a_sq*np.exp(-(phi_fold/tau_eff)**2))
    return w


def calculate_fold_w_beta(beta_arr, a0=1, tau=2*np.pi, eps=8e-3):
    '''
    For given beta_arr calculate (w, beta) values showing fold's location
    '''
    data = []
    for i,beta in enumerate(beta_arr):
        phi_b = 3*tau*np.sqrt(1+beta**2)
        phi_arr = np.linspace(-phi_b, phi_b, int(2*phi_b*100))
        err = solve_fold_equation(phi_arr, beta, a0, tau)
        idx_phi = (err < eps)
        phi_fold = phi_arr[idx_phi]
        w = calculate_w_on_fold(phi_fold, beta, a0, tau)
        data.append([phi_fold, w])
    w_fold, beta_fold = np.array([]), np.array([])
    for i,elem in enumerate(data):
        w_fold = np.concatenate((w_fold, elem[1]))
        beta_fold = np.concatenate((beta_fold, beta_arr[i]*np.ones(len(elem[1]))))
    return beta_fold, w_fold


def calculate_Taylor_correction_to_Pearcey_max(beta0, w0, a0, tau):
    '''
    Taylor correction of 2nd order to Pearcey max position. Pearcey derivatives
    near maximum were calculated numerically.
    w0, beta0 - Pearcey max location
    '''
    # Pearcey prefactor
    sq = np.sqrt(1 + beta0**2)
    g = np.sqrt(w0) * sq
    dg_dw = 0.5/np.sqrt(w0) * sq
    dg_dbeta = np.sqrt(w0)*beta0/sq
    dg_dw2 = -0.25/w0**(1.5)*sq
    dg_dbeta2 = np.sqrt(w0) / sq**3
    dg_dw_dbeta = 0.5/np.sqrt(w0)*beta0/sq

    # x and y derivatives
    tau_eff = tau * sq
    a_sq = 2*a0**2/sq * np.exp(-0.5)

    x_pref = np.sqrt(3*tau_eff/(np.sqrt(2)*w0*a_sq))
    dx_dw = x_pref*(1/(2*np.sqrt(2))*a_sq + beta0/(2*w0*tau_eff))
    dx_dbeta = -x_pref/tau_eff

    y_pref = (6*np.sqrt(2)*tau_eff**3/(w0*a_sq))**(0.25)
    dy_dw = 0.25*y_pref*(3*(1+0.5*a_sq) + (1 - beta0/(np.sqrt(2)*tau_eff))/w0)
    dy_dbeta = y_pref*((w0-1)*beta0/sq**2 + 1/(np.sqrt(2)*tau_eff))

    # Numerical values for Pearcey integral derivatives
    Pe = 2.6351
    dPe_dx2 = -0.7649
    dPe_dy2 = -1.7464
    dPe_dx_dy = 0.0010

    # Pearcey derivatives with respect to w, beta
    dPe_dw2 = dPe_dx2*(dx_dw)**2 + dPe_dy2*(dy_dw)**2 + 2*dPe_dx_dy*dx_dw*dy_dw
    dPe_dbeta2 = dPe_dx2*(dx_dbeta)**2 + dPe_dy2*(dy_dbeta)**2 + 2*dPe_dx_dy*dx_dbeta*dy_dbeta
    dPe_dw_dbeta = dPe_dx2*dx_dw*dx_dbeta + dPe_dy2*dy_dw*dy_dbeta + dPe_dx_dy*(dx_dw*dy_dbeta+dx_dbeta*dy_dw)

    # Functional derivatives
    df_dw = dg_dw*Pe**2
    df_dbeta = dg_dbeta*Pe**2
    df_dw2 = dg_dw2*Pe**2 + 2*g*Pe*dPe_dw2
    df_dbeta2 = dg_dbeta2*Pe**2 + 2*g*Pe*dPe_dbeta2
    df_dw_dbeta = dg_dw_dbeta*Pe**2 + 2*g*Pe*dPe_dw_dbeta

    # Calculate Taylor correction to beta, w
    top = df_dw_dbeta * df_dw / df_dw2 - df_dbeta
    bottom = df_dbeta2 - (df_dw_dbeta)**2/df_dw2
    delta_beta = top/bottom
    beta_T = beta0 + delta_beta

    top = df_dw + df_dw_dbeta*(beta_T-beta0)
    bottom = df_dw2
    delta_w = top/bottom
    w_T = w0 - delta_w

    return beta_T, w_T


def calculate_Taylor_correction_for_a0_arr(beta0_arr, w0_arr, a0_arr, tau):
    '''
    Do calculate_Taylor_correction_to_Pearcey_max() for an array of a0
    '''
    n_a0 = a0_arr.shape[0]
    w_T_arr, beta_T_arr = np.zeros(n_a0), np.zeros(n_a0)
    for i in range(n_a0):
        beta_T_arr[i], w_T_arr[i] = calculate_Taylor_correction_to_Pearcey_max(beta0_arr[i], w0_arr[i],
                                                                               a0_arr[i], tau)
    return beta_T_arr, w_T_arr


def save_analytics_fixed_tau(beta_arr, a0_arr, beta_cusp, beta_Pe, w_Pe,
                             beta_Taylor, w_Taylor, folder='data/analytics'):
    n_a0 = a0_arr.shape[0]
    # Save a0_arr and beta_arr
    np.save(folder + '/a0_arr.npy', a0_arr)
    np.save(folder + '/beta_arr.npy', beta_arr)
    # Save cusp data
    np.save(folder + '/cusp.npy', beta_cusp)
    # Save Pearcey data
    data = np.zeros((2,n_a0))
    data[0] = beta_Pe.copy()
    data[1] = w_Pe.copy()
    np.save(folder + '/Pearcey.npy', data)
    # Save Taylor data
    data = np.zeros((2,n_a0))
    data[0] = beta_Taylor.copy()
    data[1] = w_Taylor.copy()
    np.save(folder + '/Taylor.npy', data)
    print('Saving is finished')


def load_analytics_fixed_tau(folder='data/analytics'):
    a0_arr = np.load(folder + '/a0_arr.npy')
    beta_arr = np.load(folder + '/beta_arr.npy')
    beta_cusp = np.load(folder + '/cusp.npy')
    data = np.load(folder + '/Pearcey.npy')
    beta_Pe, w_Pe = data[0], data[1]
    data = np.load(folder + '/Taylor.npy')
    beta_Taylor, w_Taylor = data[0], data[1]
    return beta_arr, a0_arr, beta_cusp, beta_Pe, w_Pe, beta_Taylor, w_Taylor


def linear_reference(a0, tau):
    '''
    Approximation of N_ph_max in the limit of linear Compton
    '''
    return a0**2 * tau**2 / 137 / (4*np.pi)


def spectrum_estimate(w, beta, a0=1, tau=2*np.pi, Pe=2.6351):
    '''
    Estimate N_ph_max from Pearcey approximation to the radiation integral
    w, beta - values at which N_ph is estimated
    Pe = |Pe| - modulus of Pearcey integral at maximum
    '''
    tau_eff = tau*np.sqrt(1 + beta**2)
    a_sq = 2*a0**2 / np.sqrt(1 + beta**2) * np.exp(-0.5)
    prefactor = np.sqrt(6*np.sqrt(2)*tau_eff**3/(w*a_sq))
    N_ph = w / (2*4*np.pi**2) / 137 * a_sq/2 * prefactor * Pe**2
    return N_ph


def collect_Nmax_arrays(a0_arr, tau, beta_Pe, w_Pe, beta_optimal, beta_arr, N_max_beta_a0):
    '''
    Collect Nmax(a0) arrays for linear Compton, simulation and analytical estimation
    '''
    Nmax_linear = linear_reference(a0_arr, tau)
    Nmax_optimal, Nmax_estimation = [np.zeros_like(a0_arr) for i in range(2)]

    for i,a0 in enumerate(a0_arr):
        Nmax_estimation[i] = spectrum_estimate(w_Pe[i], beta_Pe[i], a0, tau)

        beta0 = beta_optimal[i]
        idx_beta0 = np.where(np.isclose(beta_arr, beta0))[0][0]
        Nmax_optimal[i] = N_max_beta_a0[idx_beta0,i]
    return Nmax_linear, Nmax_optimal, Nmax_estimation
