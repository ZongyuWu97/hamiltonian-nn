# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import autograd
import autograd.numpy as np

import scipy.integrate

solve_ivp = scipy.integrate.solve_ivp


def hamiltonian_fn(coords):
    q, p = np.split(coords, 2)
    H = p ** 2 + q ** 2  # spring hamiltonian (linear oscillator)
    return H


def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = np.split(dcoords, 2)
    S = np.concatenate([dpdt, -dqdt], axis=-1)
    return S


import numpy as np
from scipy.integrate import solve_ivp

def generate_gaussian_random_field_1d(n_points, alpha=2.0, seed=None):
    """
    Generates a 1D Gaussian random field of length n_points with a power-law
    spectral density ~ (freq)^{-alpha}.
    """
    if seed is not None:
        np.random.seed(seed)

    n_freq = n_points // 2 + 1  # for rfft
    real_part = np.random.randn(n_freq)
    imag_part = np.random.randn(n_freq)
    freq_data = real_part + 1j * imag_part

    # Power-law scaling: amplitude ~ (k+1)^(-alpha/2)
    for k in range(1, n_freq):
        amp = (k + 1)**(-alpha * 0.5)
        freq_data[k] *= amp

    # Inverse RFFT to get real correlated noise in time domain
    grf_time = np.fft.irfft(freq_data, n=n_points)
    return grf_time

def fourier_filter(signal, keep_frequency=0.2):
    """
    Applies a simple low-pass filter by retaining only the lowest-frequency
    portion of the signal's spectrum. Frequencies above 'keep_frequency'
    fraction of the spectrum are zeroed out.
    """
    n = len(signal)
    freq_data = np.fft.rfft(signal)
    cutoff = int(len(freq_data) * keep_frequency)
    freq_data[cutoff:] = 0
    filtered_signal = np.fft.irfft(freq_data, n=n)
    return filtered_signal

def five_point_derivative(signal, dt):
    """
    Computes the derivative of 'signal' with a 5-point central difference stencil
    for interior points. For the boundaries, uses simpler forward/backward differences.

    five-point stencil (centered at i):
      f'(x_i) ~ [ -f(x_{i+2}) + 8f(x_{i+1}) - 8f(x_{i-1}) + f(x_{i-2}) ] / (12 * dt)

    Parameters
    ----------
    signal : ndarray
        1D array of length N.
    dt : float
        Time step between consecutive points of 'signal'.

    Returns
    -------
    deriv : ndarray
        Approximate derivative of 'signal' at each index.
    """
    n = len(signal)
    deriv = np.zeros_like(signal)

    # For i in [2, n-3], use the 5-point formula
    for i in range(2, n - 2):
        deriv[i] = (
            -signal[i + 2]
            + 8.0 * signal[i + 1]
            - 8.0 * signal[i - 1]
            + signal[i - 2]
        ) / (12.0 * dt)

    # Near boundaries, we fall back to simpler finite differences.

    # i=0: forward difference
    deriv[0] = (signal[1] - signal[0]) / dt
    # i=1: 3-point forward difference
    deriv[1] = (signal[2] - signal[0]) / (2.0 * dt)

    # i=n-2: 3-point backward difference
    deriv[-2] = (signal[-1] - signal[-3]) / (2.0 * dt)
    # i=n-1: backward difference
    deriv[-1] = (signal[-1] - signal[-2]) / dt

    return deriv

def get_trajectory(t_span=[0, 3],
                   timescale=100,
                   radius=None,
                   y0=None,
                   noise_std=0.1,
                   keep_frequency=0.2,
                   alpha=2.0,
                   seed=None,
                   denoise=True,
                   **kwargs):
    """
    1) Solves a Hamiltonian system (via solve_ivp) to obtain q(t), p(t).
    2) Adds correlated Gaussian Random Field (GRF) noise to q and p.
    3) Applies a low-pass Fourier filter to denoise the signals.
    4) Computes dq/dt and dp/dt using a 5-point finite difference stencil.

    Parameters
    ----------
    t_span : list of float
        The time interval [t0, tf] over which to solve, e.g. [0, 3].
    timescale : int
        Number of samples per unit time, e.g. 100 => 300 points if t_span=[0,3].
    radius : float
        Radius for the initial state if y0 is None. If None, randomly sampled.
    y0 : array-like of shape (2,)
        Initial condition [q0, p0]. If None, pick randomly in [-1, 1].
    noise_std : float
        Scale factor for the correlated noise, by default 0.1.
    keep_frequency : float
        Fraction of low-frequency components to retain in the Fourier filter, by default 0.2.
    alpha : float
        Spectral exponent for the GRF. Larger alpha => stronger correlation.
    seed : int, optional
        Random seed for reproducibility.
    **kwargs : dict
        Must include 'fun': the dynamics function. Additional arguments (e.g. rtol, atol)
        are passed to solve_ivp.

    Returns
    -------
    q_denoised : ndarray
        Filtered position array (length N).
    p_denoised : ndarray
        Filtered momentum array (length N).
    dqdt : ndarray
        Derivative of q_denoised, using 5-point finite difference (length N).
    dpdt : ndarray
        Derivative of p_denoised, using 5-point finite difference (length N).
    t_eval : ndarray
        Time points at which the solution is evaluated (length N).
    """

    # ----------------------------------------------------------------
    # 1) Build time array and solve the ODE with t_eval
    # ----------------------------------------------------------------
    num_points = int(timescale * (t_span[1] - t_span[0]))
    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    if seed is not None:
        np.random.seed(seed)

    if y0 is None:
        y0 = np.random.rand(2) * 2 - 1
    if radius is None:
        radius = np.random.rand() * 0.9 + 0.1
    y0 = y0 / np.sqrt((y0**2).sum()) * radius


    # Force solve_ivp to evaluate at our desired time points
    spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, p = spring_ivp['y'][0], spring_ivp['y'][1]
    dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    dydt = np.stack(dydt).T
    dqdt, dpdt = np.split(dydt,2)

    # ----------------------------------------------------------------
    # 2) Generate correlated noise and add to q, p
    # ----------------------------------------------------------------
    grf_q = generate_gaussian_random_field_1d(num_points, alpha=alpha, seed=seed)
    # Use a different seed for p if you wish, e.g. seed+1
    grf_p = generate_gaussian_random_field_1d(num_points, alpha=alpha, seed=seed + 1 if seed is not None else None)

    q_noisy = q + grf_q * noise_std
    p_noisy = p + grf_p * noise_std
    # ----------------------------------------------------------------
    # 3) Apply Fourier-based low-pass filter
    # ----------------------------------------------------------------
    q_denoised = fourier_filter(q_noisy, keep_frequency=keep_frequency) if denoise else q_noisy
    p_denoised = fourier_filter(p_noisy, keep_frequency=keep_frequency) if denoise else p_noisy

    # ----------------------------------------------------------------
    return q_denoised, p_denoised, dqdt, dpdt, t_eval


def get_dataset(seed=0, samples=50, test_split=0.5, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    for s in range(samples):
        x, y, dx, dy, t = get_trajectory(**kwargs)
        xs.append(np.stack([x, y]).T)
        dxs.append(np.stack([dx, dy]).T)

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data


def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ys = np.stack([b.flatten(), a.flatten()])

    # get vector directions
    dydt = [dynamics_fn(None, y) for y in ys.T]
    dydt = np.stack(dydt).T

    field['x'] = ys.T
    field['dx'] = dydt.T
    return field