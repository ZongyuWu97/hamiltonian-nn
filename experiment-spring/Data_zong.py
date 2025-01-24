import autograd
import autograd.numpy as np
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

# Hamiltonian function and dynamics (unchanged from your example)
def hamiltonian_fn(coords):
    q, p = np.split(coords, 2)
    H = p**2 / 2 + q**2 / 2  # spring hamiltonian (linear oscillator)
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

    # Near boundaries, simpler finite differences
    deriv[0] = (signal[1] - signal[0]) / dt                # forward difference
    deriv[1] = (signal[2] - signal[0]) / (2.0 * dt)         # 3-point forward
    deriv[-2] = (signal[-1] - signal[-3]) / (2.0 * dt)      # 3-point backward
    deriv[-1] = (signal[-1] - signal[-2]) / dt             # backward difference

    return deriv

def get_trajectory(t_span=[0, 3],
                   timescale=100,
                   radius=None,
                   y0=None,
                   noise_std=0.1,
                   keep_frequency=0.2,
                   alpha=2.0,
                   seed=None,
                   **kwargs):
    """
    1) Generate 30% more samples than 'timescale' indicates.
    2) Solve a Hamiltonian system (via solve_ivp) to obtain q(t), p(t).
    3) Add correlated Gaussian Random Field (GRF) noise to q and p.
    4) Apply a low-pass Fourier filter to denoise the signals.
    5) Compute dq/dt and dp/dt using a 5-point finite difference stencil.
    6) Finally, discard the first 15% of time samples and the last 15%, returning the middle 70%.

    Returns
    -------
    q_denoised : ndarray
        Filtered position array (after discarding first/last 15%).
    p_denoised : ndarray
        Filtered momentum array (after discarding first/last 15%).
    dqdt : ndarray
        Derivative of q_denoised (5-point FD), middle 70%.
    dpdt : ndarray
        Derivative of p_denoised (5-point FD), middle 70%.
    t_eval : ndarray
        Time points, middle 70%.
    """
    # 1) Generate 30% more samples than usual
    overshoot_timescale = int(np.ceil(timescale * 1.3))

    # Build time array
    total_time = t_span[1] - t_span[0]
    num_points = overshoot_timescale * total_time
    # Because total_time might not be an integer, ensure integer sampling
    num_points = int(np.round(num_points))  # final integer

    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    if seed is not None:
        np.random.seed(seed)

    # Initial condition
    if y0 is None:
        y0 = np.random.rand(2) * 2 - 1
    if radius is None:
        radius = np.random.rand() * 0.9 + 0.1
    y0 = y0 / np.sqrt((y0**2).sum()) * radius

    # Solve IVP with the chosen dynamics
    # In your original code, you used "fun=dynamics_fn, rtol=1e-10, etc."
    # We'll replicate that below, but pass extras from kwargs as well.
    sol = solve_ivp(
        fun=dynamics_fn,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        rtol=1e-10,
        **kwargs
    )

    q = sol.y[0]
    p = sol.y[1]
    # Also get the exact derivatives from the ODE for reference if you want them
    # But you had that in your original code. We'll skip it unless needed.

    # 2) Generate correlated noise for q, p
    grf_q = generate_gaussian_random_field_1d(num_points, alpha=alpha, seed=seed)
    grf_p = generate_gaussian_random_field_1d(num_points, alpha=alpha, seed=seed + 1 if seed is not None else None)

    q_noisy = q + grf_q * noise_std
    p_noisy = p + grf_p * noise_std

    # 3) Low-pass filter
    q_denoised = fourier_filter(q_noisy, keep_frequency=keep_frequency)
    p_denoised = fourier_filter(p_noisy, keep_frequency=keep_frequency)

    # 4) Compute derivatives (5-point FD)
    dt = t_eval[1] - t_eval[0]
    dqdt = five_point_derivative(q_denoised, dt)
    dpdt = five_point_derivative(p_denoised, dt)

    # 5) Discard first 15% and last 15% => keep middle 70%
    N = len(t_eval)
    offset = int(0.15 * N)  # 15% offset
    start_ix = offset
    end_ix = N - offset  # up to but not including end_ix

    # In case timescale is too small, ensure offset < end_ix
    if start_ix >= end_ix:
        raise ValueError(
            "Not enough points to remove 15% from start and end. "
            f"Got N={N}, offset={offset}."
        )

    q_return = q_denoised[start_ix:end_ix]
    p_return = p_denoised[start_ix:end_ix]
    dqdt_return = dqdt[start_ix:end_ix]
    dpdt_return = dpdt[start_ix:end_ix]
    t_return = t_eval[start_ix:end_ix]

    return q_return, p_return, dqdt_return, dpdt_return, t_return


def get_dataset(seed=0, samples=50, test_split=0.5, **kwargs):
    """
    Generates 'samples' trajectories, each with the get_trajectory() pipeline.
    Then concatenates them, and splits into train/test sets at 'test_split' ratio.
    """
    data = {'meta': locals()}

    np.random.seed(seed)
    xs, dxs = [], []
    for s in range(samples):
        # get_trajectory now returns the middle 70% after overshoot
        x, y, dx, dy, t = get_trajectory(**kwargs)
        xs.append(np.stack([x, y], axis=-1))   # shape (N_i, 2)
        dxs.append(np.stack([dx, dy], axis=-1)) # shape (N_i, 2)

    # Concatenate all samples into a single array
    xs = np.concatenate(xs, axis=0)   # shape (sum(N_i), 2)
    dxs = np.concatenate(dxs, axis=0) # shape (sum(N_i), 2)

    data['x'] = xs
    data['dx'] = dxs.squeeze()

    # Make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k] = data[k][:split_ix]
        split_data['test_' + k] = data[k][split_ix:]

    data = split_data
    return data


def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    """
    Returns a simple vector field grid for demonstration or plotting.
    """
    field = {'meta': locals()}

    b, a = np.meshgrid(
        np.linspace(xmin, xmax, gridsize),
        np.linspace(ymin, ymax, gridsize)
    )
    ys = np.stack([b.flatten(), a.flatten()])  # shape (2, gridsize^2)

    # Evaluate the dynamics at each point
    dydt = [dynamics_fn(None, y) for y in ys.T]
    dydt = np.stack(dydt).T  # shape (2, gridsize^2)

    field['x'] = ys.T
    field['dx'] = dydt.T
    return field
