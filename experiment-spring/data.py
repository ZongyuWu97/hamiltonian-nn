import autograd
import autograd.numpy as np
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

##########################################################
# Helper 1: Central finite difference on a 1D array
##########################################################
def finite_difference(x, t):
    """
    Compute dx/dt for a 1D array x(t) using central differences.
    x: shape [T,] (values of x at each time step)
    t: shape [T,] (corresponding time array, assumed uniform spacing)
    Returns: shape [T,] the finite-difference approximation to dx/dt
    """
    dx = np.zeros_like(x)
    dt = t[1] - t[0]  # assume uniform spacing
    # central difference for interior points
    dx[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    # forward/backward difference for edges
    dx[0]    = (x[1] - x[0]) / dt
    dx[-1]   = (x[-1] - x[-2]) / dt
    return dx

##########################################################
# Helper 2: Simple Fourier denoising
##########################################################
def fourier_denoise(x, keep_frequencies=10):
    """
    Denoise a 1D signal x by zeroing out high-frequency components.
    Uses rFFT -> zero out beyond 'keep_frequencies' -> iFFT.

    x: shape [T,]
    keep_frequencies: int, number of low-frequency components to keep
    returns: shape [T,] the denoised signal
    """
    # Forward real FFT
    X = np.fft.rfft(x)
    # Zero out everything beyond keep_frequencies
    if keep_frequencies < len(X):
        X[keep_frequencies:] = 0
    # Inverse real FFT
    x_denoised = np.fft.irfft(X, n=len(x))
    return x_denoised


##########################################################
# Hamiltonian definitions
##########################################################
def hamiltonian_fn(coords):
    # coords is [q, p]
    q, p = np.split(coords, 2)
    H = q**2 + p**2
    return H

def dynamics_fn(t, coords):
    # For H(q,p) = q^2 + p^2,
    # dH/dq = 2q, dH/dp = 2p
    # dq/dt = dH/dp = 2p
    # dp/dt = -dH/dq = -2q
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = np.split(dcoords, 2)   # dqdt=2q, dpdt=2p
    S = np.concatenate([dpdt, -dqdt])   # [2p, -2q]
    return S

##########################################################
# get_trajectory() with 3 modes
##########################################################
def get_trajectory(t_span=[0, 3],
                   timescale=100,
                   radius=None,
                   y0=None,
                   noise_std=0.0,
                   mode=1,
                   keep_frequencies=10,
                   **kwargs):
    """
    mode=1: Return true (q, p) from ODE solver + true (dq/dt, dp/dt) from the dynamics_fn. No noise.
    mode=2: Discard p_true, optionally add noise to q. Then approximate p, dq/dt, dp/dt from finite differences.
    mode=3: Same as mode=2, but first denoise the noisy q(t) by Fourier series/FFT,
            then do finite differences on the denoised signal.
    """
    # 1) Time array
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))

    # 2) Generate a valid initial condition for (q, p)
    if y0 is None:
        y0 = np.random.rand(2)*2 - 1  # random direction
    if radius is None:
        radius = np.random.rand()*0.9 + 0.1  # random radius
    y0 = y0 / np.sqrt((y0**2).sum()) * radius

    # 3) Solve the true Hamiltonian system to get (q_true, p_true)
    ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval,
                    rtol=1e-10, **kwargs)
    q_true = ivp.y[0]  # shape [T,]
    p_true = ivp.y[1]  # shape [T,]

    # --------------------------------
    # MODE 1
    # --------------------------------
    if mode == 1:
        # (q_true, p_true) with no noise, plus exact time derivatives from the dynamics.
        # Let's compute dqdt_true, dpdt_true by calling dynamics_fn at each time.
        coords = np.stack([q_true, p_true], axis=-1)  # shape [T,2]
        dcoords = []
        for i in range(len(t_eval)):
            dcdt = dynamics_fn(t_eval[i], coords[i])  # shape [2,]
            dcoords.append(dcdt)
        dcoords = np.array(dcoords)  # shape [T, 2]
        dqdt_true = dcoords[:, 0]
        dpdt_true = dcoords[:, 1]
        return q_true, p_true, dqdt_true, dpdt_true, t_eval

    # --------------------------------
    # MODE 2
    # --------------------------------
    elif mode == 2:
        # Discard p_true, optionally add noise to q_true
        q_noisy = q_true + np.random.randn(len(q_true)) * noise_std if noise_std > 0 else q_true

        # Approximate p(t)=dq/dt, and dp/dt from finite differences
        dqdt_approx = finite_difference(q_noisy, t_eval)
        p_approx    = dqdt_approx
        dpdt_approx = finite_difference(p_approx, t_eval)

        return q_noisy, p_approx, dqdt_approx, dpdt_approx, t_eval

    # --------------------------------
    # MODE 3
    # --------------------------------
    elif mode == 3:
        # Same as mode 2, but first denoise q_noisy with a Fourier representation
        q_noisy = q_true + np.random.randn(len(q_true)) * noise_std if noise_std > 0 else q_true
        p_noisy = p_true + np.random.randn(len(p_true)) * noise_std if noise_std > 0 else p_true
        # Fourier denoise
        q_denoised = fourier_denoise(q_noisy, keep_frequencies=keep_frequencies)
        p_denoised = fourier_denoise(p_noisy, keep_frequencies=keep_frequencies)

        coords = np.stack([q_denoised, p_denoised], axis=-1)  # shape [T,2]
        dcoords = []
        for i in range(len(t_eval)):
            dcdt = dynamics_fn(t_eval[i], coords[i])  # shape [2,]
            dcoords.append(dcdt)
        dcoords = np.array(dcoords)  # shape [T, 2]
        dqdt_true = dcoords[:, 0]
        dpdt_true = dcoords[:, 1]
        
        return q_denoised, p_denoised, dqdt_true, dpdt_true, t_eval
    
    else:
        raise ValueError("Invalid mode: {}".format(mode))




##########################################################
# get_dataset(): randomly sample multiple trajectories
#                (now uses the 3-mode get_trajectory)
##########################################################
def get_dataset(seed=0, samples=50, test_split=0.5, mode=1, **kwargs):
    """
    Collect multiple trajectories (q(t), p(t)) plus their derivatives
    (dq/dt, dp/dt). The shape of x and dx depends on 'mode':
       mode=1 -> x=[q, p], dx=[dqdt, dpdt] (true ODE solution)
       mode=2 -> x=[q, p_approx], dx=[dqdt_approx, dpdt_approx] (finite diff from position)
       mode=3 -> x=[q_denoised, p_approx], dx=[dqdt_approx, dpdt_approx] (fourier-smoothed, then finite diff)
    """
    data = {'meta': locals()}
    np.random.seed(seed)

    xs, dxs = [], []
    for s in range(samples):
        # each get_trajectory() returns (q, p, dq, dp, t_eval)
        q, p, dq, dp, t_eval = get_trajectory(mode=mode, **kwargs)
        # shape [T,] for each

        # stack them so that x = [q, p], dx = [dq, dp]
        # shape [T, 2]
        x_traj  = np.stack([q, p],  axis=1)
        dx_traj = np.stack([dq, dp], axis=1)

        xs.append(x_traj)
        dxs.append(dx_traj)

    data['x']  = np.concatenate(xs,  axis=0)  # shape [T*samples, 2]
    data['dx'] = np.concatenate(dxs, axis=0)  # shape [T*samples, 2]

    # train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k]             = data[k][:split_ix]
        split_data['test_' + k]   = data[k][split_ix:]
    data = split_data

    return data

##########################################################
# get_field(): sample a grid in (q,p)-space and compute
#              the "true" Hamiltonian vector field
##########################################################
def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    field = {'meta': locals()}

    b, a = np.meshgrid(
        np.linspace(xmin, xmax, gridsize),
        np.linspace(ymin, ymax, gridsize)
    )
    ys = np.stack([b.flatten(), a.flatten()])  # shape [2, gridsize^2]

    # get vector directions via the same true dynamics
    dydt = [dynamics_fn(None, y) for y in ys.T]  # each is shape (2,)
    dydt = np.stack(dydt).T  # shape [2, gridsize^2]

    field['x']  = ys.T      # shape [gridsize^2, 2]
    field['dx'] = dydt.T    # shape [gridsize^2, 2]
    return field
