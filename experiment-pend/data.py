# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

##########################################################
# Helper: Central finite difference on a 1D array
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
# Unchanged: Defines the Hamiltonian H(q,p) for the pendulum
##########################################################
def hamiltonian_fn(coords):
    q, p = np.split(coords, 2)
    H = 6 * (1 - np.cos(q)) + p**2 / 2  # pendulum Hamiltonian
    return H

##########################################################
# Unchanged: Uses Autograd to get true dH/dq, dH/dp for ODE integration
##########################################################
def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = np.split(dcoords, 2)
    S = np.concatenate([dpdt, -dqdt], axis=-1)
    return S

##########################################################
# MODIFIED get_trajectory():
#   1) Solve the ODE to get (q_true, p_true).
#   2) Discard p_true and keep only q_true.
#   3) Add noise to q_true.
#   4) Approximate p(t), dq/dt, dp/dt via finite differences.
#   5) Return them in the same shape as before.
##########################################################
def get_trajectory(t_span=[0,3], timescale=100, radius=None, y0=None, noise_std=0.2, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))

    # 1) Generate a valid initial condition for (q, p)
    if y0 is None:
        y0 = np.random.rand(2)*2 - 1
    if radius is None:
        radius = np.random.rand() + 1.3  # sample a range of radii
    y0 = y0 / np.sqrt((y0**2).sum()) * radius  # normalize to desired radius

    # 2) Solve the "true" Hamiltonian system to get (q_true, p_true)
    pendulum_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0,
                             t_eval=t_eval, rtol=1e-10, **kwargs)

    q_true = pendulum_ivp['y'][0]  # shape [T,]
    # p_true = pendulum_ivp['y'][1]  # we won't use p_true

    q_noisy = q_true

    # 4) Approximate p, dq/dt, dp/dt from q_noisy via finite differences
    dqdt_approx = finite_difference(q_noisy, t_eval)      # partial q / partial t
    p_approx    = dqdt_approx                      # from q' = p (for small oscillations)
    dpdt_approx = finite_difference(p_approx, t_eval)    # partial p / partial t

    # 3) Add noise only to q
    q_noisy = q_true + np.random.randn(*q_true.shape) * noise_std

    # 5) Return: q_noisy, p_approx, dqdt_approx, dpdt_approx, t_eval
    return q_noisy, p_approx, dqdt_approx, dpdt_approx, t_eval

##########################################################
# Unchanged: randomly sample multiple trajectories
#            but now it uses the modified get_trajectory()
##########################################################
def get_dataset(seed=0, samples=100, test_split=0.5, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    for s in range(samples):
        # note that get_trajectory() now returns
        #    (q, p, dqdt, dpdt, t_eval)
        x, y, dx, dy, t = get_trajectory(**kwargs)
        xs.append(np.stack([x, y]).T)      # shape [T, 2]
        dxs.append(np.stack([dx, dy]).T)   # shape [T, 2]

    data['x'] = np.concatenate(xs)   # shape [T*samples, 2]
    data['dx'] = np.concatenate(dxs) # shape [T*samples, 2]

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data

##########################################################
# Unchanged: sample a grid in (q,p)-space and compute
#            the "true" Hamiltonian vector field
##########################################################
def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize),
                       np.linspace(ymin, ymax, gridsize))
    ys = np.stack([b.flatten(), a.flatten()])

    # get vector directions via the same true dynamics
    dydt = [dynamics_fn(None, y) for y in ys.T]
    dydt = np.stack(dydt).T

    field['x'] = ys.T
    field['dx'] = dydt.T
    return field
