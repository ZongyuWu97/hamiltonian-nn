# Kolmogorov–Arnold Representation Theorem-Based Hamiltonian Neural Network

## Usage

First install the dependencies with
`conda env create --name yourenvname --file=environments.yml`

To train a Hamiltonian Neural Network (HNN):

- Task 1: Ideal mass-spring system: `python3 experiment-spring/train.py --verbose`
- Task 2: Ideal pendulum: `python3 experiment-pend/train.py --verbose`
- Task 3: Two-body problem: `python3 experiment-2body/train.py --verbose`
- Task 4: Three-body problem: `python3 experiment-3body/train.py --verbose`

To train a Baseline model:
Add `--baseline` to the commands above.

To train a KAR-HNN model:
Find the `train-task.ipynb` file in the corresponding `experiment-task` folder. For example `train-spring.ipynb` in `experiment-spring`.

To analyze results

- Task 1: Ideal mass-spring system: [`analyze-spring.ipnyb`](analyze-spring.ipynb)
- Task 2: Ideal pendulum: [`analyze-pend.ipnyb`](analyze-pend.ipynb)
- Task 3: Two-body problem: [`analyze-2body.ipnyb`](analyze-2body.ipynb)
- Task 4: Three-body problem: [`analyze-3body.ipnyb`](analyze-3body.ipynb)
