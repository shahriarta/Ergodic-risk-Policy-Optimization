import numpy as np
import scipy.linalg as la
import random
import matplotlib.pyplot as plt
import matplotlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

# matplotlib.use('Agg')
np.random.seed(1234)

# Function to calculate the LQR gain
def lqr(A, B, Q, R):
    P = la.solve_discrete_are(A, B, Q, R)
    K = -np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K, P

# Function to calculate Sigma (solution to the Lyapunov equation)
def Sigma(K, A, B, H, Sigma_W):
    return la.solve_discrete_lyapunov(A_cl(K, A, B), H @ Sigma_W @ H.T)

# Function to calculate the closed-loop dynamics matrix
def A_cl(K, A, B):
    return A + B @ K

# Function to calculate the performance index
def J(K, A, B, H, Sigma_W, Q, R):
    Q_K = Q + K.T @ R @ K
    return np.trace(Q_K @ Sigma(K, A, B, H, Sigma_W))

# Function to calculate the asymptotic conditional variance
def gamma_N_sq(Sigma_K, Q_c, H, Sigma_W):
    return 4 * np.trace(Q_c @ H @ Sigma_W @ H.T @ Q_c @ (Sigma_K - H @ Sigma_W @ H.T))

# Function to calculate P_lambda
def P_lambda(K, lmbda, A, B, H, Sigma_W, Q, R, Q_c):
    return la.solve_discrete_lyapunov(A_cl(K, A, B).T, Q + K.T @ R @ K + 4 * lmbda * Q_c @ H @ Sigma_W @ H.T @ Q_c)

# Function to calculate the gradient of K
def calculate_grad_K(K, lmbda, A, B, H, Sigma_W, Q, R, Q_c):
    P_lambda_opt = P_lambda(K, lmbda, A, B, H, Sigma_W, Q, R, Q_c)
    grad_K = 2 * (R @ K + B.T @ P_lambda_opt @ A_cl(K, A, B)) @ Sigma(K, A, B, H, Sigma_W)
    return grad_K

# Function to calculate the Newton direction
def Newton_dir(K, lmbda, A, B, H, Sigma_W, Q, R, Q_c):
    P_lambda_K = P_lambda(K, lmbda, A, B, H, Sigma_W, Q, R, Q_c)
    G = np.linalg.solve((R + B.T @ P_lambda_K @ B), (R @ K + B.T @ P_lambda_K @ A_cl(K, A, B)))
    return G

# Limiting Risk LQR Algorithm
def limiting_risk_lqr(A, B, H, Sigma_W, Q, R, Q_c, R_c, K0, lmbda0, beta, epsilon=1e-6, max_iters=100000, eta=1e-1):
    try:
        lmbda = [lmbda0]
        K = [K0]
        Sigma_K0 = Sigma(K[-1], A, B, H, Sigma_W)

        grad_norms = []
        constraint_violations = []
        comp_slacks = []

        for iter in range(max_iters):
            while True:
                G = Newton_dir(K[-1], lmbda[-1], A, B, H, Sigma_W, Q, R, Q_c)
                K.append(K[-1] - G)
                if np.max(np.abs(la.eigvals(A_cl(K[-1], A, B)))) >= 1:
                    break
                if la.norm(G) < epsilon:
                    break

            Sigma_K = Sigma(K[-1], A, B, H, Sigma_W)
            grad_K = calculate_grad_K(K[-1], lmbda[-1], A, B, H, Sigma_W, Q, R, Q_c)
            grad_lmbda = gamma_N_sq(Sigma_K, Q_c, H, Sigma_W) - beta
            comp_slack = lmbda[-1] * (gamma_N_sq(Sigma_K, Q_c, H, Sigma_W) - beta)

            grad_norm = la.norm(grad_K) / la.norm(calculate_grad_K(K0, lmbda0, A, B, H, Sigma_W, Q, R, Q_c))
            constraint_violation = np.abs(grad_lmbda) / np.abs(gamma_N_sq(Sigma_K0, Q_c, H, Sigma_W) - beta)
            comp_slack_norm = np.abs(comp_slack) / np.abs(lmbda0 * (gamma_N_sq(Sigma_K0, Q_c, H, Sigma_W) - beta))

            grad_norms.append(grad_norm)
            constraint_violations.append(constraint_violation)
            comp_slacks.append(comp_slack_norm)

            if np.abs(comp_slack) > epsilon or la.norm(grad_K) > epsilon or np.abs(grad_lmbda) > epsilon:
                lmbda.append(max(0, lmbda[-1] + eta / (np.sqrt(1 + iter) * gamma_N_sq(Sigma_K0, Q_c, H, Sigma_W)) * (gamma_N_sq(Sigma_K, Q_c, H, Sigma_W) - beta)))
            else:
                break

        return grad_norms, constraint_violations, comp_slacks

    except Exception as e:
        print(f"Error in simulation: {e}")
        traceback.print_exc()
        return None

# Run a single simulation with random parameters
def run_single_simulation():
    try:
        n = 4
        m = 3
        d = 4
        A = np.random.randn(n, n)
        B = np.random.randn(n, m)
        H = np.random.randn(n, d)
        Sigma_W = np.eye(d)
        Q = np.eye(n)
        R = np.eye(m)
        Q_c = np.eye(n)
        R_c = np.eye(m)

        K0, P = lqr(A, B, Q, R)
        lmbda0 = 1
        beta = 0.9 * gamma_N_sq(Sigma(K0, A, B, H, Sigma_W), Q_c, H, Sigma_W)

        return limiting_risk_lqr(A, B, H, Sigma_W, Q, R, Q_c, R_c, K0, lmbda0, beta, max_iters=10000)
    
    except Exception as e:
        print(f"Error in run_single_simulation: {e}")
        traceback.print_exc()
        return None

# Run multiple simulations in parallel
def run_multiple_simulations_in_parallel(n_runs=1, max_workers=None):
    grad_norms_all = []
    constraint_violations_all = []
    comp_slacks_all = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_simulation) for _ in range(n_runs)]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                grad_norms, constraint_violations, comp_slacks = result
                grad_norms_all.append(grad_norms)
                constraint_violations_all.append(constraint_violations)
                comp_slacks_all.append(comp_slacks)

    return grad_norms_all, constraint_violations_all, comp_slacks_all

# Plot the progress over iterations
def plot_progress_over_iterations(grad_norms_all, constraint_violations_all, comp_slacks_all):
    max_iters = min(map(len, grad_norms_all))
    grad_norms_array = np.array([run[:max_iters] for run in grad_norms_all])
    constraint_violations_array = np.array([run[:max_iters] for run in constraint_violations_all])
    comp_slacks_array = np.array([run[:max_iters] for run in comp_slacks_all])

    iterations = np.arange(max_iters)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].plot(iterations, np.median(grad_norms_array, axis=0), label='Median')
    axs[0].fill_between(iterations, np.min(grad_norms_array, axis=0), np.max(grad_norms_array, axis=0), alpha=0.3, label='Range')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Normalized Gradient Norm')
    axs[0].set_title(r'$\overline{\|\nabla_K L(K^*(\lambda_t),\lambda_t)\|}$')
    axs[0].set_yscale('log')
    axs[0].legend()

    axs[1].plot(iterations, np.median(constraint_violations_array, axis=0), label='Median', color='orange')
    axs[1].fill_between(iterations, np.min(constraint_violations_array, axis=0), np.max(constraint_violations_array, axis=0), alpha=0.3, label='Range', color='orange')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Normalized Constraint Violation')
    axs[1].set_title(r'$\overline{|\nabla_\lambda L(K^*(\lambda_t),\lambda_t)|}$')
    axs[1].set_yscale('log')
    axs[1].legend()

    axs[2].plot(iterations, np.median(comp_slacks_array, axis=0), label='Median', color='green')
    axs[2].fill_between(iterations, np.min(comp_slacks_array, axis=0), np.max(comp_slacks_array, axis=0), alpha=0.3, label='Range', color='green')
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('Normalized Complementary Slackness')
    axs[2].set_title(r'$\overline{|\mathrm{CS}(\lambda_t)|}$')
    axs[2].set_yscale('log')
    axs[2].legend()

    # Change the fontsize and legend size
    for ax in axs:
        ax.set_ylabel('')
        ax.title.set_fontsize(40)
        for item in ([ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(35)
        ax.legend(prop={'size': 25})
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))

    plt.tight_layout(pad=3.0)
    # plt.show()

    # save it as vector form for latex use
    fig.savefig('risk_lqr_parallel.pdf', format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    grad_norms_all, constraint_violations_all, comp_slacks_all = run_multiple_simulations_in_parallel(n_runs=100, max_workers=None)
    plot_progress_over_iterations(grad_norms_all, constraint_violations_all, comp_slacks_all)
