import numpy as np
import scipy.linalg as la
import random
import matplotlib.pyplot as plt
from scipy.stats import t
import control

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

def generate_normalized_t_distribution(nu, size=10000):
    """
    Generates i.i.d. samples from the Student's t-distribution with degrees of freedom `nu`,
    normalized to have zero mean and identity variance.
    
    Parameters:
    - nu: Degrees of freedom for the Student's t-distribution (ν > 2 for finite variance)
    - size: Number of samples to generate
    
    Returns:
    - normalized_samples: Samples with zero mean and identity variance
    """
    # Generate raw samples from the Student's t-distribution
    samples = t.rvs(df=nu, size=size)
    
    # The variance of t-distribution is Var = ν / (ν - 2), so we scale the samples
    scaling_factor = np.sqrt((nu - 2) / nu)
    
    # Scale the samples to have identity variance
    normalized_samples = samples * scaling_factor
 
    return normalized_samples


def lqr(A, B, Q, R):
    """
    Solve the Continuous-Time Linear Quadratic Regulator (LQR) problem for discrete-time system.
    A: System dynamics matrix
    B: Input matrix
    Q: State cost matrix
    R: Control cost matrix
    """
    P = la.solve_discrete_are(A, B, Q, R)
    K = -np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K, P

def Sigma(K):
    """
    Calculates the solution to the discrete Lyapunov equation using the given controller gain matrix.

    Parameters:
    - K: Controller gain matrix

    Returns:
    - Sigma: Solution to the discrete Lyapunov equation
    """
    return la.solve_discrete_lyapunov(A_cl(K), H @ Sigma_W @ H.T)

def A_cl(K):
    """
    Calculates the closed-loop dynamics matrix given the controller gain matrix K.

    Parameters:
    - K: Controller gain matrix

    Returns:
    - A_cl: Closed-loop dynamics matrix
    """
    return A + B @ K

def J(K):
    """
    Calculates the performance index J(K) for a given controller gain matrix K.

    Parameters:
    - K: Controller gain matrix

    Returns:
    - J(K): Performance index
    """
    # Calculate J(K) = tr(Q_K * Sigma_K)
    Q_K = Q + K.T @ R @ K
    return np.trace(Q_K @ Sigma(K))

def gamma_N_sq(Sigma_K):
    """
    Computes the asymptotic conditional variance gamma_N^2(K) for a given Sigma_K.

    Parameters:
    - Sigma_K: Solution to the discrete Lyapunov equation

    Returns:
    - gamma_N_sq: Asymptotic conditional variance
    """
    # Compute the asymptotic conditional variance gamma_N^2(K)
    return 4 * np.trace(Q_c @ H @ Sigma_W @ H.T @ Q_c @ (Sigma_K - H @ Sigma_W @ H.T))

def P_lambda(K, lmbda):
    """
    Calculates the matrix P_lambda for a given controller gain matrix K and lambda.

    Parameters:
    - K: Controller gain matrix
    - lmbda: Lambda value

    Returns:
    - P_lambda: Matrix P_lambda
    """
    return la.solve_discrete_lyapunov(A_cl(K).T, Q + K.T @ R @ K + 4 * lmbda * Q_c @ H @ Sigma_W @ H.T @ Q_c)

def calculate_grad_K(K, lmbda):
    """
    Calculates the gradient of K for a given controller gain matrix K and lambda.

    Parameters:
    - K: Controller gain matrix
    - lmbda: Lambda value

    Returns:
    - grad_K: Gradient of K
    """
    P_lambda_opt = P_lambda(K, lmbda)
    grad_K = 2 * (R @ K + B.T @ P_lambda_opt @ A_cl(K)) @ Sigma(K)
    return grad_K

def Newton_dir(K, lmbda):
    """
    Calculates the Newton direction G for a given controller gain matrix K and lambda.

    Parameters:
    - K: Controller gain matrix
    - lmbda: Lambda value

    Returns:
    - G: Newton direction
    """
    P_lambda_K = P_lambda(K, lmbda)
    G = np.linalg.solve((R + B.T @ P_lambda_K @ B),(R @ K + B.T @ P_lambda_K @ A_cl(K)))
    return G

def simulate_response(A, B, K, x0, num_steps=100):
    """
    Simulates the response of the closed-loop system with a given controller K.

    Parameters:
    - A: System dynamics matrix
    - B: Input matrix
    - K: Controller gain matrix
    - x0: Initial state vector
    - num_steps: Number of time steps to simulate

    Returns:
    - X: State trajectory (each column is the state at a given time step)
    """
    n = A.shape[0]
    X = np.zeros((n, num_steps))
    X[:, 0] = x0

    for t in range(1, num_steps):
        u = K @ X[:, t-1]  # Control input
        # w = np.random.multivariate_normal(np.zeros(d), Sigma_W)  # Sample noise
        
        # Set degrees of freedom ν (must be greater than 2 for finite variance)
        nu = 5
        # Generate normalized samples
        w = generate_normalized_t_distribution(nu, size=n)
        if False:#t%20==0:
            X[:, t] = A @ X[:, t-1] + B @ u + H @ w + 50*np.ones(n)   # State update
        else:
            X[:, t] = A @ X[:, t-1] + B @ u + H @ w   # State update
    return X

def simulate_response_kicks(A, B, K, x0, w, num_steps=100):
    """
    Simulates the response of the closed-loop system with a given controller K.

    Parameters:
    - A: System dynamics matrix
    - B: Input matrix
    - K: Controller gain matrix
    - x0: Initial state vector
    - w: noise simulating kicks
    - num_steps: Number of time steps to simulate

    Returns:
    - X: State trajectory (each column is the state at a given time step)
    """
    n = A.shape[0]
    X = np.zeros((n, num_steps))
    X[:, 0] = x0

    for t in range(1, num_steps):
        u = K @ X[:, t-1]  # Control input
        
        if False:#t%20==0:
            X[:, t] = A @ X[:, t-1] + B @ u + H @ w + 50*np.ones(n)   # State update
        else:
            X[:, t] = A @ X[:, t-1] + B @ u + H @ w[t]   # State update
    return X


def compute_S_t(A, B, H, Sigma_W, Q_c, K, X_traj):
    """
    Computes the matrix S_t for a given controller gain matrix K.

    Parameters:
    - A: System dynamics matrix
    - B: Input matrix
    - H: Output matrix
    - Sigma_W: Covariance matrix of the noise
    - K: Controller gain matrix

    Returns:
    - S_t: Matrix S_t
    """
    # Compute the closed-loop dynamics matrix
    A_K = A + B @ K
    # Compute Sigma_K = Sigma(K)
    Sigma_K = la.solve_discrete_lyapunov(A_K, H @ Sigma_W @ H.T)

    S = []
    Gamma = [X_traj[:,0] @ X_traj[:,0].T]

    for t in range(1, X_traj.shape[1]):
        Gamma.append(Gamma[t-1] + np.outer(X_traj[:,t], X_traj[:,t]))
        S.append(np.trace((Q_c - A_K.T @ Q_c @ A_K) @ (Gamma[t] - t * Sigma_K)) + np.trace(A_K.T @ Q_c @ A_K @ (np.outer(X_traj[:,t], X_traj[:,t]) - np.outer(X_traj[:,0], X_traj[:,0]))))

    return S

def compare_controllers_response(A, B, K_LQR, K_final, x0, num_steps=1000):
    """
    Compares the closed-loop response of the LQR optimal controller and the final limiting-risk controller.

    Parameters:
    - A: System dynamics matrix
    - B: Input matrix
    - K_LQR: Optimal LQR controller gain matrix
    - K_final: Final limiting-risk controller gain matrix
    - x0: Initial state vector
    - num_steps: Number of time steps to simulate
    """
    # Simulate the response for both controllers
    X_LQR = simulate_response(A, B, K_LQR, x0, num_steps)
    X_final = simulate_response(A, B, K_final, x0, num_steps)

    # Plot the state trajectories for each state
    fig, ax = plt.subplots(figsize=(10, 8))

    # get S for LQR
    S_LQR = compute_S_t(A, B, H, Sigma_W, Q_c, K_LQR, X_LQR)

    # get S for final
    S_final = compute_S_t(A, B, H, Sigma_W, Q_c, K_final, X_final)

    time = np.arange(1,num_steps)
    S_LQR = np.array(S_LQR)**2 / time
    S_final = np.array(S_final)**2 / time
    ax.plot(time, S_LQR, label='LQR Optimal', linestyle='--')
    ax.plot(time, S_final, label='Final Limiting-Risk', linestyle='-')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('S / t')
    ax.legend(prop={'size': 40}, loc='upper left', bbox_to_anchor=(0.65, 1.15))
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def limiting_risk_lqr_with_plots(A, B, H, Sigma_W, Q, R, Q_c, R_c, K0, lmbda0, beta, epsilon=1e-6, max_iters=100000, eta=1e1):
    """
    Primal-Dual Limiting-Risk-Aware LQR Algorithm with plotting routine.
    """
    # Initialize K and lambda
    lmbda = []
    K = []
    K.append(K0)
    lmbda.append(lmbda0)
    Sigma_K0 = Sigma(K[-1])
    
    grad_norms = []
    constraint_violations = []
    comp_slacks = []

    # Perform primal-dual iterations
    for iter in range(max_iters):
        
        # Gradient descent step for K
        while True:
            # Compute the quasi-Newton descent iteration
            G = Newton_dir(K[-1], lmbda[-1])
            K.append(K[-1] - G)
            if np.max(np.abs(la.eigvals(A_cl(K[-1])))) >= 1:
                print('Unstable controller')
                break

            if la.norm(G) < epsilon:
                break

        # Compute metrics
        Sigma_K = Sigma(K[-1])
        grad_K = calculate_grad_K(K[-1], lmbda[-1])
        grad_lmbda = gamma_N_sq(Sigma_K) - beta
        comp_slack = lmbda[-1] * (gamma_N_sq(Sigma_K) - beta)
        
        grad_norm = la.norm(grad_K) / la.norm(calculate_grad_K(K0, lmbda0))
        constraint_violation = np.abs(grad_lmbda) / np.abs(gamma_N_sq(Sigma(K0)) - beta)
        comp_slack_norm = np.abs(comp_slack) / np.abs(lmbda0 * (gamma_N_sq(Sigma_K0) - beta))
        
        grad_norms.append(grad_norm)
        constraint_violations.append(constraint_violation)
        comp_slacks.append(comp_slack_norm)

        # Check constraint violation
        if np.abs(comp_slack) > epsilon or la.norm(grad_K) > epsilon or np.abs(grad_lmbda) > epsilon:
            lmbda.append( max(0, lmbda[-1] + eta/(np.sqrt(1+iter)*gamma_N_sq(Sigma_K0)) * (gamma_N_sq(Sigma_K) - beta)) )
        else:
            print(f"Iteration {iter+1}: J(K) = {J(K[-1])}, lambda = {lmbda[-1]}")
            break
        
        if (iter+1) % 1000 == 1:
            print(f"Iteration {iter+1}: J(K) = {J(K[-1])}, lambda = {lmbda[-1]}")

    # Plotting the results
    iterations = np.arange(len(grad_norms))

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Normalized la.norm(grad_K)/la.norm(grad_K0)
    axs[0].plot(iterations, grad_norms)
    axs[0].set_xlabel('Iteration')
    axs[0].set_title(r'$\overline{\|\nabla_K L(K^*(\lambda_t),\lambda_t)\|}$')
    axs[0].set_yscale('log')  # Set y-axis to log scale

    # Plot 2: Normalized constraint violation grad_lmbda/grad_lmbda0
    axs[1].plot(iterations, constraint_violations, color='orange')
    axs[1].set_xlabel('Iteration')
    axs[1].set_title(r'$\overline{|\nabla_\lambda L(K^*(\lambda_t),\lambda_t)|}$')
    axs[1].set_yscale('log')  # Set y-axis to log scale

    # Plot 3: Normalized complementary slackness comp_slack/comp_slack0
    axs[2].plot(iterations, comp_slacks, color='green')
    axs[2].set_xlabel('Iteration')
    axs[2].set_title(r'$\overline{|\mathrm{CS}(\lambda_t)|}$')
    axs[2].set_yscale('log')  # Set y-axis to log scale
    # Change the fontsize and legend size
    for ax in axs:
        ax.title.set_fontsize(35)
        for item in ([ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(35)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.offsetText.set_fontsize(40)
        # Add legends outside of each subplot
        # ax.legend(loc='upper center', bbox_to_anchor=(0, 1.2), ncol=3, fancybox=True, shadow=True, fontsize=40)  # Legend outside the first subplot

    # add padding between the plots
    plt.tight_layout(pad=3.0)
    plt.show()

    # save it as vector form for latex use
    fig.savefig('risk_lqr_single_X29.pdf', format='pdf', bbox_inches='tight')

    return K, lmbda



def compute_C_t(A, B, H, Sigma_W, Q_c, K, X_traj):
    """
    Computes the sequence C_t for a given controller K.

    Parameters:
    - A: System dynamics matrix
    - B: Input matrix
    - H: Output matrix
    - Sigma_W: Covariance matrix of the noise
    - Q_c: Cost matrix
    - K: Controller gain matrix
    - X_traj: State trajectory (columns are state vectors over time)

    Returns:
    - C_t: Sequence of C_t values
    """
    A_K = A + B @ K  # Closed-loop dynamics
    trace_term = np.trace(Q_c @ H @ Sigma_W @ H.T)  # Constant term

    C_t_values = []

    for t in range(1, X_traj.shape[1]):
        x_t = X_traj[:, t]
        x_t_minus_1 = X_traj[:, t - 1]

        C_t = (x_t.T @ Q_c @ x_t) - (x_t_minus_1.T @ A_K.T @ Q_c @ A_K @ x_t_minus_1) - trace_term
        C_t_values.append(C_t)

    return np.array(C_t_values)






# Example usage with random parameters 
np.random.seed(1234)

# ############### Random example ###############
# n = 4
# m = 2
# d = 4
# A = np.random.randn(n, n)
# B = np.random.randn(n, m)
# dt = 1

# ############### Aircraft example ###############
# Longitudinal control of X-29 aircraft ND-PA
# Longitudinal control
A1 = [[ -0.4272e-01 , -0.8541e+01 , -0.4451   , -0.3216e+02 ],
        [ -0.7881e-03 , -0.5291     ,  0.9896   ,  0.1639e-09 ],
        [  0.4010e-03 ,  0.3542e+01 , -0.2228   ,  0.6150e-08 ],
        [  0.0        ,  0.0        ,  0.10e+01 ,  0.0        ]]

B1 = [[ -0.3385e-01 , -0.9386e-01 ,  0.4888e-02 ],
        [ -0.1028e-02 , -0.1297e-02 , -0.4054e-03 ],
        [  0.2718e-01 , -0.5744e-02 , -0.1351e-01 ],
        [  0.0        ,  0.0        ,  0.0        ]]

# Lateral-directional control
A2 = [[ -0.1817     ,  0.1496     , -0.9825 ,  0.1119     ],
        [ -0.3569e+01 , -0.1704e+01 ,  0.9045 , -0.5531e-06 ],
        [  0.1218e+01 , -0.8208e-01 , -0.1826 , -0.4630e-07 ],
        [  0.0        ,  0.1000e+01 ,  0.1513 ,  0.0        ]]

B2 = [[ -0.4327e-03 ,  0.3901e-03 ],
        [  0.3713     ,  0.5486e-01 ],
        [  0.2648e-01 , -0.1353e-01 ],
        [  0.0        ,  0.0        ]]



# Longitudinal control of X-29 aircraft ND-UA
# A3 = [[ -0.1170e-01 , -0.6050e+01 , -0.3139   , -0.3211e+02 ],
#         [ -0.1400e-03 , -0.8167     ,  0.9940   ,  0.2505e-10 ],
#         [  0.3213e-03 ,  0.1214e+02 , -0.4136   ,  0.3347e-08 ],
#         [  0.0        ,  0.0        ,  0.10e+01 ,  0.0        ]]

# B3 = [[ -0.6054e-01 , -0.1580     ,  0.1338e-01 ],
#         [ -0.8881e-03 , -0.3604e-02 , -0.5869e-03 ],
#         [  0.1345     , -0.8383e-01 , -0.4689e-01 ],
#         [  0.0        ,  0.0        ,  0.0        ]]

# # Lateral-directional control
# A4 = [[ -0.1596     ,  0.7150e-01 , -0.9974     , 0.4413e-01 ],
#         [ -0.1520e+02 , -0.2602e+01 ,  0.1106e+01 , 0.0        ],
#         [  0.6840e+01 , -0.1026     , -0.6375e-01 , 0.0        ],
#         [  0.0        ,  0.10e+01   ,  0.7168e-01 , 0.0        ]]

# B4 = [[ -0.5980e-03 ,  0.6718e-03 ],
#         [  0.1343e+01 ,  0.2345     ],
#         [  0.8974e-01 , -0.7097e-01 ],
#         [  0.0        ,  0.0        ]]

n = 8
m = 5
d = 8
A_cont = np.block([[np.array(A1), np.zeros((4, 4))], [np.zeros((4, 4)), np.array(A2)]])
B_cont = np.block([[np.array(B1), np.zeros((4,2))], [np.zeros((4,3)), np.array(B2)]])

# discrete-time system
C = np.zeros((n, n))
D = np.zeros((n, m))      
dt = 0.05
sys1 = control.StateSpace(A_cont, B_cont, C, D)
sysd = sys1.sample(dt)
A = np.asarray(sysd.A)
B = np.asarray(sysd.B)
# ############### End of system example ###############

H = np.block([[np.eye(4), np.zeros((4, 4))], [np.zeros((4, 4)), np.eye(4)]])

Sigma_W = np.eye(d)
Q = np.eye(n)
R = np.eye(m)
Q_c = np.eye(n)
R_c = np.eye(m)

# Initialize K and lambda
K0, P = lqr(A, B, Q, R)
lmbda0 = 1
beta = 0.8 * gamma_N_sq(Sigma(K0))

K, lmbda = limiting_risk_lqr_with_plots(A, B, H, Sigma_W, Q, R, Q_c, R_c, K0, lmbda0, beta)

K_optimal = K[-1]
lambda_optimal = lmbda[-1]

print("The performance:")
print("risk-constrained - LQR cost = ", J(K_optimal)- J(K0))
print("Optimal K:", K_optimal)
print("Optimal lambda:", lambda_optimal)
print("Constraint violation (if positive):", gamma_N_sq(Sigma(K_optimal))- beta)

grad_K = calculate_grad_K(K_optimal, lambda_optimal)
print("\|grad_K L(K,lmbda)\|:", la.norm(grad_K))

G = Newton_dir(K_optimal, lambda_optimal)
print("\|G\|:", la.norm(G))



## Compare the responses of the LQR controller and the final limiting-risk controller
# Optimal LQR controller
K_LQR, _ = lqr(A, B, Q, R)

# Final limiting-risk controller (using the output from the previous limiting-risk LQR function)
K_final = K[-1]  # Assuming this is the final controller from the limiting_risk_lqr_with_plots function

# Initial state for the simulation
x0 = np.random.randn(n)

# Compare the responses of the LQR controller and the final limiting-risk controller
num_steps = 2000
num_simulations = 1000
S_LQR_list = []
S_final_list = []
time = np.arange(1,num_steps)

for sim in range(num_simulations):
    X_LQR = simulate_response(A, B, K_LQR, x0, num_steps)
    X_final = simulate_response(A, B, K_final, x0, num_steps)
    S_LQR_run = compute_S_t(A, B, H, Sigma_W, Q_c, K_LQR, X_LQR)
    S_final_run = compute_S_t(A, B, H, Sigma_W, Q_c, K_final, X_final)
    
    S_LQR_list.append( np.array(S_LQR_run)**2 / time )
    S_final_list.append( np.array(S_final_run)**2 / time )

mean_S_LQR = np.mean(S_LQR_list, axis=0)
mean_S_final = np.mean(S_final_list, axis=0)
var_S_LQR = np.var(S_LQR_list, axis=0)
var_S_final = np.var(S_final_list, axis=0)



# Compute C_t for LQR controller
C_LQR = compute_C_t(A, B, H, Sigma_W, Q_c, K_LQR, X_LQR)
# Compute C_t for final limiting-risk controller
C_final = compute_C_t(A, B, H, Sigma_W, Q_c, K_final, X_final)



fig, ax = plt.subplots(1, 1, figsize=(20, 8))
# Plot 1: Mean S_t^2 / t
ax.plot(time, mean_S_LQR, label='LQR Optimal', linestyle='--', linewidth=2.5)
ax.plot(time, mean_S_final, label='Ergodic-risk Optimal', linestyle='-', linewidth=2.5)
ax.set_xlabel('Time Step')
ax.set_ylabel(r' Average ($S_t^2 / t)$')
ax.grid(True)

ax.title.set_fontsize(40)
for item in ([ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(40)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.offsetText.set_fontsize(40)
ax.legend(prop={'size': 40}, loc='lower left', bbox_to_anchor=(0.65, 0.05))
ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))

plt.tight_layout(pad=3.0)
plt.show()

# save it as vector form for latex use
fig.savefig('risk_vs_lqr_X29.pdf', format='pdf', bbox_inches='tight')


# plot 2---kicks every 200 time steps
time0 = np.arange(0,num_steps)
kick_every = 500
# Generate kicks every 20 time steps otherwise t-distribution with nu = 5
w = np.array([20*np.array([1,1,0,0, 0,0,0,0]) if t % kick_every == 0 else 0.1*generate_normalized_t_distribution(nu=5, size=n) for t in range(num_steps)])
X_LQR = simulate_response_kicks(A, B, K_LQR, x0, w, num_steps)
X_final = simulate_response_kicks(A, B, K_final, x0, w, num_steps)

fig, ax = plt.subplots(1, 1, figsize=(20, 8))
ax.plot(time0, la.norm(X_LQR,axis=0), label='LQR Optimal', linestyle='--', linewidth=2.5)
ax.plot(time0, la.norm(X_final,axis=0), label='Ergodic-risk Optimal', linestyle='-', linewidth=2.5)
# plot a vertical line at every 20 time steps
for i in range(0, num_steps, kick_every):
    ax.axvline(x=i, color='gray', linestyle='--', linewidth=1.5)
ax.set_xlabel('Time Step')
ax.set_ylabel(r'$\|X_t\|$')
ax.grid(True)

ax.title.set_fontsize(40)
for item in ([ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(40)
ax.legend(prop={'size': 40}, loc='upper left', bbox_to_anchor=(0.65, 1.15))
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.offsetText.set_fontsize(40)

plt.tight_layout(pad=3.0)
plt.show()

# save it as vector form for latex use
fig.savefig('risk_vs_lqr_kicks_X29.pdf', format='pdf', bbox_inches='tight')



# plot 3: C_t
# Time axis
time = np.arange(1, num_steps)

fig, ax = plt.subplots(figsize=(20, 8))

ax.plot(time, C_LQR, label='LQR Optimal', linestyle='--', linewidth=2.5)
ax.plot(time, C_final, label='Ergodic-risk Optimal', linestyle='-', linewidth=1.5)
ax.set_xlabel('Time Step')
ax.set_ylabel(r'$C_t$')
ax.grid(True)

ax.title.set_fontsize(40)
for item in ([ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(40)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
# ax.set_ylim(-2000, 5000)
ax.yaxis.offsetText.set_fontsize(40)
ax.legend(prop={'size': 40}, loc='upper right')
ax.ticklabel_format(axis='y', style='sci', scilimits=(4,2))

plt.tight_layout(pad=3.0)
plt.show()

# Save plot
fig.savefig('C_t_comparison.pdf', format='pdf', bbox_inches='tight')



###### combined plots for ACC

# Create a figure with two subplots, sharing the x-axis
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 10))

# First subplot: C_t
axs[0].plot(time, C_LQR, label='LQR Optimal', linestyle='-', linewidth=2.5)
axs[0].plot(time, C_final, label='Ergodic-risk Optimal', linestyle='--', linewidth=1)
axs[0].set_ylabel(r'$C_t$', fontsize=40)
axs[0].grid(True)
axs[0].tick_params(axis='both', labelsize=35)
# axs[0].set_ylim(-2000, 2000)
axs[0].ticklabel_format(axis='y', style='sci', scilimits=(4, 2))

# Increase font size of the scientific notation exponent
axs[0].yaxis.get_offset_text().set_fontsize(40)

# Format tick labels and axes
axs[0].title.set_fontsize(40)
for item in ([axs[0].xaxis.label, axs[0].yaxis.label] +
                axs[0].get_xticklabels() + axs[0].get_yticklabels()):
    item.set_fontsize(40)
axs[0].xaxis.set_major_locator(plt.MaxNLocator(5))

# Second subplot: Norm of X_t
axs[1].plot(time0, la.norm(X_LQR, axis=0), label='LQR Optimal', linestyle='-', linewidth=2.5)
axs[1].plot(time0, la.norm(X_final, axis=0), label='Ergodic-risk Optimal', linestyle='--', linewidth=2)
for i in range(0, num_steps, kick_every):  # Vertical lines for kicks
    axs[1].axvline(x=i, color='gray', linestyle='--', linewidth=1.5)
axs[1].set_xlabel('Time Step', fontsize=40)
axs[1].set_ylabel(r'$\|X_t\|$', fontsize=40)
axs[1].grid(True)
axs[1].tick_params(axis='both', labelsize=35)
axs[1].ticklabel_format(axis='y', style='sci', scilimits=(4, 2))

# Increase font size of the scientific notation exponent
axs[1].yaxis.get_offset_text().set_fontsize(40)

# Format tick labels and axes
axs[1].title.set_fontsize(40)
for item in ([axs[1].xaxis.label, axs[1].yaxis.label] +
                axs[1].get_xticklabels() + axs[1].get_yticklabels()):
    item.set_fontsize(40)
axs[1].xaxis.set_major_locator(plt.MaxNLocator(5))

# Add legend only in the first plot
axs[1].legend(prop={'size': 40}, loc='lower left', bbox_to_anchor=(0.6, 0.75))

# Adjust layout to minimize gaps between subplots
plt.subplots_adjust(hspace=0.0)
plt.tight_layout(pad=3.0)

plt.show()

# Save figure as PDF
fig.savefig('Ct_and_NormX_comparison_ACC.pdf', format='pdf', bbox_inches='tight')
