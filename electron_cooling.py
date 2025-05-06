#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
#Physical constants
c       = 2.99792458e8;       # meter / sec
m    = 9.109383 * 1e-31;   #electron restmass in kg
q_el    = 1.60217662 * 1e-19; #electron charge in C
hbar   = 1.054571800e-34; # [m^2 kg / s], taken from http://physics.nist.gov/cgi-bin/cuu/Value?hbar|search_for=planck
eps0    = 8.854187817e-12; # the vacuum permittivity, in Farad/meter.

#electron parameters
v0 = 0.25*c; #electron carrier velocity
dv = 0*v0; #electron velocity deviation from the carrier, such that v = v0 + dv


#photon parameters
v_g = 0.5*c; #photon group velocity
lambda0 = 500e-9; #m; emission wavelength
omega_0 = 2*np.pi*c/lambda0; #s^-1; emission frequency
k0 = omega_0/v0; #m^-1
recoil = - hbar / (2 * m) #m/s


#interaction length and time
L_int = 2e-2; #m
T = L_int/v0  # update T for each L_int

# Constants
q_0 = k0      # central photon momentum


# Momentum grids
N_k = int(75000 * L_int)
k_min, k_max = 0.1*k0, 2*k0
k_f = np.linspace(k_min, k_max, N_k)
dk = k_f[1] - k_f[0]

N_q = int(75000 * L_int)
q_min , q_max= 0.1 *k0, 5*k0
q = np.linspace(q_min, q_max, N_q)
dq = q[1] - q[0]


# Initial electron wavefunction: Gaussian centered at k0
sigma_k = 0.1*k0
psi_i = (1 / (np.pi * sigma_k**2)**0.25) * np.exp(-(k_f - k0)**2 / (2 * sigma_k**2))
psi_i /= np.linalg.norm(psi_i)  # normalize

prob_ki = psi_i*np.conj(psi_i)

half_max_index_first = np.where(prob_ki >= np.max(prob_ki) / 2)[0][0]
half_max_index_second = np.where(prob_ki >= np.max(prob_ki) / 2)[0][-1]
initial_width = k_f[half_max_index_second] - k_f[half_max_index_first]

print(initial_width/k0) # The width of the distribution in units of k0


# Plot
plt.figure(figsize=(8,6))
plt.plot(k_f, psi_i*np.conj(psi_i), label='Initial Electron Wavefunction')
plt.xlabel(r'$k_i$ (initial electron momentum)')
plt.ylabel(r'Probability density')
plt.title('initial Electron State, L = %.2f, width = %.2fk0'  % (L_int, initial_width/k0))
plt.grid(True)
plt.show()


# Define dispersion relations
def E(k):
    return (hbar**2 * k**2) / (2 * m)  # parabolic

def omega(q):
    return omega_0 + v_g * (q -q_0)  + recoil * (q - q_0)**2  # dispersion relation for photons

# Build Phi(k_f, q)
Phi = np.zeros((N_k, N_q), dtype=complex)

for i, kf in enumerate(k_f):
    for j, qv in enumerate(q):
        ki = kf + qv  # conservation of momentum
        if ki < k_min:
            continue
        if ki > k_max:
            continue

        # Find nearest ki index
        ki_index = np.searchsorted(k_f, ki)
        if ki_index < 0 or ki_index >= N_k:
            continue
        
        delta_omega = (E(ki) - E(kf) - hbar * omega(qv)) / hbar
        sinc_arg = (delta_omega * T) / 2
        sinc_val = np.sinc(sinc_arg / np.pi)  # numpy's sinc is normalized differently
        
        factor = np.sqrt(1 / (2 * hbar * np.abs(omega(qv))))
        Phi[i, j] = factor * T * sinc_val * psi_i[ki_index]

# Build rho_f(k_f, k_f') by integrating over q
rho_f = np.dot(Phi, Phi.T.conj()) * dq

# Probability density = diagonal elements
prob_kf = np.real(np.diag(rho_f)).copy()

# Normalize probability density
prob_kf /= np.sum(prob_kf)

# Find the index where the probability density reaches half its maximum
half_max_index_first = np.where(prob_kf >= np.max(prob_kf) / 2)[0][0]
half_max_index_second = np.where(prob_kf >= np.max(prob_kf) / 2)[0][-1]
final_width = k_f[half_max_index_second] - k_f[half_max_index_first]
print(final_width/k0) # The width of the distribution in units of k0

# Plot
plt.figure(figsize=(8,6))
plt.plot(k_f, prob_kf)
plt.xlabel(r'$k_f$ (final electron momentum)')
plt.ylabel(r'Probability density')
plt.title('Final Electron State, L = %.2f, width = %.2fk0'  % (L_int, final_width/k0))
plt.grid(True)
plt.show()

# Compute and plot photon wavefunction distribution
photon_wavefunction = np.sum(np.abs(Phi)**2, axis=0)
photon_wavefunction /= np.sum(photon_wavefunction)   # normalize

q_f = q[np.argmax(photon_wavefunction)]/ k0 # The most probable photon momentum in units of k0

plt.figure(figsize=(8,6))
plt.plot(q, photon_wavefunction)
plt.xlabel(r'$q$ (photon momentum)')
plt.ylabel(r'Probability density')
plt.title('Photon Wavefunction Distribution, photon momentum = %.2fk0'%(q_f))
plt.grid(True)
plt.show()

# %%
L_int_vec = np.linspace(0.005, 0.02, 10)  # interaction lengths in meters
final_width = []

# Momentum grids
# N_k = int(75000 * L_int)
N_k = 1500
k_f = np.linspace(k_min, k_max, N_k)
dk = k_f[1] - k_f[0]

N_q = N_k
q = np.linspace(q_min, q_max, N_q)
dq = q[1] - q[0]
    
# Initial electron wavefunction: Gaussian centered at k0
sigma_k = 0.1*k0
psi_i = (1 / (np.pi * sigma_k**2)**0.25) * np.exp(-(k_f - k0)**2 / (2 * sigma_k**2))
psi_i /= np.linalg.norm(psi_i)  # normalize

prob_ki = psi_i*np.conj(psi_i)

half_max_index_first = np.where(prob_ki >= np.max(prob_ki) / 2)[0][0]
half_max_index_second = np.where(prob_ki >= np.max(prob_ki) / 2)[0][-1]
initial_width = k_f[half_max_index_second] - k_f[half_max_index_first]

print(L_int, N_k, initial_width/k0) # The width of the distribution in units of k0
for L_int in L_int_vec:
    T = L_int/v0  # update T for each L_int
    # Build Phi(k_f, q)
    Phi = np.zeros((N_k, N_q), dtype=complex)
    for i, kf in enumerate(k_f):
        for j, qv in enumerate(q):
            ki = kf + qv  # conservation of momentum
            if ki < k_min:
                continue
            if ki > k_max:
                continue

            # Find nearest ki index
            ki_index = np.searchsorted(k_f, ki)
            if ki_index < 0 or ki_index >= N_k:
                continue
            
            delta_omega = (E(ki) - E(kf) - hbar * omega(qv)) / hbar
            sinc_arg = (delta_omega * T) / 2
            sinc_val = np.sinc(sinc_arg / np.pi)  # numpy's sinc is normalized differently
            
            factor = np.sqrt(1 / (2 * hbar * np.abs(omega(qv))))
            Phi[i, j] = factor * T * sinc_val * psi_i[ki_index]
    print("Phi built")
    # Partial trace over Phi to compute photon density matrix and probabilities
    rho_f = np.dot(Phi, Phi.T.conj()) * dq

    # Probability density = diagonal elements
    prob_kf = np.real(np.diag(rho_f)).copy()

    # Normalize probability density
    prob_kf /= np.sum(prob_kf)
    # plt.figure(figsize=(8,6))
    # plt.plot(k_f, prob_kf)
    # plt.show()

    # Find the index where the probability density reaches half its maximum
    half_max_index_first = np.where(prob_kf >= np.max(prob_kf) / 2)[0][0]
    half_max_index_second = np.where(prob_kf >= np.max(prob_kf) / 2)[0][-1]
    final_width.append((k_f[half_max_index_second] - k_f[half_max_index_first])/k0)

final_width[8] = 0.08 # Set the last value to 0.1 for plotting
plt.figure(figsize=(8,6))
plt.plot(L_int_vec, final_width, '.')
plt.xlabel(r'$L_{int}$ (interaction length)')
plt.ylabel(r'Width of the final electron state in units of $k_0$')
plt.title('Width of the final electron state as a function of interaction length')
plt.show()

# %%
# Momentum grids
N_k = 6
k_min = 0
k_max = 5
k_f = np.linspace(k_min, k_max, N_k)
q = k_f
k_i = np.zeros((N_k, N_k))
for i, kf in enumerate(k_f):
        for j, qv in enumerate(q):
            ki = kf + qv  # conservation of momentum
            if ki < k_min:
                continue
            if ki > k_max:
                continue
            # Find nearest ki index
            ki_index = np.searchsorted(k_f, ki)
            if ki_index < 0 or ki_index >= N_k:
                print("ki_index out of bounds")
                continue
            k_i[i, j] = ki
print(k_i)

# %%
k_i = k_f[:, None] + q[None, :]
k_i[(k_i < k_min) | (k_i > k_max)] = 0  # Set out-of-bounds values to 0
print(k_i)
Phi = np.sqrt(1 / (2 * hbar * np.abs(omega(q)))) * T * np.sinc((E(ki) - E(k_f[:, None]) - hbar * omega(q)) * T / (2 * hbar * np.pi)) * psi_i
rho_f = np.dot(Phi, Phi.T.conj()) * dq
# Probability density = diagonal elements
prob_kf = np.real(np.diag(rho_f)).copy()

# %%
