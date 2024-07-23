import pandapower
import numpy as np
import matplotlib.pyplot as plt

num_buses = 3
admittances = {
    (0, 1): 5 - 1j * 15,   # admittance between bus 0 and bus 1 (Slack <-> PQ)
    (1, 2): 15 - 1j * 50,  # admittance between bus 1 and bus 2 (PQ <-> PV)
    (0, 2): 10 - 1j * 40   # admittance between bus 0 and bus 2 (Slack <-> PV)
}

# Initialize the Y-bus matrix with zeros
Y_bus = np.zeros((num_buses, num_buses), dtype=complex)

# Populate the Y-bus matrix
for (i, j), y in admittances.items():
    Y_bus[i, j] = -y
    Y_bus[j, i] = -y
    Y_bus[i, i] += y
    Y_bus[j, j] += y

# Print the Y-bus matrix
print("Imaginary Y-bus matrix (rectangular form):")
print(Y_bus)  # Only the imaginary part

# Compute the magnitude and phase angle of the Y-bus matrix elements
Y_bus_magnitude = np.abs(Y_bus)
Y_bus_angle = np.angle(Y_bus)

# Convert Y-bus to polar form
Y_bus_polar = np.zeros((num_buses, num_buses), dtype=object)

# Populate Y_bus_polar with magnitude and angle tuples
for i in range(num_buses):
    for j in range(num_buses):
        Y_bus_polar[i, j] = (Y_bus_magnitude[i, j], np.degrees(Y_bus_angle[i, j]))

# Print the Y-bus matrix in polar form
print("Y-bus matrix in polar form:")
for i in range(num_buses):
    for j in range(num_buses):
        magnitude, angle = Y_bus_polar[i, j]
        print(f"Y_bus[{i},{j}] = {magnitude:.4f} ∠ {angle:.4f}°")

# Node 0: slack (reference) bus
d_0 = 0 # [°], voltage angle
v_0 = 1.02 # [p.u.], voltage magnitude

# Node 1: PQ (load) bus (define load as negative)
s_1 = -2 - 1j * 0.5 # [p.u.] dividing real (MW) and imag (MVAR) by the base 100 MVA

# Node 2: PV (gen) bus (define gen as positive)
p_2 = 1.5 # [p.u.] dividing by 100 MW
v_2 = 1.03 # [p.u.], voltage magnitude


# Maximum iterations and tolerance
max_iterations = 25
tolerance = 1e-6

def net_P1(Y_bus_polar, V, delta):
    P1 = (V[1] * V[0] * Y_bus_polar[1, 0][0] * np.cos(np.deg2rad(Y_bus_polar[1, 0][1] - delta[1] + delta[0])) +
          V[1]**2 *  Y_bus_polar[1, 1][0] * np.cos(np.deg2rad(Y_bus_polar[1, 1][1])) +
          V[1] * V[2] * Y_bus_polar[1, 2][0] * np.cos(np.deg2rad(Y_bus_polar[1, 2][1] - delta[1] + delta[2])))
    return P1

def net_P2(Y_bus_polar, V, delta):
    P2 = (V[2] * V[0] * Y_bus_polar[2, 0][0] * np.cos(np.deg2rad(Y_bus_polar[2, 0][1] - delta[2] + delta[0])) +
      V[2]**2 *  Y_bus_polar[2, 2][0] * np.cos(np.deg2rad(Y_bus_polar[2, 2][1])) +
      V[2] * V[1] * Y_bus_polar[2, 1][0] * np.cos(np.deg2rad(Y_bus_polar[2, 1][1] - delta[2] + delta[1])))
    return P2

def net_Q1(Y_bus_polar, V, delta):
    Q1 = (-V[1] * V[0] * Y_bus_polar[1, 0][0] * np.sin(np.deg2rad(Y_bus_polar[1, 0][1] - delta[1] + delta[0])) -
          V[1]**2 *  Y_bus_polar[1, 1][0] * np.sin(np.deg2rad(Y_bus_polar[1, 1][1])) -
          V[1] * V[2] * Y_bus_polar[1, 2][0] * np.sin(np.deg2rad(Y_bus_polar[1, 2][1] - delta[1] + delta[2])))
    return Q1

def calculate_jacobian(Y_bus_polar, V, delta):
    # Initialize the J matrix with zeros
    J = np.zeros((len(Y_bus_polar),len(Y_bus_polar)))

    # Row 1
    # dP_1/dd_1
    J[0, 0] = V[1] * V[0] * Y_bus_polar[1, 0][0] * np.sin(np.deg2rad(Y_bus_polar[1, 0][1] - delta[1] + delta[0])) + V[1] * V[2] * Y_bus_polar[1, 2][0] * np.sin(np.deg2rad(Y_bus_polar[1, 2][1] - delta[1] + delta[2]))

    # dP_1/dd_2
    J[0, 1] = -V[1] * V[2] * Y_bus_polar[1, 2][0] * np.sin(np.deg2rad(Y_bus_polar[1, 2][1] - delta[1] + delta[2]))
    
    # dP_1/dv_1
    J[0, 2] = V[0] * Y_bus_polar[1, 0][0] * np.cos(np.deg2rad(Y_bus_polar[1, 0][1] - delta[1] + delta[0])) + V[2] * Y_bus_polar[1, 2][0] * np.cos(np.deg2rad(Y_bus_polar[1, 2][1] - delta[1] + delta[2])) + 2 * V[1] * Y_bus_polar[1, 1][0] * np.cos(np.deg2rad(Y_bus_polar[1, 1][1]))

    # Row 2
    # dP_2/dd_1
    J[1, 0] = -V[2] * V[1] * Y_bus_polar[2, 1][0] * np.sin(np.deg2rad(Y_bus_polar[2, 1][1] - delta[2] + delta[1]))
    
    # dP_2/dd_2
    J[1, 1] = V[2] * V[0] * Y_bus_polar[2, 0][0] * np.sin(np.deg2rad(Y_bus_polar[2, 0][1] - delta[2] + delta[0])) + V[2] * V[1] * Y_bus_polar[2, 1][0] * np.sin(np.deg2rad(Y_bus_polar[2, 1][1] - delta[2] + delta[1]))
    
    # dP_2/dv_1
    J[1, 2] = V[2] * Y_bus_polar[2, 1][0] * np.cos(np.deg2rad(Y_bus_polar[2, 1][1] - delta[2] + delta[1]))

    # Row 3
    # dQ_1/dd_1
    J[2, 0] = V[1] * V[0] * Y_bus_polar[1, 0][0] * np.cos(np.deg2rad(Y_bus_polar[1, 0][1] - delta[1] + delta[0])) + V[1] * V[2] * Y_bus_polar[1, 2][0] * np.cos(np.deg2rad(Y_bus_polar[1, 2][1] - delta[1] + delta[2]))
    
    # dQ_1/dd_2
    J[2, 1] = -V[1] * V[2] * Y_bus_polar[1, 2][0] * np.cos(np.deg2rad(Y_bus_polar[1, 2][1] - delta[1] + delta[2]))
    
    # dQ_1/dv_1
    J[2,2] = -V[0] * Y_bus_polar[1, 0][0] * np.sin(np.deg2rad(Y_bus_polar[1, 0][1] - delta[1] + delta[0])) - V[2] * Y_bus_polar[1, 2][0] * np.sin(np.deg2rad(Y_bus_polar[1, 2][1] - delta[1] + delta[2])) - 2 * V[1] * Y_bus_polar[1, 1][0] * np.sin(np.deg2rad(Y_bus_polar[1, 1][1]))

    return J

# Initial guess for voltages X := [d_1, d_2, v_1]
d_1 = 0
d_2 = 0
v_1 = 1

# Define scheduled/expected power for variables: [P_sch_1, P_sch_2, Q_sch_1]
scheduled_powers = [-2, 1.5, -0.5]

# Lists to store the convergence data
iterations = []
d1_values = []
d2_values = []
v1_values = []
power_mismatches = []

for iteration in range(max_iterations):
    # Voltage magnitudes and angles
    V = [v_0, v_1, v_2]
    delta = [d_0, d_1, d_2]
    
    # Calculate powers at iteration
    p_1 = net_P1(Y_bus_polar, V, delta)
    p_2 = net_P2(Y_bus_polar, V, delta)
    q_1 = net_Q1(Y_bus_polar, V, delta)

    # Calculate the difference of scheduled power to the current iteration
    delta_vector = scheduled_powers - np.array([p_1, p_2, q_1])

    # Store the current state
    iterations.append(iteration)
    d1_values.append(d_1)
    d2_values.append(d_2)
    v1_values.append(v_1)
    power_mismatches.append(np.linalg.norm(delta_vector, np.inf))
    
    # Check for convergence
    if np.linalg.norm(delta_vector, np.inf) < tolerance:
        break

    # Calculate the Jacobian
    Jacobian = calculate_jacobian(Y_bus_polar, V, delta)
    
    # Extract variables
    X = np.linalg.solve(Jacobian, delta_vector) # Units [rad, rad, p.u.]

    # Update voltage mags and angles
    d_1 += np.rad2deg(X[0])
    d_2 += np.rad2deg(X[1])
    v_1 += X[2]

    print(f"Iteration: {iteration}\nBus 1: {v_1}<{d_1}°\nBus 2: {v_2}<{d_2}°\n")

# Plot voltage angles
plt.subplot(2, 1, 1)
plt.plot(iterations, d1_values, label='Delta 1 (degrees)')
plt.plot(iterations, d2_values, label='Delta 2 (degrees)')
plt.xlabel('Iteration')
plt.ylabel('Voltage Angle (degrees)')
plt.title('Convergence of Voltage Angles')
plt.legend()
plt.grid(True)
plt.xticks(np.arange(min(iterations), max(iterations)+1, 1))

# Plot voltage magnitudes
plt.subplot(2, 1, 2)
plt.plot(iterations, v1_values, label='V1 (p.u.)')
plt.xlabel('Iteration')
plt.ylabel('Voltage Magnitude (p.u.)')
plt.title('Convergence of Voltage Magnitudes')
plt.legend()
plt.grid(True)
plt.xticks(np.arange(min(iterations), max(iterations)+1, 1))

plt.tight_layout()
plt.show()

# Plot power mismatches
plt.figure(figsize=(6, 4))
plt.plot(iterations, power_mismatches, label='Power Mismatch (p.u.)')
plt.axhline(y=tolerance, color='r', linestyle='--', label='Tolerance')
plt.xlabel('Iteration')
plt.ylabel('Mismatch (p.u.)')
plt.title('Convergence of Power Mismatches (P_sch - P_k)')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.xticks(np.arange(min(iterations), max(iterations)+1, 1))
plt.show()


#### TODO REPEAT THE SAME PROBLEM IN PANDA POWER TO COMPARE THE RESULTS