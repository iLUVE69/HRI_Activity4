import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------
# Parameters for the manipulator
# ---------------------------
l1 = 1.0  # Length of link 1 (thigh) in meters
l2 = 1.0  # Length of link 2 (shank) in meters

# ---------------------------
# Parameters for cable length functions (muscle model)
# ---------------------------
# Nominal cable lengths (these can be arbitrary since we focus on rates)
L_PS0    = 0.5
L_RF0    = 0.5
L_BFLH0  = 0.5
# Sensitivity coefficients (how cable length changes with joint angle)
alpha_PS    = 1.0    # PS depends only on q1
alpha_RF1   = 0.8    # RF dependency on q1
alpha_RF2   = 0.5    # RF dependency on q2
alpha_BFLH  = 1.2    # BFLH depends only on q2

# Construct the joint-to-cable Jacobian (3x2)
J_c = np.array([[alpha_PS,    0],
                [alpha_RF1, alpha_RF2],
                [0,        alpha_BFLH]])

# ---------------------------
# Time parameters
# ---------------------------
T_total = 10.0   # Total simulation time (seconds)
N = 500          # Number of time steps
t = np.linspace(0, T_total, N)
dt = t[1] - t[0]

# ---------------------------
# Prescribed joint trajectories (for simulation)
# ---------------------------
# Example: sinusoidal variations
q1 = 0.3 * np.sin(2 * np.pi * t / T_total)   # q1(t) in radians
q2 = 0.2 * np.cos(2 * np.pi * t / T_total)   # q2(t) in radians

# Compute joint velocities using finite differences
dq1_dt = np.gradient(q1, dt)
dq2_dt = np.gradient(q2, dt)
dq_dt = np.vstack((dq1_dt, dq2_dt))  # Shape: (2, N)

# ---------------------------
# Define cable length functions
# ---------------------------
# For PS: l_PS = L_PS0 + alpha_PS * q1
l_PS = L_PS0 + alpha_PS * q1
# For RF: l_RF = L_RF0 + alpha_RF1 * q1 + alpha_RF2 * q2
l_RF = L_RF0 + alpha_RF1 * q1 + alpha_RF2 * q2
# For BFLH: l_BFLH = L_BFLH0 + alpha_BFLH * q2
l_BFLH = L_BFLH0 + alpha_BFLH * q2

# Compute cable length rates analytically:
dl_PS_dt = alpha_PS * dq1_dt
dl_RF_dt = alpha_RF1 * dq1_dt + alpha_RF2 * dq2_dt
dl_BFLH_dt = alpha_BFLH * dq2_dt

# Stack actual cable velocities (shape: 3 x N)
dl_actual = np.vstack((dl_PS_dt, dl_RF_dt, dl_BFLH_dt))

# ---------------------------
# Task Space Jacobian for 2R Manipulator
# ---------------------------
def J_task(q1, q2):
    J = np.array([[-l1 * np.sin(q1) - l2 * np.sin(q1 + q2), -l2 * np.sin(q1 + q2)],
                  [ l1 * np.cos(q1) + l2 * np.cos(q1 + q2),  l2 * np.cos(q1 + q2)]])
    return J  # Shape: (2,2)

# ---------------------------
# Mapping: For each time step, compute:
#   v = J_t * dq_dt  (end-effector velocity)
#   Predicted cable rate: dl_pred = J_c * dq_dt = J_c * J_t^{-1} * v
# ---------------------------
v_task = np.zeros((2, N))       # End-effector velocities
dl_pred = np.zeros((3, N))      # Predicted cable velocities via mapping

for i in range(N):
    J_t = J_task(q1[i], q2[i])
    dq = dq_dt[:, i]  # Joint velocities at time step i
    # Compute task space velocity (foot velocity)
    v = J_t @ dq  # Shape: (2,)
    v_task[:, i] = v
    # Also compute predicted cable rates:
    dl_pred[:, i] = J_c @ dq  # Which should match the analytical dl_actual[:, i]
    # Alternatively, using the mapping: dl_pred = J_c * J_t^{-1} * v (if J_t is invertible)
    # dl_pred_alt = J_c @ np.linalg.inv(J_t) @ v  # should be equal to J_c @ dq

# ---------------------------
# Define forward kinematics (to display the manipulator)
# ---------------------------
def forward_kinematics(q1, q2):
    base = np.array([0, 0])
    joint1 = np.array([l1 * np.cos(q1), l1 * np.sin(q1)])
    ee = joint1 + np.array([l2 * np.cos(q1 + q2), l2 * np.sin(q1 + q2)])
    return base, joint1, ee

# ---------------------------
# Define cable attachment points on the manipulator for visualization
# ---------------------------
def attachment_points(q1, q2):
    # Assume:
    # - PS attaches at the midpoint of link1
    attach_PS = np.array([0.5 * l1 * np.cos(q1), 0.5 * l1 * np.sin(q1)])
    # - RF attaches at the end of link1 (the knee)
    joint1 = np.array([l1 * np.cos(q1), l1 * np.sin(q1)])
    attach_RF = joint1 + 0.5 * np.array([l2 * np.cos(q1 + q2), l2 * np.sin(q1 + q2)])
    # - BFLH attaches at the midpoint of link2
    # Compute joint1 then midpoint of link2:
    attach_BFLH = joint1 + 0.5 * np.array([l2 * np.cos(q1 + q2), l2 * np.sin(q1 + q2)])
    return attach_PS, attach_RF, attach_BFLH

# Fixed anchor positions for cables (for visualization)
anchor_PS   = np.array([-1.5, 1.5])
anchor_RF   = np.array([ 0.0, 1.5])
anchor_BFLH = np.array([ 1.5, -1.5])

# ---------------------------
# Set up matplotlib animation
# ---------------------------
fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Cable-Velocity to Task-Space Velocity Mapping")

# Plot elements: manipulator, cable lines, and velocity vector
line_manip, = ax.plot([], [], 'o-', lw=3, color='blue', label='Manipulator')
cable_line_PS, = ax.plot([], [], 'r--', lw=1.5, label='Cable PS')
cable_line_RF, = ax.plot([], [], 'g--', lw=1.5, label='Cable RF')
cable_line_BFLH, = ax.plot([], [], 'm--', lw=1.5, label='Cable BFLH')
vel_arrow = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='red')

# Text overlay to display cable rate values and task velocity magnitude
info_text = ax.text(-1.8, 1.7, "", fontsize=8, color="black")

def init_anim():
    line_manip.set_data([], [])
    cable_line_PS.set_data([], [])
    cable_line_RF.set_data([], [])
    cable_line_BFLH.set_data([], [])
    vel_arrow.set_UVC(0, 0)
    info_text.set_text("")
    return line_manip, cable_line_PS, cable_line_RF, cable_line_BFLH, vel_arrow, info_text

def animate(i):
    # Current joint angles
    q1_i = q1[i]
    q2_i = q2[i]
    
    # Forward kinematics for manipulator positions
    base, joint1, ee = forward_kinematics(q1_i, q2_i)
    x_manip = [base[0], joint1[0], ee[0]]
    y_manip = [base[1], joint1[1], ee[1]]
    line_manip.set_data(x_manip, y_manip)
    
    # Compute cable attachment points
    att_PS, att_RF, att_BFLH = attachment_points(q1_i, q2_i)
    
    # Update cable lines: from attachment point to fixed anchor
    cable_line_PS.set_data([att_PS[0], anchor_PS[0]], [att_PS[1], anchor_PS[1]])
    cable_line_RF.set_data([att_RF[0], anchor_RF[0]], [att_RF[1], anchor_RF[1]])
    cable_line_BFLH.set_data([att_BFLH[0], anchor_BFLH[0]], [att_BFLH[1], anchor_BFLH[1]])
    
    # Compute current task space velocity (v) using the task Jacobian and joint velocities:
    J_t = J_task(q1_i, q2_i)
    dq = np.array([dq1_dt[i], dq2_dt[i]])
    v = J_t @ dq  # End-effector velocity (2,)
    
    # Set velocity arrow at end-effector (red arrow)
    vel_arrow.set_offsets([ee[0], ee[1]])
    vel_arrow.set_UVC(v[0], v[1])
    
    # Get cable velocities from our mapping (from our precomputed dl_actual, which equals J_c dq)
    rate_PS = dl_actual[0, i]
    rate_RF = dl_actual[1, i]
    rate_BFLH = dl_actual[2, i]
    
    # Also compute norm of task velocity
    v_norm = np.linalg.norm(v)
    
    # Update info text to include these values
    info = (f"||V|| = {v_norm:.3f} m/s\n"
            f"Cable Rates:\n"
            f"  PS: {rate_PS:.3f} m/s\n"
            f"  RF: {rate_RF:.3f} m/s\n"
            f"  BFLH: {rate_BFLH:.3f} m/s")
    info_text.set_text(info)
    
    return line_manip, cable_line_PS, cable_line_RF, cable_line_BFLH, vel_arrow, info_text

ani = animation.FuncAnimation(fig, animate, frames=N, init_func=init_anim,
                              interval=20, blit=True)

ax.legend(loc='upper right')
ani.save('mapping1r.gif', writer='pillow', fps=30)
plt.show()
