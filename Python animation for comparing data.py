import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the problem: harmonic oscillator
def harmonic_oscillator(t):
    return np.cos(t)

# Define a basic neural network prediction (this is a dummy example)
def neural_network(t):
    return np.cos(t) + 0.2*np.sin(3*t)

# Define a physics-informed neural network prediction
def physics_informed_nn(t):
    return np.cos(t) + 0.1*np.sin(5*t)

# Generate time points
t = np.linspace(0, 10, 500)

# Create the figure and axis
fig, ax = plt.subplots()
ax.set_xlim((0, 10))
ax.set_ylim((-1.5, 1.5))

# Lines for each of the curves
line1, = ax.plot([], [], lw=2, label='Exact solution')
line2, = ax.plot([], [], lw=2, label='Neural Network Prediction', linestyle='--')
line3, = ax.plot([], [], lw=2, label='Physics-Informed NN', linestyle=':')

# Scatter points to simulate training data
scat = ax.scatter([], [], color='orange', label='Training Data')

# Initialize the animation function
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    scat.set_offsets(np.empty((0, 2)))  # Clear scatter data
    return line1, line2, line3, scat

# Animation function to update the plot
def update(frame):
    # Select up to 'frame' points from the time array
    t_plot = t[:frame]
    
    # Update the lines
    line1.set_data(t_plot, harmonic_oscillator(t_plot))
    line2.set_data(t_plot, neural_network(t_plot))
    line3.set_data(t_plot, physics_informed_nn(t_plot))
    
    # Update scatter plot (sample every 50 points)
    scat_data = np.c_[t_plot[::50], harmonic_oscillator(t_plot[::50])]  # 2D array with x, y coordinates
    scat.set_offsets(scat_data)
    
    return line1, line2, line3, scat

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=False, interval=20)

# Add legend and labels
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.set_title('Harmonic Oscillator: Neural Network vs Physics-Informed NN')

# Show the animation
plt.show()
