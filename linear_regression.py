import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import genfromtxt

def compute_error_for_line_given_points(b, m, points):
    # Computes the sum of squared errors
    total_error = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m * x + b)) ** 2
    return total_error / len(points)

def step_gradient(b, m, points, learning_rate):
    # Computes the gradient step
    b_gradient = 0
    m_gradient = 0
    N = len(points)
    for i in range(N):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - (m * x + b))
        m_gradient += -(2/N) * x * (y - (m * x + b))
    b -= learning_rate * b_gradient
    m -= learning_rate * m_gradient
    return b, m

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    progress = [(b, m)]  # Track progress for animation
    for _ in range(num_iterations):
        b, m = step_gradient(b, m, points, learning_rate)
        progress.append((b, m))
    return progress

def animate(i, points, progress, line):
    b, m = progress[i]
    x = points[:, 0]
    line.set_data(x, m * x + b)
    return line,

def run():
    # Step 1 - Collect data
    if not os.path.exists('data.csv'):
        raise FileNotFoundError("The file 'data.csv' does not exist.")
    points = genfromtxt('data.csv', delimiter=',')

    # Step 2 - Define hyperparameters
    learning_rate = 0.0001
    initial_b = 0  # Initial y-intercept
    initial_m = 0  # Initial slope
    num_iterations = 100

    # Step 3 - Train the model
    print("Starting gradient descent...")
    progress = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    # Step 4 - Visualize the progress
    fig, ax = plt.subplots()
    x = points[:, 0]
    y = points[:, 1]
    ax.scatter(x, y, color='blue', label='Data Points')
    line, = ax.plot([], [], color='red', label='Regression Line')
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Gradient Descent Progress')
    ax.legend()

    ani = FuncAnimation(fig, animate, frames=len(progress), fargs=(points, progress, line), interval=100, blit=True)
    plt.show()

if __name__ == '__main__':
    run()
