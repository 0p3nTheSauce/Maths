import numpy as np
import matplotlib.pyplot as plt

# Define the functions
def f(t):
    return 6 + (t**2) / 4 + (t**3) / 4

def dif_f(t):
    return (3 / 8) * t**2 + (1 / 2) * t

def petrol():
    # Generate a range of t values
    t_values = np.linspace(-10, 10, 400)  # Adjust the range and number of points as needed

    # Calculate the corresponding y values for both functions
    y_values_f = f(t_values)
    y_values_dif_f = dif_f(t_values)

    # Plot both functions
    plt.plot(t_values, y_values_f, label='6 + t^2/4 + t^3/4', color='b')
    plt.plot(t_values, y_values_dif_f, label='(3/8) * t^2 + (1/2) * t', color='r', linestyle='--')

    # Add labels, title, and legend
    plt.xlabel('t')
    plt.ylabel('Function values')
    plt.title('Plot of the functions')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()

def q6():
    # Generate x values from -240 to 240
    x_values = np.linspace(-240, 240, 480)
    
    # Assuming g(x), a(x), and h(x) are defined functions
    y_val_g = g(x_values)
    y_val_g_shift = g_shift(x_values)
    y_val_a = a(x_values)
    y_val_h = h(x_values)
    
    
    plt.plot(x_values, y_val_g, label='g(x) = sin(x + 30)', color='b')
    plt.plot(x_values, y_val_g_shift, label='i(x) = sin(x + 90)', color='g')
    # Uncomment if you want to plot y_val_a
    #plt.plot(x_values, y_val_a, label='a(x) = -cos(x + 90)', color='g')
    plt.plot(x_values, y_val_h, label='h(x) = -cos(x)', color='r')

    plt.xlabel('x (degrees)')
    plt.ylabel('y')
    plt.title('Plot of functions')
    plt.grid(True)
    plt.legend()
    plt.show()

# Example definitions for g, a, and h functions
def g(x):
    return np.sin(np.radians(x + 30))  # Ensure x is in degrees

def g_shift(x):
    return np.sin(np.radians(x + 90))

def a(x):
    return -np.cos(np.radians(x - 30))  # Ensure x is in degrees

def h(x):
    return -np.cos(np.radians(x))  # Ensure x is in degrees
    
def main():
    q6()
    
if __name__ == "__main__":
    main()    