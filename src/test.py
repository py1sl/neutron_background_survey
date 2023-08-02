import numpy as np
import matplotlib.pyplot as plt
import math
from time import perf_counter


def pi_estimate(counts):

    radii = []
    inside_count = 0
    outside_count = 0

    for i in range(counts):

        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        radius = np.sqrt(x**2 + y**2)
        radii.append(radius)

        if radius < 1:
            # true for if inside circle
            inside_circle_x.append(x)
            inside_circle_y.append(y)
            inside_count += 1

        elif radius > 1:
            outside_circle_x.append(x)
            outside_circle_y.append(y)
            outside_count += 1

    # pi is the area of a circle with a radius of 1
    pi_estimatation = 4 * inside_count / counts
    return pi_estimatation


def plot_circle(counts):

    pi = pi_estimate(counts)
    print("The estimate of pi for ", counts, " counts is ", pi)
    plt.figure(figsize=(5, 5))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(inside_circle_x, inside_circle_y)
    plt.scatter(outside_circle_x, outside_circle_y)
    plt.Circle((0, 0), 0.5, color="r")
    plt.title("Accepted or rejected points for a circle of radius 1")
    plt.show()

    return None


number_counts = 10000
plot_circle(number_counts)

# start the counter for checking runtime for varying the number of counts
time_start = perf_counter()
number_counts_array = np.arange(1, 10000, 100)
pi_estimates = []

for counts in number_counts_array:

    inside_circle_x = []
    inside_circle_y = []
    outside_circle_x = []
    outside_circle_y = []
    on_circle_x = []
    on_circle_y = []
    pi = pi_estimate(counts)
    pi_estimates.append(pi)

plt.figure(figsize=(5, 5))
plt.plot(number_counts_array, pi_estimates)
plt.title("Estimate of pi vs number of counts")
plt.xlabel("Number of counts")
plt.ylabel("Estimate of Pi")
plt.axhline(y=math.pi, color="r")
plt.show()
time_end = perf_counter()

print("Total runtime is  {0:0.5f} seconds".format(time_end - time_start))