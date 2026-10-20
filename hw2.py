import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid


bins, count = [], []

with open("hist2.csv", "r") as f:
    for line in f.readlines():
        b, c = [float(i) for i in line.split(",")]
        bins.append(b)
        count.append(c)

bins = np.array(bins)
count = np.array(count)


def particle(x, a, b, c):
    return a * np.exp(-((x - b)**2) / (2 * c**2))


def total_particle(x, a1, b1, c1, a2, b2, c2):
    return particle(x, a1, b1, c1) + particle(x, a2, b2, c2)


initial_guess = [850, 1.0, 0.25, 550, 2.0, 0.6]

params, covariance = curve_fit(
    total_particle,
    bins,
    count,
    p0=initial_guess,
    bounds=(
        [0, 0, 0.01, 0, 0, 0.01],
        [np.inf, 4, 4, np.inf, 4, 4]
    ),
    maxfev=20000
)

a1, b1, c1, a2, b2, c2 = params


x = np.linspace(0, 4, 500)

y1 = particle(x, a1, b1, c1)
y2 = particle(x, a2, b2, c2)
y_total = y1 + y2


area1 = trapezoid(y1, x)
area2 = trapezoid(y2, x)

ratio = area1 / area2

print("Particle A area =", area1)
print("Particle B area =", area2)
print("A : B =", ratio, ": 1")

print("A fraction =", area1 / (area1 + area2))
print("B fraction =", area2 / (area1 + area2))


plt.bar(bins, count, width=0.08,
        color="lightgray", label="Experimental data")

plt.plot(x, y1, "b--", label="Particle A")
plt.plot(x, y2, "g--", label="Particle B")
plt.plot(x, y_total, "r-", label="Total fitting")

plt.xlabel("Energy")
plt.ylabel("Counts")
plt.title("Gaussian Fitting of Energy Spectrum")

plt.legend()
plt.grid(True)
plt.show()
