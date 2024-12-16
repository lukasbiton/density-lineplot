"""
For a given timeseries, reduce the frequency (e.g. from every second to every hour)
and plot the density per new time unit over time.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interpn

plt.style.use("dark_background")

def jitter_x_axis(x: npt.NDArray):
    """Jitter data points on the x-axis."""
    return x + np.random.uniform(-0.2, 0.2, x.shape)

def make_density_color(x, y, bins):
    """Stolen from
    https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density/53865762#53865762
    Uses a histogram to density map data points.
    """
    data, x_e, y_e = np.histogram2d(x, y, bins = bins, density = True)
    z = interpn(
        (0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])),
        data,
        np.vstack([x,y]).T,
        method = "splinef2d",
        bounds_error = False
    )
    
    return z

def density_lineplot(data: npt.NDArray, out: str):
    """Plot density over time for grouped data."""

    fig, ax = plt.subplots(figsize=(20,10))
    
    for row_idx in range(data.shape[0]):

        x = row_idx*np.ones(data.shape[1])
        jittered_x = jitter_x_axis(x)
        y_set = data[row_idx]
        z = make_density_color(x, y_set, 20)

        ax.scatter(
            x=jittered_x,
            y=y_set,
            alpha=0.1,
            c=z,
            cmap="magma",
        )

    plt.savefig(f"{out}.png")
    plt.close()

if __name__ == "__main__":

    out = "out_density_line/"
    os.makedirs(out, exist_ok=True)
    
    time_indices = 100
    data_points_per_time = 500

    xs = (np.ones((time_indices, data_points_per_time))
          * np.reshape(np.linspace(1, time_indices, time_indices), (-1, 1)))

    data = np.random.negative_binomial(100,1/xs)
    density_lineplot(data, f"{out}neg_bin")

    data = np.random.normal(np.sin(xs), np.random.uniform(2,8,xs.shape))
    density_lineplot(data, f"{out}normal")

    data = np.random.gamma(xs, np.random.uniform(2,8,xs.shape))
    density_lineplot(data, f"{out}gamma")
