# shooter_sim

FRC 2026 Turret Ball Shooter Simulation — a physics-based tool for computing optimal launch angles and velocities for a ball shooter mechanism targeting an elevated hub.

## Overview

This project simulates a ball shooter with **quadratic air drag** (smooth-sphere model) to help FRC teams tune their shooter subsystem. It provides:

- **Graphical analysis** — sweep a range of distances and visualize optimal angles, velocities, and trajectories
- **Command-line lookup** — quickly query the recommended angle/velocity for a specific distance, or find angles for a known velocity
- **2D lookup table generation** — pre-compute a grid of (distance, turret height) → (angle, velocity) for use on a robot

### Physics Model

Trajectory integration uses symplectic Euler with quadratic drag, JIT-compiled via [Numba](https://numba.pydata.org/) for ~50–100× speedup over pure Python:

$$a_x = -\frac{k}{m} \lvert v \rvert \, v_x, \qquad a_y = -g - \frac{k}{m} \lvert v \rvert \, v_y$$

where $k = \tfrac{1}{2} C_d \rho A$ is the drag constant for a smooth sphere ($C_d = 0.47$) at sea-level air density.

### Default Parameters

| Parameter | Value |
|---|---|
| Ball diameter | 5.91 in |
| Ball mass | 0.448–0.500 lb (avg ≈ 0.215 kg) |
| Hub height | 72.0 in |
| Turret height | 27.0 in (configurable) |
| Drag coefficient | 0.47 |

## Files

| File | Description |
|---|---|
| `shooter_physics.py` | Shared physics engine — constants, Numba-JIT trajectory simulation, solvers |
| `shooter_sim.py` | Graphical analysis — sweeps distances and generates matplotlib plots |
| `shooter_lookup.py` | CLI lookup tool — recommend, angle, and table commands |
| `generate_table.py` | Generates a 2D interpolation table saved to `shooter_table.npz` |

## Installation

Requires Python 3.8+.

```bash
pip install numpy scipy numba matplotlib
```

## Usage

### Graphical Simulation (`shooter_sim.py`)

Sweeps a range of distances and produces plots of optimal launch angle, required velocity, sample trajectories, drag comparison, and velocity penalty from air resistance.

```bash
# Default parameters
python shooter_sim.py

# Custom turret height and distance range
python shooter_sim.py --turret-height 30 --dist-min 5 --dist-max 20

# Save figure to file instead of displaying
python shooter_sim.py --save output.png
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--turret-height` | 27.0 | Turret height in inches |
| `--hub-height` | 72.0 | Hub target height in inches |
| `--dist-min` | 5.0 | Minimum distance in feet |
| `--dist-max` | 25.0 | Maximum distance in feet |
| `--num-points` | 50 | Number of sample distances |
| `--save` | *(none)* | Save figure to file path |

### Shooter Lookup (`shooter_lookup.py`)

Quick command-line queries for shot parameters.

```bash
# Recommend optimal angle & velocity for a 15 ft shot
python shooter_lookup.py recommend --distance 15

# Find launch angle(s) for a known velocity
python shooter_lookup.py angle --distance 15 --velocity 28

# Print a full lookup table
python shooter_lookup.py table --dist-min 5 --dist-max 25 --step 2
```

**Commands:**

- **`recommend`** — Given a distance, returns the optimal (minimum-velocity) launch angle and required velocity, plus wheel RPM estimates for 4″ and 6″ shooter wheels.
- **`angle`** — Given a distance and velocity, finds all valid launch angles (low arc / high arc) using root-finding.
- **`table`** — Prints a formatted lookup table of angle and velocity across a range of distances.

All commands accept `--turret-height` and `--hub-height` overrides.

### Table Generation (`generate_table.py`)

Pre-computes a 2D grid of solutions across distance (3–30 ft) and turret height (18–42 in), saving the result to `shooter_table.npz` for fast interpolation on a robot.

```bash
python generate_table.py
```

Output file contains NumPy arrays: `distances_ft`, `turret_heights_in`, `hub_height_in`, `angle_deg`, and `velocity_fps`.
