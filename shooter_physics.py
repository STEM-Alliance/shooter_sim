"""
shooter_physics.py — Shared physics engine for FRC 2026 shooter simulation
===========================================================================

Contains:
  - Physical constants (ball, air, hub, turret defaults)
  - Numba-JIT-compiled trajectory simulation with quadratic drag
  - Solver functions: velocity from angle, optimal angle from distance
  - Unit conversion helpers

All trajectory math runs through Numba-compiled functions for ~50-100x
speed improvement over pure Python loops.
"""

import math
import numpy as np
from numba import njit
from scipy.optimize import brentq

# ===========================================================================
# Physical constants
# ===========================================================================
G = 9.81            # m/s²
RHO_AIR = 1.225     # kg/m³  (sea-level standard)
CD_SPHERE = 0.47    # drag coefficient for a smooth sphere

# ===========================================================================
# Ball parameters  (FRC 2026 game piece)
# ===========================================================================
BALL_DIAMETER_IN = 5.91
BALL_DIAMETER_M = BALL_DIAMETER_IN * 0.0254
BALL_RADIUS_M = BALL_DIAMETER_M / 2.0
BALL_MASS_LB_LO = 0.448
BALL_MASS_LB_HI = 0.500
BALL_MASS_KG = (BALL_MASS_LB_LO + BALL_MASS_LB_HI) / 2.0 * 0.453592
BALL_CROSS_SECTION = math.pi * BALL_RADIUS_M ** 2

# Drag constant: F_drag = K_DRAG * v²
K_DRAG = 0.5 * CD_SPHERE * RHO_AIR * BALL_CROSS_SECTION
KM = K_DRAG / BALL_MASS_KG  # k/m  (used in EOM)

# ===========================================================================
# Game-field dimensions  (FRC 2026)
# ===========================================================================
HUB_WIDTH_IN = 47.0
HUB_OPENING_IN = 41.7   # hexagonal opening
HUB_HEIGHT_IN = 72.0    # front edge of opening off carpet
HUB_WIDTH_M = HUB_WIDTH_IN * 0.0254
HUB_HEIGHT_M = HUB_HEIGHT_IN * 0.0254

# ===========================================================================
# Robot / turret defaults
# ===========================================================================
DEFAULT_TURRET_HEIGHT_IN = 27.0
DEFAULT_TURRET_HEIGHT_M = DEFAULT_TURRET_HEIGHT_IN * 0.0254

# ===========================================================================
# Default simulation distance range
# ===========================================================================
DIST_MIN_FT = 5.0
DIST_MAX_FT = 25.0
NUM_DISTANCES = 50

# ===========================================================================
# Conversion helpers
# ===========================================================================
IN_TO_M = 0.0254
FT_TO_M = 0.3048
M_TO_IN = 1.0 / IN_TO_M
M_TO_FT = 1.0 / FT_TO_M
LB_TO_KG = 0.453592


# ===========================================================================
# JIT-compiled trajectory simulation
# ===========================================================================
@njit(cache=True)
def _simulate_xy(v0, angle_rad, turret_height_m, km, g, dt, n_steps):
    """
    Core trajectory integration — Numba JIT compiled.

    Symplectic Euler with quadratic air drag:
        ax = -(k/m) * |v| * vx
        ay = -g - (k/m) * |v| * vy

    Returns (xs, ys, count) where count is number of valid points.
    """
    vx = v0 * math.cos(angle_rad)
    vy = v0 * math.sin(angle_rad)
    x = 0.0
    y = turret_height_m

    xs = np.empty(n_steps, dtype=np.float64)
    ys = np.empty(n_steps, dtype=np.float64)

    for i in range(n_steps):
        xs[i] = x
        ys[i] = y

        v = math.sqrt(vx * vx + vy * vy)
        drag = km * v

        vx -= drag * vx * dt
        vy -= (g + drag * vy) * dt

        x += vx * dt
        y += vy * dt

        if y < 0.0:
            return xs, ys, i + 1

    return xs, ys, n_steps


@njit(cache=True)
def _simulate_xyvv(v0, angle_rad, turret_height_m, km, g, dt, n_steps):
    """
    Same as _simulate_xy but also returns vx, vy arrays.
    """
    vx = v0 * math.cos(angle_rad)
    vy = v0 * math.sin(angle_rad)
    x = 0.0
    y = turret_height_m

    xs = np.empty(n_steps, dtype=np.float64)
    ys = np.empty(n_steps, dtype=np.float64)
    vxs = np.empty(n_steps, dtype=np.float64)
    vys = np.empty(n_steps, dtype=np.float64)

    for i in range(n_steps):
        xs[i] = x
        ys[i] = y
        vxs[i] = vx
        vys[i] = vy

        v = math.sqrt(vx * vx + vy * vy)
        drag = km * v

        vx -= drag * vx * dt
        vy -= (g + drag * vy) * dt

        x += vx * dt
        y += vy * dt

        if y < 0.0:
            return xs[:i + 1], ys[:i + 1], vxs[:i + 1], vys[:i + 1]

    return xs, ys, vxs, vys


@njit(cache=True)
def _get_y_at_x(v0, angle_rad, turret_height_m, target_x, km, g, dt, n_steps):
    """
    JIT-compiled: simulate and interpolate y at a given x distance.
    Returns y value, or -1e9 if ball doesn't reach target_x.
    """
    vx = v0 * math.cos(angle_rad)
    vy = v0 * math.sin(angle_rad)
    x = 0.0
    y = turret_height_m
    prev_x = 0.0
    prev_y = turret_height_m

    for _ in range(n_steps):
        v = math.sqrt(vx * vx + vy * vy)
        drag = km * v

        vx -= drag * vx * dt
        vy -= (g + drag * vy) * dt

        prev_x = x
        prev_y = y
        x += vx * dt
        y += vy * dt

        if x >= target_x:
            # Linear interpolation
            dx = x - prev_x
            if dx < 1e-12:
                return y
            frac = (target_x - prev_x) / dx
            return prev_y + frac * (y - prev_y)

        if y < 0.0:
            return -1e9  # hit ground before reaching target

    return -1e9  # ran out of time


# ===========================================================================
# Python-level wrappers
# ===========================================================================
def simulate_trajectory(v0, angle_deg, turret_height_m, dt=0.001, t_max=4.0):
    """
    Simulate ball trajectory.
    Returns (xs, ys, vxs, vys) — arrays trimmed to valid region.
    """
    angle_rad = np.radians(angle_deg)
    n_steps = int(t_max / dt)
    return _simulate_xyvv(v0, angle_rad, turret_height_m, KM, G, dt, n_steps)


def simulate_trajectory_xy(v0, angle_deg, turret_height_m, dt=0.001, t_max=4.0):
    """
    Simulate ball trajectory (x, y only — faster when velocities not needed).
    Returns (xs, ys).
    """
    angle_rad = np.radians(angle_deg)
    n_steps = int(t_max / dt)
    xs, ys, count = _simulate_xy(v0, angle_rad, turret_height_m, KM, G, dt, n_steps)
    return xs[:count], ys[:count]


def get_y_at_x(v0, angle_deg, turret_height_m, target_x, dt=0.001, t_max=4.0):
    """
    Return the ball's y-coordinate when it reaches horizontal distance target_x.
    Returns None if the ball doesn't reach that distance.
    """
    angle_rad = math.radians(angle_deg)
    n_steps = int(t_max / dt)
    y = _get_y_at_x(v0, angle_rad, turret_height_m, target_x, KM, G, dt, n_steps)
    if y < -1e8:
        return None
    return y


# ===========================================================================
# Solvers
# ===========================================================================
def solve_for_velocity(angle_deg, turret_height_m, target_x, target_y,
                       v_min=1.0, v_max=50.0, tol=0.005):
    """
    For a given launch angle, find the velocity that makes the ball
    pass through (target_x, target_y).
    """
    def residual(v0):
        y = get_y_at_x(v0, angle_deg, turret_height_m, target_x)
        return (y - target_y) if y is not None else -target_y

    f_lo = residual(v_min)
    f_hi = residual(v_max)
    if f_lo * f_hi > 0:
        return None
    return brentq(residual, v_min, v_max, xtol=tol)


def solve_for_angle(target_x, target_y, turret_height_m,
                    angle_min=10.0, angle_max=80.0, v_max=50.0):
    """
    Find the optimal (minimum-velocity) launch angle.
    Two-pass: coarse sweep (30 pts) then fine refinement (20 pts).
    Returns (angle_deg, velocity_m_s) or (None, None).
    """
    best_angle = None
    best_vel = None

    # Coarse sweep
    for angle in np.linspace(angle_min, angle_max, 30):
        v = solve_for_velocity(angle, turret_height_m, target_x, target_y,
                               v_min=1.0, v_max=v_max)
        if v is not None and (best_vel is None or v < best_vel):
            best_vel = v
            best_angle = angle

    # Fine refinement
    if best_angle is not None:
        lo = max(angle_min, best_angle - 5.0)
        hi = min(angle_max, best_angle + 5.0)
        for angle in np.linspace(lo, hi, 20):
            v = solve_for_velocity(angle, turret_height_m, target_x, target_y,
                                   v_min=1.0, v_max=v_max)
            if v is not None and v < best_vel:
                best_vel = v
                best_angle = angle

    return best_angle, best_vel


def find_angles_for_velocity(target_x, target_y, turret_height_m, v0_ms,
                             angle_min=5.0, angle_max=85.0, n_sweep=200):
    """
    Given a fixed velocity, find all launch angles that hit (target_x, target_y).
    Returns list of (angle_deg, label_str).
    """
    angles_sweep = np.linspace(angle_min, angle_max, n_sweep)
    residuals = np.empty(n_sweep)
    for i, a in enumerate(angles_sweep):
        y = get_y_at_x(v0_ms, a, turret_height_m, target_x)
        residuals[i] = (y - target_y) if y is not None else -target_y

    solutions = []
    for i in range(len(residuals) - 1):
        if residuals[i] * residuals[i + 1] < 0:
            a_lo, a_hi = angles_sweep[i], angles_sweep[i + 1]

            def angle_residual(a):
                y = get_y_at_x(v0_ms, a, turret_height_m, target_x)
                return (y - target_y) if y is not None else -target_y

            try:
                a_sol = brentq(angle_residual, a_lo, a_hi, xtol=0.05)
                label = "low arc" if a_sol < 50 else "high arc"
                solutions.append((a_sol, label))
            except ValueError:
                pass

    return solutions


# ===========================================================================
# Warm up JIT on import (first call compiles; subsequent calls are fast)
# ===========================================================================
def _warmup():
    """Trigger Numba compilation so first real call isn't slow."""
    _get_y_at_x(10.0, 0.7, 0.5, 3.0, KM, G, 0.001, 1000)
    _simulate_xy(10.0, 0.7, 0.5, KM, G, 0.001, 1000)
    _simulate_xyvv(10.0, 0.7, 0.5, KM, G, 0.001, 1000)

_warmup()
