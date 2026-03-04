#!/usr/bin/env python3
"""
FRC 2026 Turret Ball Shooter Simulation — graphical analysis
=============================================================

Sweeps distances and generates matplotlib plots showing:
  1. Optimal launch angle vs distance
  2. Required launch velocity vs distance
  3. Sample ball trajectories
  4. Drag vs no-drag comparison
  5. Velocity penalty from air resistance
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from shooter_physics import (
    G, CD_SPHERE, KM, K_DRAG,
    BALL_DIAMETER_IN, BALL_DIAMETER_M, BALL_MASS_KG,
    HUB_HEIGHT_IN, HUB_HEIGHT_M,
    DEFAULT_TURRET_HEIGHT_IN, DEFAULT_TURRET_HEIGHT_M,
    DIST_MIN_FT, DIST_MAX_FT, NUM_DISTANCES,
    IN_TO_M, FT_TO_M, M_TO_IN, M_TO_FT,
    simulate_trajectory, solve_for_angle,
)


# ===========================================================================
# Distance sweep
# ===========================================================================
def sweep_distances(turret_height_m, target_height_m,
                    dist_min_ft, dist_max_ft, n):
    distances_ft = np.linspace(dist_min_ft, dist_max_ft, n)

    angles = np.empty(n)
    velocities_ms = np.empty(n)
    velocities_fps = np.empty(n)

    for i, d_ft in enumerate(distances_ft):
        d_m = d_ft * FT_TO_M
        print(f"  [{i+1:3d}/{n}] dist = {d_ft:5.1f} ft ... ", end="", flush=True)
        angle, vel = solve_for_angle(d_m, target_height_m, turret_height_m)
        if angle is not None:
            angles[i] = angle
            velocities_ms[i] = vel
            velocities_fps[i] = vel * M_TO_FT
            print(f"angle = {angle:5.1f}°   v = {vel:5.2f} m/s  ({vel*M_TO_FT:5.1f} ft/s)")
        else:
            angles[i] = velocities_ms[i] = velocities_fps[i] = np.nan
            print("NO SOLUTION")

    return distances_ft, angles, velocities_ms, velocities_fps


# ===========================================================================
# Plotting
# ===========================================================================
def plot_results(distances_ft, angles, velocities_ms, velocities_fps,
                 turret_height_m, target_height_m):
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        "FRC 2026 — Turret Ball Shooter Simulation\n"
        f"Turret: {turret_height_m*M_TO_IN:.1f} in  |  "
        f"Hub: {target_height_m*M_TO_IN:.1f} in  |  "
        f"Ball: {BALL_DIAMETER_IN:.2f} in, {BALL_MASS_KG:.3f} kg  |  "
        f"Cd = {CD_SPHERE}",
        fontsize=13, fontweight='bold'
    )

    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.30,
                           top=0.91, bottom=0.06, left=0.08, right=0.96)
    mask = ~np.isnan(angles)

    # Panel 1: Angle vs Distance
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(distances_ft[mask], angles[mask], 'b-o', markersize=3, linewidth=1.5)
    ax1.set_xlabel("Distance to Hub (ft)")
    ax1.set_ylabel("Launch Angle (°)")
    ax1.set_title("Optimal Launch Angle vs Distance")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Velocity vs Distance
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(distances_ft[mask], velocities_fps[mask], 'r-o', markersize=3, linewidth=1.5)
    ax2_m = ax2.twinx()
    ax2_m.set_ylabel("Velocity (m/s)", color='gray')
    ymin, ymax = np.nanmin(velocities_ms), np.nanmax(velocities_ms)
    margin = (ymax - ymin) * 0.1 if ymax > ymin else 1.0
    ax2_m.set_ylim(ymin - margin, ymax + margin)
    ax2.set_xlabel("Distance to Hub (ft)")
    ax2.set_ylabel("Launch Velocity (ft/s)")
    ax2.set_title("Required Launch Velocity vs Distance")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Sample trajectories
    ax3 = fig.add_subplot(gs[1, :])
    valid_dists = distances_ft[mask]
    sample_dists_ft = np.linspace(valid_dists[0], valid_dists[-1], 6)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sample_dists_ft)))

    for d_ft, color in zip(sample_dists_ft, colors):
        d_m = d_ft * FT_TO_M
        angle, vel = solve_for_angle(d_m, target_height_m, turret_height_m)
        if angle is None:
            continue
        x_arr, y_arr, _, _ = simulate_trajectory(vel, angle, turret_height_m)
        ax3.plot(x_arr * M_TO_FT, y_arr * M_TO_IN, color=color, linewidth=2,
                 label=f"{d_ft:.0f} ft — {angle:.1f}° @ {vel*M_TO_FT:.0f} ft/s")

    ax3.axhline(y=target_height_m * M_TO_IN, color='green', linestyle='--',
                linewidth=1.5, alpha=0.7,
                label=f"Hub height ({target_height_m*M_TO_IN:.0f} in)")
    ax3.axhline(y=turret_height_m * M_TO_IN, color='orange', linestyle=':',
                linewidth=1.5, alpha=0.7,
                label=f"Turret height ({turret_height_m*M_TO_IN:.0f} in)")
    ax3.set_xlabel("Horizontal Distance (ft)")
    ax3.set_ylabel("Height (in)")
    ax3.set_title("Ball Trajectories (with air resistance)")
    ax3.legend(fontsize=8, loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    # Panel 4: Drag effect comparison
    ax4 = fig.add_subplot(gs[2, 0])
    mid_idx = len(valid_dists) // 2
    mid_dist_m = valid_dists[mid_idx] * FT_TO_M
    angle_drag, vel_drag = solve_for_angle(mid_dist_m, target_height_m, turret_height_m)

    if angle_drag is not None:
        x1, y1, _, _ = simulate_trajectory(vel_drag, angle_drag, turret_height_m)
        ax4.plot(x1 * M_TO_FT, y1 * M_TO_IN, 'b-', linewidth=2,
                 label=f"With drag (v₀={vel_drag*M_TO_FT:.1f} ft/s)")

        angle_rad = np.radians(angle_drag)
        vx0 = vel_drag * np.cos(angle_rad)
        vy0 = vel_drag * np.sin(angle_rad)
        t_nd = np.linspace(0, 3, 2000)
        x_nd = vx0 * t_nd
        y_nd = turret_height_m + vy0 * t_nd - 0.5 * G * t_nd ** 2
        m = y_nd >= 0
        ax4.plot(x_nd[m] * M_TO_FT, y_nd[m] * M_TO_IN, 'r--', linewidth=2,
                 label="Without drag (same v₀)")
        ax4.axhline(y=target_height_m * M_TO_IN, color='green',
                    linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_title(f"Drag Effect @ {valid_dists[mid_idx]:.0f} ft, {angle_drag:.1f}°")

    ax4.set_xlabel("Horizontal Distance (ft)")
    ax4.set_ylabel("Height (in)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)

    # Panel 5: Velocity with vs without drag
    ax5 = fig.add_subplot(gs[2, 1])
    delta_y = target_height_m - turret_height_m
    nodrag_vels_fps = np.full_like(distances_ft, np.nan)

    for i, (d_ft, angle) in enumerate(zip(distances_ft, angles)):
        if np.isnan(angle):
            continue
        d_m = d_ft * FT_TO_M
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        tan_a = np.tan(angle_rad)
        denom = d_m * tan_a - delta_y
        if denom <= 0:
            continue
        v0_nd = (d_m / cos_a) * np.sqrt(G / (2 * denom))
        nodrag_vels_fps[i] = v0_nd * M_TO_FT

    ax5.plot(distances_ft[mask], velocities_fps[mask], 'b-o', markersize=3,
             linewidth=1.5, label="With air resistance")
    nd_mask = ~np.isnan(nodrag_vels_fps)
    ax5.plot(distances_ft[nd_mask], nodrag_vels_fps[nd_mask], 'r--o', markersize=3,
             linewidth=1.5, label="No air resistance")
    both = mask & nd_mask
    if np.any(both):
        ax5.fill_between(distances_ft[both], nodrag_vels_fps[both],
                         velocities_fps[both], alpha=0.15, color='purple',
                         label="Drag penalty")
    ax5.set_xlabel("Distance to Hub (ft)")
    ax5.set_ylabel("Launch Velocity (ft/s)")
    ax5.set_title("Velocity: With vs Without Air Resistance")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    return fig


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="FRC 2026 Turret Ball Shooter Simulation"
    )
    parser.add_argument("--turret-height", type=float, default=DEFAULT_TURRET_HEIGHT_IN,
                        help=f"Turret height in inches (default: {DEFAULT_TURRET_HEIGHT_IN})")
    parser.add_argument("--hub-height", type=float, default=HUB_HEIGHT_IN,
                        help=f"Hub target height in inches (default: {HUB_HEIGHT_IN})")
    parser.add_argument("--dist-min", type=float, default=DIST_MIN_FT,
                        help=f"Minimum distance in feet (default: {DIST_MIN_FT})")
    parser.add_argument("--dist-max", type=float, default=DIST_MAX_FT,
                        help=f"Maximum distance in feet (default: {DIST_MAX_FT})")
    parser.add_argument("--num-points", type=int, default=NUM_DISTANCES,
                        help=f"Number of sample points (default: {NUM_DISTANCES})")
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to file instead of showing")
    args = parser.parse_args()

    turret_h_m = args.turret_height * IN_TO_M
    hub_h_m = args.hub_height * IN_TO_M

    print("=" * 65)
    print("  FRC 2026 — Turret Ball Shooter Simulation")
    print("=" * 65)
    print(f"  Turret height : {args.turret_height:6.1f} in  ({turret_h_m:.4f} m)")
    print(f"  Hub height    : {args.hub_height:6.1f} in  ({hub_h_m:.4f} m)")
    print(f"  Height delta  : {args.hub_height - args.turret_height:6.1f} in  "
          f"({hub_h_m - turret_h_m:.4f} m)")
    print(f"  Ball diameter : {BALL_DIAMETER_IN:6.2f} in  ({BALL_DIAMETER_M:.4f} m)")
    print(f"  Ball mass     : {BALL_MASS_KG:.4f} kg  ({BALL_MASS_KG/0.453592:.3f} lb)")
    print(f"  Drag coeff Cd : {CD_SPHERE}")
    print(f"  k = ½CdρA    : {K_DRAG:.6f} N·s²/m²")
    print(f"  k/m           : {KM:.6f} 1/m")
    print(f"  Distance range: {args.dist_min:.0f} – {args.dist_max:.0f} ft")
    print(f"  Sample points : {args.num_points}")
    print("=" * 65)
    print()

    print("Computing optimal angle & velocity for each distance...")
    distances_ft, angles, velocities_ms, velocities_fps = sweep_distances(
        turret_h_m, hub_h_m, args.dist_min, args.dist_max, args.num_points
    )

    print()
    print("=" * 65)
    print(f"  {'Dist (ft)':>10} {'Angle (°)':>10} {'Vel (ft/s)':>12} {'Vel (m/s)':>10}")
    print("  " + "-" * 61)
    for d, a, vf, vm in zip(distances_ft, angles, velocities_fps, velocities_ms):
        if np.isnan(a):
            print(f"  {d:10.1f} {'N/A':>10} {'N/A':>12} {'N/A':>10}")
        else:
            print(f"  {d:10.1f} {a:10.1f} {vf:12.1f} {vm:10.2f}")
    print("=" * 65)

    print("\nGenerating plots...")
    fig = plot_results(distances_ft, angles, velocities_ms, velocities_fps,
                       turret_h_m, hub_h_m)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {args.save}")
    else:
        plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
