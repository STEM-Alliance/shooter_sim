#!/usr/bin/env python3
"""
Generate a 2D interpolation table: (distance_ft, turret_height_in) → (angle, velocity)

Saves to shooter_table.npz for fast lookup.
"""

import numpy as np
import time

from shooter_physics import (
    HUB_HEIGHT_IN, HUB_HEIGHT_M,
    IN_TO_M, FT_TO_M, M_TO_FT,
    solve_for_angle,
)


def main():
    dist_min_ft, dist_max_ft, dist_step_ft = 3.0, 30.0, 0.5
    turret_min_in, turret_max_in, turret_step_in = 18.0, 42.0, 1.0

    distances_ft = np.arange(dist_min_ft, dist_max_ft + dist_step_ft / 2, dist_step_ft)
    turret_heights_in = np.arange(turret_min_in, turret_max_in + turret_step_in / 2, turret_step_in)

    n_dist = len(distances_ft)
    n_turret = len(turret_heights_in)
    total = n_dist * n_turret

    print(f"Generating {n_dist} x {n_turret} = {total} grid points")
    print(f"  Distance: {dist_min_ft}–{dist_max_ft} ft (step {dist_step_ft})")
    print(f"  Turret:   {turret_min_in}–{turret_max_in} in (step {turret_step_in})")
    print()

    angle_table = np.full((n_dist, n_turret), np.nan)
    vel_table = np.full((n_dist, n_turret), np.nan)

    t0 = time.time()
    count = 0

    for j, th_in in enumerate(turret_heights_in):
        th_m = th_in * IN_TO_M
        for i, d_ft in enumerate(distances_ft):
            count += 1
            d_m = d_ft * FT_TO_M
            angle, vel = solve_for_angle(d_m, HUB_HEIGHT_M, th_m)
            if angle is not None:
                angle_table[i, j] = angle
                vel_table[i, j] = vel * M_TO_FT
            if count % 100 == 0:
                elapsed = time.time() - t0
                rate = count / elapsed
                eta = (total - count) / rate
                print(f"  [{count:5d}/{total}]  "
                      f"turret={th_in:.0f}in  dist={d_ft:.1f}ft  "
                      f"{rate:.1f} pts/s  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({total/elapsed:.1f} pts/s)")

    outfile = "shooter_table.npz"
    np.savez_compressed(
        outfile,
        distances_ft=distances_ft,
        turret_heights_in=turret_heights_in,
        hub_height_in=np.array([HUB_HEIGHT_IN]),
        angle_deg=angle_table,
        velocity_fps=vel_table,
    )
    print(f"Saved to {outfile}")
    print(f"  distances_ft:      {distances_ft.shape}")
    print(f"  turret_heights_in: {turret_heights_in.shape}")
    print(f"  angle_deg:         {angle_table.shape}")
    print(f"  velocity_fps:      {vel_table.shape}")


if __name__ == "__main__":
    main()
