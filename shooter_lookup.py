#!/usr/bin/env python3
"""
FRC 2026 Shooter Lookup Tool
==============================

Commands:
  recommend  — Given distance → optimal angle & velocity
  angle      — Given distance + velocity → required angle(s)
  table      — Print a full lookup table

Usage:
  python shooter_lookup.py recommend -d 15
  python shooter_lookup.py angle -d 15 -v 28
  python shooter_lookup.py table --dist-min 5 --dist-max 25 --step 2
"""

import argparse
import sys
import numpy as np

from shooter_physics import (
    BALL_DIAMETER_IN, BALL_MASS_KG, CD_SPHERE,
    HUB_HEIGHT_IN, DEFAULT_TURRET_HEIGHT_IN,
    IN_TO_M, FT_TO_M, M_TO_FT,
    solve_for_angle, find_angles_for_velocity,
)


# ===========================================================================
# Mode 1: Recommend angle & velocity for a given distance
# ===========================================================================
def recommend(distance_ft, turret_height_in=DEFAULT_TURRET_HEIGHT_IN,
              hub_height_in=HUB_HEIGHT_IN):
    """
    Returns (angle_deg, velocity_fps, velocity_ms) or (None, None, None).
    """
    turret_h = turret_height_in * IN_TO_M
    hub_h = hub_height_in * IN_TO_M
    target_x = distance_ft * FT_TO_M

    angle, vel = solve_for_angle(target_x, hub_h, turret_h)
    if angle is None:
        return None, None, None
    return angle, vel * M_TO_FT, vel


# ===========================================================================
# Mode 2: Find angle given distance and velocity
# ===========================================================================
def find_angle(distance_ft, velocity_fps, turret_height_in=DEFAULT_TURRET_HEIGHT_IN,
               hub_height_in=HUB_HEIGHT_IN):
    """
    Returns list of (angle_deg, label_str).
    """
    turret_h = turret_height_in * IN_TO_M
    hub_h = hub_height_in * IN_TO_M
    target_x = distance_ft * FT_TO_M
    v0 = velocity_fps * FT_TO_M

    return find_angles_for_velocity(target_x, hub_h, turret_h, v0)


# ===========================================================================
# CLI handlers
# ===========================================================================
def cmd_recommend(args):
    distance = args.distance
    turret_h = args.turret_height
    hub_h = args.hub_height

    angle, vel_fps, vel_ms = recommend(distance, turret_h, hub_h)

    print()
    print("=" * 55)
    print("  RECOMMENDED SHOT PARAMETERS")
    print("=" * 55)
    print(f"  Distance to hub : {distance:.1f} ft")
    print(f"  Turret height   : {turret_h:.1f} in")
    print(f"  Hub height      : {hub_h:.1f} in")
    print("-" * 55)

    if angle is None:
        print("  No solution found for this distance.")
        print("     Try a shorter distance or higher turret.")
    else:
        print(f"  Launch angle   : {angle:.1f} deg")
        print(f"  Launch velocity: {vel_fps:.1f} ft/s  ({vel_ms:.2f} m/s)")

        wheel_diameters = [4, 6]
        print()
        print("  Wheel RPM (direct-contact single-wheel shooter):")
        for d_in in wheel_diameters:
            circumference_ft = (d_in * np.pi) / 12.0
            rpm = vel_fps / circumference_ft * 60
            print(f"    {d_in}\" wheel: {rpm:.0f} RPM")

    print("=" * 55)
    print()


def cmd_angle(args):
    distance = args.distance
    velocity = args.velocity
    turret_h = args.turret_height
    hub_h = args.hub_height

    opt_angle, opt_vel_fps, _ = recommend(distance, turret_h, hub_h)
    solutions = find_angle(distance, velocity, turret_h, hub_h)

    print()
    print("=" * 55)
    print("  ANGLE CALCULATION FROM KNOWN VELOCITY")
    print("=" * 55)
    print(f"  Distance to hub  : {distance:.1f} ft")
    print(f"  Launch velocity  : {velocity:.1f} ft/s  ({velocity * FT_TO_M:.2f} m/s)")
    print(f"  Turret height    : {turret_h:.1f} in")
    print(f"  Hub height       : {hub_h:.1f} in")
    print("-" * 55)

    if opt_angle is not None:
        print(f"  (Optimal: {opt_angle:.1f} deg @ {opt_vel_fps:.1f} ft/s minimum)")
        if velocity < opt_vel_fps * 0.99:
            deficit = opt_vel_fps - velocity
            print(f"  WARNING: Velocity is {deficit:.1f} ft/s below minimum needed!")
        print("-" * 55)

    if not solutions:
        print("  No valid angle found.")
        print("     The ball cannot reach the hub at this velocity.")
        if opt_vel_fps is not None:
            print(f"     Minimum velocity needed: {opt_vel_fps:.1f} ft/s")
    else:
        for i, (angle, label) in enumerate(solutions):
            print(f"  Solution {i+1} ({label}): {angle:.1f} deg")

        if len(solutions) >= 2:
            print()
            print(f"  Recommendation: Use the LOW ARC ({solutions[0][0]:.1f} deg)")
            print(f"     - Faster flight time, less affected by wind")
            print(f"     - More consistent and easier to tune")
        elif len(solutions) == 1:
            print()
            print(f"  Use {solutions[0][0]:.1f} deg ({solutions[0][1]})")

    print("=" * 55)
    print()


def cmd_table(args):
    turret_h = args.turret_height
    hub_h = args.hub_height
    distances = np.arange(args.dist_min, args.dist_max + args.step / 2, args.step)

    print()
    print("=" * 65)
    print("  SHOOTER LOOKUP TABLE")
    print(f"  Turret: {turret_h:.0f} in  |  Hub: {hub_h:.0f} in  |  "
          f"Ball: {BALL_DIAMETER_IN}\" {BALL_MASS_KG:.3f} kg  |  Cd={CD_SPHERE}")
    print("=" * 65)
    print(f"  {'Dist (ft)':>10}  {'Angle (deg)':>11}  {'Vel (ft/s)':>12}  {'Vel (m/s)':>10}")
    print("  " + "-" * 61)

    for d in distances:
        angle, vel_fps, vel_ms = recommend(d, turret_h, hub_h)
        if angle is not None:
            print(f"  {d:10.1f}  {angle:11.1f}  {vel_fps:12.1f}  {vel_ms:10.2f}")
        else:
            print(f"  {d:10.1f}  {'N/A':>11}  {'N/A':>12}  {'N/A':>10}")

    print("=" * 65)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="FRC 2026 Shooter Lookup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s recommend --distance 15
  %(prog)s angle --distance 15 --velocity 28
  %(prog)s table --dist-min 5 --dist-max 25 --step 2
        """
    )

    parser.add_argument("--turret-height", type=float, default=DEFAULT_TURRET_HEIGHT_IN,
                        help=f"Turret height in inches (default: {DEFAULT_TURRET_HEIGHT_IN})")
    parser.add_argument("--hub-height", type=float, default=HUB_HEIGHT_IN,
                        help=f"Hub height in inches (default: {HUB_HEIGHT_IN})")

    sub = parser.add_subparsers(dest="command", help="Command")

    p_rec = sub.add_parser("recommend", help="Recommend angle & velocity for a distance")
    p_rec.add_argument("--distance", "-d", type=float, required=True,
                       help="Distance to hub in feet")

    p_ang = sub.add_parser("angle", help="Find angle given distance and velocity")
    p_ang.add_argument("--distance", "-d", type=float, required=True,
                       help="Distance to hub in feet")
    p_ang.add_argument("--velocity", "-v", type=float, required=True,
                       help="Achievable launch velocity in ft/s")

    p_tbl = sub.add_parser("table", help="Print a full lookup table")
    p_tbl.add_argument("--dist-min", type=float, default=5.0)
    p_tbl.add_argument("--dist-max", type=float, default=25.0)
    p_tbl.add_argument("--step", type=float, default=1.0)

    args = parser.parse_args()

    if args.command == "recommend":
        cmd_recommend(args)
    elif args.command == "angle":
        cmd_angle(args)
    elif args.command == "table":
        cmd_table(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
