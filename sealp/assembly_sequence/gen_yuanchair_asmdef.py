"""
Generate YuanChair Assembly Definition (.asmdef)
=================================================

Creates a ``.asmdef`` file for the round chair using the new format.
Demonstrates: shared model library, symmetry groups, and relative poses.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sealp.assembly_sequence import AssemblyDef, PartDef, StepDef

# ── Model paths ──────────────────────────────────────────────
ASSET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "models", "yuanchair"))
LEG_STL = os.path.join(ASSET_DIR, "yuanchair-part2.stl")
SEAT_STL = os.path.join(ASSET_DIR, "yuanchair-part1.stl")


def generate() -> AssemblyDef:
    """Build the YuanChair assembly definition."""
    asm = AssemblyDef(
        name="YuanChair",
        description="Round chair assembly: 1 seat + 4 identical legs",
    )

    # ── Model library (shared models declared once) ──────────
    asm.add_model("seat_model", SEAT_STL)
    asm.add_model("leg_model", LEG_STL)

    # ── Parts ────────────────────────────────────────────────
    asm.add_part(PartDef(
        part_id="seat", name="Seat",
        model="seat_model", mass=1.5,
    ))
    for pid, name in [
        ("leg_fl", "Front-Left Leg"),
        ("leg_fr", "Front-Right Leg"),
        ("leg_bl", "Back-Left Leg"),
        ("leg_br", "Back-Right Leg"),
    ]:
        asm.add_part(PartDef(
            part_id=pid, name=name,
            model="leg_model", mass=0.3,  # same model for all legs
        ))

    # ── Symmetry groups ──────────────────────────────────────
    asm.add_symmetry_group("legs", ["leg_fl", "leg_fr", "leg_bl", "leg_br"])

    # ── Assembly steps (poses relative to parent) ────────────

    # Step 0: seat → fixture (the workspace surface)
    # Seat sits at the center of the assembly area
    asm.add_step(StepDef(
        step_id=0,
        part_id="seat",
        parent_id="fixture",
        rel_pos=np.array([0.30, 0.0, 0.0]),
        rel_rotmat=np.eye(3),
        deps=[],
        notes="Place the seat at the assembly station.",
    ))

    # Steps 1-4: legs → seat (relative to the seat position)
    # From pick_and_place_chair.py, goal positions are absolute.
    # Seat is at [0.30, 0, 0].
    # Relative to seat: leg_pos_rel = leg_pos_abs - seat_pos
    seat_pos = np.array([0.30, 0.0, 0.0])
    goal_abs = [
        np.array([0.19, 0.11, 0.02]),    # leg_fl
        np.array([0.19, -0.11, 0.02]),    # leg_fr
        np.array([0.41, -0.11, 0.02]),    # leg_bl
        np.array([0.41, 0.11, 0.02]),     # leg_br
    ]
    leg_ids = ["leg_fl", "leg_fr", "leg_bl", "leg_br"]
    leg_names = ["Front-Left Leg", "Front-Right Leg",
                 "Back-Left Leg", "Back-Right Leg"]

    for i, (pid, name) in enumerate(zip(leg_ids, leg_names)):
        rel_pos = goal_abs[i] - seat_pos  # relative to seat
        asm.add_step(StepDef(
            step_id=i + 1,
            part_id=pid,
            parent_id="seat",
            rel_pos=rel_pos,
            rel_rotmat=np.eye(3),
            deps=[0],  # seat must be placed first
            notes=f"Insert {name} into the seat.",
        ))

    # ── Validate ─────────────────────────────────────────────
    errors = asm.validate(strict=False)
    if errors:
        print("Validation warnings:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("Validation passed.")

    return asm


def main():
    asm = generate()

    out_dir = os.path.join(
        os.path.dirname(__file__), "_demo_output")
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, "yuanchair.asmdef")
    asm.save(path)

    print(f"\nSaved: {path}")
    print()
    print(asm.summary())


if __name__ == "__main__":
    main()
