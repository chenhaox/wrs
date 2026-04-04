"""Quick smoke test for SEALP config & colliders."""
import os
import sys

# Test 1: Load sample config
from sealp.config import load_config, ROBOT_REGISTRY

config_path = os.path.join(os.path.dirname(__file__), "config", "sample_config.yaml")
cfg = load_config(config_path)
print(f"[OK] Config loaded: project={cfg.project_name}, robot={cfg.robot.type}")
print(f"     Obstacles: {len(cfg.obstacle_defs)}")
for o in cfg.obstacle_defs:
    print(f"       - {o['name']} ({o['type']})")
print(f"     Known robots: {sorted(ROBOT_REGISTRY.keys())}")

# Test 2: Setup (robot + collision world)
from sealp.config import setup_from_config

setup = setup_from_config(config_path, config=cfg)
print(f"\n[OK] Setup complete:")
print(f"     Robot: {type(setup.robot).__name__}")
print(f"     Sequence: {setup.sequence}")
print(f"     Collision obstacles: {len(setup.collision_world.obstacle_list)}")

# Test 3: Add runtime obstacle
setup.collision_world.user_obstacles.add_box(
    "test_block", extent=[0.1, 0.1, 0.1], pos=[0.5, 0, 0.05]
)
print(f"     After adding user obstacle: {len(setup.collision_world.obstacle_list)}")

# Test 4: Sequence validation (expect error on missing model_path)
from sealp.assembly_sequence import AssemblySequence, AssemblyPart, AssemblyStep
import numpy as np

seq = AssemblySequence(name="test")
seq.add_part(AssemblyPart(
    part_id="p1", name="Part1", model_path="nonexistent_file.stl",
))
seq.add_step(AssemblyStep(step_id=0, part_id="p1"))
try:
    seq.validate(strict=True)
    print("\n[FAIL] Validation should have raised ValueError for missing file!")
    sys.exit(1)
except ValueError as e:
    print(f"\n[OK] Validation correctly caught missing model_path:")
    print(f"     {e}")

# Test 5: Unknown robot type
from sealp.config.sealp_config import RobotConfig, SEALPConfig
bad_cfg = SEALPConfig(robot=RobotConfig(type="unknown_robot"))
try:
    setup_from_config("dummy", config=bad_cfg)
    print("\n[FAIL] Should have raised for unknown robot type!")
    sys.exit(1)
except KeyError:
    print(f"\n[OK] Unknown robot type correctly rejected")

print("\n✓ All smoke tests passed.")
