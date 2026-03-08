# MujoCol - ASCII MuJoCo-like Physics Engine
# Makefile for all phases
#
# ── Architecture notes ────────────────────────────────────────────
# By default this builds for the host machine (x86-64 or native).
# To cross-compile for 64-bit ARM (aarch64 / Apple Silicon / Pi 5):
#
#   make aarch64
#
# This uses the aarch64-linux-gnu-gcc cross-compiler, which can be
# installed on Ubuntu/Debian with:
#   sudo apt install gcc-aarch64-linux-gnu
#
# The resulting binaries run on any aarch64 Linux system (Raspberry Pi
# 4/5, NVIDIA Jetson, AWS Graviton, etc.) unchanged — the code uses
# only standard POSIX APIs and avoids x86-specific intrinsics.
# ─────────────────────────────────────────────────────────────────

CC      = gcc
CFLAGS  = -O3 -Wall
LDFLAGS = -lm

# aarch64 cross-compiler settings
CC_AARCH64      = aarch64-linux-gnu-gcc
CFLAGS_AARCH64  = -O3 -Wall
# Suffix appended to every binary name when cross-compiling
AARCH64_SUFFIX  = .aarch64

# All phase executables
TARGETS = phase1_term phase2_math3d phase3_sdf_render phase4_camera \
          phase5_physics phase6_scene phase7_sim phase8_rl phase9_robot_sdf

# Sub-phase exercise files (compiled individually, no install target)
SUBPHASE_SRCS = \
  phase1a_raw_mode.c phase1b_screen_buf.c phase1c_animation.c \
  phase2a_vec3.c     phase2b_mat4.c       phase2c_quat.c      \
  phase3a_sdf_prims.c phase3b_ray_march.c phase3c_shading.c   \
  phase4a_orbit_pos.c phase4b_orbit_input.c phase4c_orbit_smooth.c \
  phase5a_body_init.c phase5b_collision.c  phase5c_response.c  \
  phase6a_xml_parse.c phase6b_urdf_load.c  phase6c_fk_render.c

SUBPHASE_BINS  = $(SUBPHASE_SRCS:.c=)
AARCH64_BINS   = $(patsubst %,%$(AARCH64_SUFFIX),$(TARGETS)) \
                 $(patsubst %,%$(AARCH64_SUFFIX),$(SUBPHASE_BINS))

.PHONY: all subphases aarch64 aarch64-subphases clean test

all: $(TARGETS)

# ── Sub-phase exercise builds ──────────────────────────────────────
# Build every sub-phase exercise file individually.
# Usage: make subphases
subphases: $(SUBPHASE_BINS)

$(SUBPHASE_BINS): %: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ── aarch64 cross-compilation ─────────────────────────────────────
# Builds all main-phase binaries suffixed with .aarch64.
# Requires: sudo apt install gcc-aarch64-linux-gnu
# Usage: make aarch64
aarch64: $(patsubst %,%$(AARCH64_SUFFIX),$(TARGETS))

# aarch64 sub-phase binaries
aarch64-subphases: $(patsubst %,%$(AARCH64_SUFFIX),$(SUBPHASE_BINS))

%$(AARCH64_SUFFIX): %.c
	$(CC_AARCH64) $(CFLAGS_AARCH64) -o $@ $< $(LDFLAGS)

# Phase 1: Terminal Framework
phase1_term: phase1_term.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Phase 2: 3D Math Library
phase2_math3d: phase2_math3d.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Phase 3: SDF Ray Marcher
phase3_sdf_render: phase3_sdf_render.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Phase 4: Interactive Camera
phase4_camera: phase4_camera.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Phase 5: Rigid Body Physics
phase5_physics: phase5_physics.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Phase 6: Scene Graph + URDF Parser
phase6_scene: phase6_scene.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Phase 7: Full Simulation Loop
phase7_sim: phase7_sim.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Phase 8: RL Training (CartPole Stick Balancer)
phase8_rl: phase8_rl.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Phase 9: Quadruped Robot SDF Renderer
phase9_robot_sdf: phase9_robot_sdf.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Solutions (fully implemented — run these to see the expected output)
phase8_rl_sol: solutions/phase8_rl.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

phase9_robot_sdf_sol: solutions/phase9_robot_sdf.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Run math tests
test: phase2_math3d
	./phase2_math3d

# Clean all binaries
clean:
	rm -f $(TARGETS) $(SUBPHASE_BINS) $(AARCH64_BINS)

# Individual run targets
run-term: phase1_term
	./phase1_term

run-math: phase2_math3d
	./phase2_math3d

run-render: phase3_sdf_render
	./phase3_sdf_render

run-camera: phase4_camera
	./phase4_camera

run-physics: phase5_physics
	./phase5_physics

run-scene: phase6_scene
	./phase6_scene

run-cartpole: phase7_sim
	./phase7_sim cartpole

run-pendulum: phase7_sim
	./phase7_sim pendulum

run-reacher: phase7_sim
	./phase7_sim reacher

run-rl: phase8_rl
	./phase8_rl

run-robot: phase9_robot_sdf
	./phase9_robot_sdf

run-rl-sol: phase8_rl_sol
	./phase8_rl_sol

run-robot-sol: phase9_robot_sdf_sol
	./phase9_robot_sdf_sol

# Help
help:
	@echo "MujoCol - ASCII MuJoCo-like Physics Engine"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build all phases"
	@echo "  clean        - Remove all binaries"
	@echo "  test         - Run math library tests"
	@echo ""
	@echo "Run individual phases:"
	@echo "  run-term     - Phase 1: Terminal framework demo"
	@echo "  run-math     - Phase 2: Math library tests"
	@echo "  run-render   - Phase 3: Static SDF scene"
	@echo "  run-camera   - Phase 4: Interactive camera"
	@echo "  run-physics  - Phase 5: Rigid body simulation"
	@echo "  run-scene    - Phase 6: URDF robot viewer"
	@echo "  run-cartpole - Phase 7: CartPole environment"
	@echo "  run-pendulum - Phase 7: Pendulum environment"
	@echo "  run-reacher  - Phase 7: Reacher environment"
	@echo "  run-rl       - Phase 8: RL CartPole trainer (exercise skeleton)"
	@echo "  run-robot    - Phase 9: Quadruped robot SDF renderer (exercise skeleton)"
	@echo "  run-rl-sol   - Phase 8: RL CartPole trainer SOLUTION"
	@echo "  run-robot-sol - Phase 9: Quadruped robot SDF renderer SOLUTION"
	@echo ""
	@echo "Sub-phase exercises (individual files):"
	@echo "  subphases          - Build all sub-phase exercise files (host arch)"
	@echo "  aarch64            - Cross-compile all phases for aarch64 Linux"
	@echo "  aarch64-subphases  - Cross-compile all sub-phase files for aarch64"
	@echo ""
	@echo "aarch64 cross-compiler: sudo apt install gcc-aarch64-linux-gnu"
