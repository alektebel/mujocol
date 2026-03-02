# MujoCol - ASCII MuJoCo-like Physics Engine
# Makefile for all phases

CC = gcc
CFLAGS = -O3 -Wall
LDFLAGS = -lm

# All phase executables
TARGETS = phase1_term phase2_math3d phase3_sdf_render phase4_camera \
          phase5_physics phase6_scene phase7_sim phase8_rl phase9_robot_sdf

.PHONY: all clean test

all: $(TARGETS)

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
	rm -f $(TARGETS)

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
	@echo "  run-reacher  - Phase 7: Reacher environment
  run-rl       - Phase 8: RL CartPole trainer (exercise skeleton)
  run-robot    - Phase 9: Quadruped robot SDF renderer (exercise skeleton)
  run-rl-sol   - Phase 8: RL CartPole trainer SOLUTION
  run-robot-sol- Phase 9: Quadruped robot SDF renderer SOLUTION"
