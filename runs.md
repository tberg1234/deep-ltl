# Plotting

```bash
python plot_logs.py --env <env> --names <run name>
```

Or compare multiple runs:

```bash
python plot_logs.py --names my_run my_run2 my_run3
```

Options:

--env — defaults to SafetyBallNav-v0
--seeds — which seeds to include, e.g. --seeds 1 2 3
--smooth 20 — rolling average window (default 10)
The script plots 9 panels: success rate, violation rate, return, entropy, policy/value loss, steps/episode, gradient norm, and per-task success for each object type.

plots save to plots/<env>/


# CONTINUOUS ENVIRONMENT
## Install bullet_safety_gym
pip install -e src/envs/bullet_safety_gym/

g flags:

Scenario	Flags
Both randomized (default)	(nothing extra)
Agent only	--no-randomize_objects
Objects only	--no-randomize_agent
Neither	--no-randomize_agent --no-randomize_objects
The randomize_objects=False implementation works by capturing obstacle positions on the first reset and calling set_locs(), which hooks into set_obstacles() so all subsequent resets reuse those same positions. Each BulletSafetyGymWrapper instance gets its own fixed layout (different envs in multi-process training will each have their own randomly-drawn fixed layout).

## Train
python run_bullet.py --name my_run --seed 1 --device cpu --num_procs 4

## Fine Tune

 To fine-tune the existing my_run policy on static agent position:
python run_bullet.py \
  --name my_run_finetune --seed 1 --device cpu --num_procs 8 \
  --no-randomize_agent \
  --finetune_from my_run
What this does:

Creates a new experiment my_run_finetune (fresh log, step count starts at 0, curriculum restarts at stage 0)
Loads the model weights from my_run seed 1 (tries load_best_model(), falls back to final status.pth)
Trains with obstacles randomized but agent fixed at (0,0)
If you want to finetune from a different seed, add --finetune_seed 2.

## Eval
 The test sequences for RepoMan need to be added to eval_test_tasks_finite.py in the env_to_tasks dict, using the 5 compound propositions

To visualize an existing policy:
```bash
 PYTHONPATH=src/ python src/evaluation/simulate.py \
  --env SafetyBallNav-v0 --exp my_run --seed 1 \
  --formula "F blue_box" --render --num_episodes 1
```
  
## Run Record:

### my_run:
ExplicitCurriculumStage 0 = "min" (default)

```bash
python run_bullet.py --name my_run --seed 1 --device cpu --num_procs 4
```

Training took 4:38:41

second round

```bash
python run_bullet.py --name my_run --seed 1 --device cpu --num_procs 4
```
Training took 4:30:10.

### my_run2:
ExplicitCurriculumStage 0 = "mean" 

```bash
python run_bullet.py --name my_run2 --seed 1 --device cpu --num_procs 8
```

Training took 8:22:43.

### my_run3:
ExplicitCurriculumStage 0 = "min" (default), static objects and agent

```bash
 python run_bullet.py --name my_run3 --seed 1 --device cpu --num_procs 8 --no_randomize_agent --no_randomize_objects
```

Training took 8:24:15.


# VIDEO GAME WORLD

## Train
```bash
python run_repoman.py --name my_run --seed 1 --device cpu --num_procs 8
```

# default — both randomized (training as normal)
```bash
python run_repoman.py --name my_run --seed 1 --device cpu
```

# fix object positions, randomize agent only
```bash
python run_repoman.py --name fixed_obj --seed 1 --device cpu --no-randomize_objects
```

# fix everything
```bash
python run_repoman.py --name fixed_all --seed 1 --device cpu --no-randomize_agent --no-randomize_objects
```
### Training Spec
REPOMAN_CURRICULUM defines the 4 training stages, each using a function from repoman_sequence_samplers.py:

Stage 0: all_reach_tasks_repoman(1) — reach any single collectible
Stage 1: all_reach_avoid_tasks_repoman(1) — reach one, avoid another
Stage 2: sample_reach_avoid_repoman(1, (1,2), (0,2)) — random reach-avoid
Stage 3: sample_reach_avoid_repoman(2, (1,2), (1,2)) — longer sequences

## Evaluation

### Specs
For evaluation, the specs come from LTL formula strings in simulate.py (--formula "F square_blue") which are parsed and converted to LDBASequence via FixedSampler. Test formula sets would live in eval_test_tasks_finite.py under env_to_tasks['RepoMan-v0'] — which doesn't exist yet.

### Visualization
PYTHONPATH=src/ python src/evaluation/simulate.py \
  --env RepoMan-v0 --exp my_run --seed 1 \
  --formula "F square_blue" --render --num_episodes 1
The proposition names are the compound object names: square_purple, circle_purple, square_beige, circle_beige, square_blue, circle_blue.

  --save_gifs with --render
  --gif_dir overrides root directory
  '--no-randomize_agent',    # fix agent start
  '--no-randomize_objects',  # fix object positions

## Example formulas:

"F square_blue" — reach the blue square
"F circle_purple & !square_blue U circle_purple" — reach purple circle while avoiding blue square
"F square_beige & F circle_blue" — reach beige square then blue circle

video game world:

Saved training status
Training took 1:42:45.

