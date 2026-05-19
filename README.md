# meta-env

The always-on environment every brain lives in.

A brain never exists "outside" an environment. Between task environments, it lives in the `MetaEnvironment`. Entering and exiting task environments happens from within the MetaEnvironment — the brain navigates by acting on field gradient observations, not by following a designer-prescribed curriculum.

## Installation

```bash
pip install meta-env
```

Depends on `ecoframe`.

## Concept

In standard RL, agents are launched into environments by external code. Here, the brain is always in some environment. The MetaEnvironment is the environment it inhabits when no task environment is active.

**Curriculum is navigation.** The brain observes the field — which environments are nearby, how much each one is currently teaching, how full each one is — and acts to navigate toward the most informative one. No schedule, no designer-prescribed ordering. Pure prediction error.

```
Brain in MetaEnvironment
  ↓ observes field gradient (curiosity, load, direction to each env)
  ↓ acts to navigate
  → auto-enters task env when curiosity > threshold
  → runs task env until done or mastered
  → returns to MetaEnvironment
```

## Usage

```python
from ecoframe.field import Field
from meta_env.environment import MetaEnvironment

field    = Field()
meta_env = MetaEnvironment(field=field, enter_threshold=2.0, verbose=True)

# Register task environments
meta_env.register_env(roundabout_env)
meta_env.register_env(highway_env)

# Wire to BrainRegistry for cert tracking (optional)
meta_env.registry = registry

# Use as a standard EnvironmentProtocol
from ecoframe.training_engine import TrainingEngine

engine = TrainingEngine(brain, meta_env)
for step, metrics in engine.run(n_steps=2_000_000):
    log(step, metrics)
```

`MetaEnvironment` conforms to `EnvironmentProtocol`. `TrainingEngine` cannot distinguish it from a task environment.

## Observations

While in meta (no active task env), `step_wait()` returns `SensorBundle` per brain:

| Sensor | Shape | Contents |
|--------|-------|----------|
| `env_field` | `(64,)` | Top-8 environments: curiosity, load, difficulty, distance, direction |
| `self_state` (proprioceptive) | `(4,)` | `[ce_ema, session_count, steps_in_meta, energy]` |

While proxying a task env, `step_wait()` returns that env's bundles unchanged and exposes that env's manifest.

## Cert signal handling

When a certification environment finishes evaluating a brain, it publishes a `CertSignal` into the Field. `MetaEnvironment` reads these on every `step_wait()` and calls `registry.record_cert()` if a `BrainRegistry` is wired in.

```
CertSignal.passed = 1.0  → brain earns the cert, prerequisite-gated envs unlock
CertSignal.passed = 0.0  → brain must complete retry_after_steps before retrying
```

Rate limiting is step-based (gradient steps earned), not wall-clock time. The brain earns the right to retry.

## Entry / exit policy

- `enter()` always succeeds — MetaEnvironment has unlimited capacity.
- On `reset()`, MetaEnvironment auto-enters the first registered task env if one is available.
- `navigate(brain_id, target_env_id, brain_obj)` triggers explicit env entry when the target's curiosity exceeds `enter_threshold`.
- When a task env signals `done=True`, MetaEnvironment exits it and returns the brain to meta.

## Relation to ecoframe and ecoframe-ecology

```
meta-env ─depends on─► ecoframe          (protocol, field, signals)
meta-env ─integrates─► ecoframe-ecology  (BrainRegistry for cert tracking)
```

`ecoframe-ecology` is optional — MetaEnvironment works without a registry; cert signals are simply ignored.
