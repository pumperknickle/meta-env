"""
meta-env: the always-on environment every brain lives in.

MetaEnvironment is the outermost EnvironmentProtocol. The brain enters it once
and never exits. Task environments (MetaDrive, NAVSIM, etc.) are registered and
entered from within MetaEnvironment based on Field gradient signals.

Usage:
    from meta_env import MetaEnvironment, META_MANIFEST
    from ecoframe import Field, TrainingEngine

    field = Field(backend='local')
    meta  = MetaEnvironment(field=field)
    meta.register_env(task_env)    # MetaDriveEnvironment, NavsimEnvironment, etc.

    engine = TrainingEngine(brain, meta)
    for step, metrics in engine.run(n_steps=2_000_000):
        log(metrics)

The brain lives in MetaEnvironment:
  - step_async/step_wait proxy to the active task env
  - manifest switches to the task env's manifest transparently
  - auto-enters first registered env on reset()
  - Field receives EnvironmentSignal from each task env every 100 steps
"""
from meta_env.environment import MetaEnvironment, META_MANIFEST

__version__ = "0.1.0"
__all__ = ["MetaEnvironment", "META_MANIFEST"]
