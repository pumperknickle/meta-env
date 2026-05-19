"""
Checkpointer: save and restore brain state dicts.

Works standalone or wired into MetaEnvironment:

    # Standalone (external loop):
    ckpt = Checkpointer(path='/tmp/brains')
    ckpt.save('brain_0', brain.get_state(), step=1000)
    state = ckpt.load('brain_0')
    brain.set_state(state)

    # Inside MetaEnvironment (automatic):
    meta = MetaEnvironment(checkpointer=ckpt, checkpoint_interval=500)
    meta.enter('brain_0', brain_obj=brain)

Backends:
    'disk'  (default) — one file per checkpoint, human-inspectable
    'redis'           — keyed by brain_id, for distributed workers
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


class Checkpointer:
    """
    Saves and restores brain state dicts.

    Backend is swappable — same interface for disk and Redis.
    """

    def __init__(self, backend: str = 'disk', path: str = './checkpoints',
                 keep_last: int = 3, **kwargs):
        self._keep_last = keep_last
        if backend == 'disk':
            self._backend = _DiskBackend(path, keep_last)
        elif backend == 'redis':
            self._backend = _RedisBackend(keep_last=keep_last, **kwargs)
        else:
            raise ValueError(
                f"Unknown checkpoint backend: {backend!r}. "
                "Available: 'disk', 'redis'."
            )

    def save(self, brain_id: str, state: dict, step: int = 0) -> None:
        """Persist brain state. Prunes old checkpoints beyond keep_last."""
        self._backend.save(brain_id, state, step)

    def load(self, brain_id: str) -> dict | None:
        """Return the most recent checkpoint state, or None if none exists."""
        return self._backend.load(brain_id)

    def latest_step(self, brain_id: str) -> int:
        """Step number of the most recent checkpoint, or 0 if none exists."""
        return self._backend.latest_step(brain_id)

    def exists(self, brain_id: str) -> bool:
        return self._backend.load(brain_id) is not None


class _DiskBackend:
    def __init__(self, path: str, keep_last: int):
        self._root = Path(path)
        self._keep_last = keep_last

    def save(self, brain_id: str, state: dict, step: int) -> None:
        brain_dir = self._root / brain_id
        brain_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = brain_dir / f"step_{step:010d}.pkl"
        tmp_path  = ckpt_path.with_suffix('.tmp')
        with open(tmp_path, 'wb') as f:
            pickle.dump({'step': step, 'state': state}, f)
        tmp_path.rename(ckpt_path)
        self._prune(brain_dir)

    def load(self, brain_id: str) -> dict | None:
        brain_dir = self._root / brain_id
        if not brain_dir.exists():
            return None
        ckpts = sorted(brain_dir.glob('step_*.pkl'))
        if not ckpts:
            return None
        with open(ckpts[-1], 'rb') as f:
            return pickle.load(f)['state']

    def latest_step(self, brain_id: str) -> int:
        brain_dir = self._root / brain_id
        if not brain_dir.exists():
            return 0
        ckpts = sorted(brain_dir.glob('step_*.pkl'))
        if not ckpts:
            return 0
        # filename: step_0000001000.pkl → strip prefix/suffix
        try:
            return int(ckpts[-1].stem.split('_')[1])
        except (IndexError, ValueError):
            return 0

    def _prune(self, brain_dir: Path) -> None:
        ckpts = sorted(brain_dir.glob('step_*.pkl'))
        for old in ckpts[:-self._keep_last]:
            old.unlink(missing_ok=True)


class _RedisBackend:
    _KEY = "ecoframe:ckpt:{brain_id}:state"
    _STEP_KEY = "ecoframe:ckpt:{brain_id}:step"

    def __init__(self, url: str = 'redis://localhost:6379',
                 keep_last: int = 3, **kwargs):
        try:
            import redis
        except ImportError:
            raise ImportError("redis required: pip install meta-env[redis]")
        self._r = redis.from_url(url, **kwargs)

    def save(self, brain_id: str, state: dict, step: int) -> None:
        key      = self._KEY.format(brain_id=brain_id)
        step_key = self._STEP_KEY.format(brain_id=brain_id)
        self._r.set(key, pickle.dumps(state))
        self._r.set(step_key, str(step))

    def load(self, brain_id: str) -> dict | None:
        raw = self._r.get(self._KEY.format(brain_id=brain_id))
        return pickle.loads(raw) if raw else None

    def latest_step(self, brain_id: str) -> int:
        raw = self._r.get(self._STEP_KEY.format(brain_id=brain_id))
        return int(raw) if raw else 0
