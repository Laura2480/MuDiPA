"""Pluggable discourse-engine registry for MuDiPA.

An *engine* bundles the model-specific ways to
  (a) ``activate``      -- warm up / check readiness (load a GPU adapter, verify an
                           API key, ...);
  (b) ``score_links``   -- score candidate parents p<t for a target EDU;
  (c) ``score_rel``     -- score SDRT relations for an arc source->target;
  (d) ``explain``       -- produce a C1-shaped link+relation+contrastive explanation.

Endpoints resolve an engine *by name* from :data:`REGISTRY` and delegate, so adding
a new engine means registering one object -- not editing every ``/api/suggest/*``
route. Concrete engines live in ``app.py`` (they wrap the existing helpers); this
module stays dependency-free so it can be imported anywhere without pulling in Flask.

Capabilities are advertised via the ``can_*`` flags. A method whose flag is False
may raise :class:`NotImplementedError`; callers should check the flag (or use
:meth:`Registry.for_capability`) before dispatching, and treat a ``None`` result
from :meth:`Engine.explain` as "unavailable -- fall back to another engine".
"""
from __future__ import annotations


class Engine:
    """Base class for a pluggable engine. Subclass, set the capability flags for
    what you implement, override the matching methods, then ``register()`` it."""

    name = "base"
    label = "Base engine"

    # -- capabilities: default off; a subclass turns on only what it implements --
    can_score_links = False
    can_score_rel = False
    can_score_pair = False
    can_explain = False
    needs_activation = False

    # None => supports every dataset; otherwise a collection of dataset ids.
    datasets = None

    def supports_dataset(self, dataset) -> bool:
        return self.datasets is None or dataset in self.datasets

    # -- operations (override the ones your capability flags advertise) ---------

    def activate(self, dataset=None) -> dict:
        """Warm up / check readiness. Returns a dict with at least ``steps`` (a
        list of human-readable status lines) and ``ready`` (bool)."""
        return {"steps": [], "ready": True}

    def score_links(self, edus, speakers, target, dataset):
        """Return raw link candidates ``[{"source": int, "score": float}, ...]``.
        SDRT post-filters (right-frontier, etc.) are the caller's job, not the
        engine's."""
        raise NotImplementedError(f"{self.name} cannot score links")

    def score_rel(self, edus, speakers, source, target, dataset):
        """Return raw relation scores ``[{"relation": str, "score": float}, ...]``.
        Structural rule filtering is applied by the caller."""
        raise NotImplementedError(f"{self.name} cannot score relations")

    def score_pair(self, edus, speakers, source, target, dataset, target_text=None):
        """Return a scalar attach score in [0, 1] that ``target`` links to ``source``,
        or ``None`` if unavailable. ``target_text`` overrides ``edus[target]`` (used
        for occlusion saliency). This is the *faithful* signal behind /api/explain/arc:
        re-scoring with a word removed shows how much that word drives the link."""
        raise NotImplementedError(f"{self.name} cannot score a pair")

    def explain(self, edus, speakers, source, target, dataset, **ctx):
        """Return a C1-shaped candidate dict, or ``None`` if the engine is
        unavailable (the caller may then fall back to another engine)."""
        raise NotImplementedError(f"{self.name} cannot explain")

    def info(self) -> dict:
        """Machine-readable descriptor (for the /api/engines discovery route)."""
        return {
            "name": self.name,
            "label": self.label,
            "can_score_links": self.can_score_links,
            "can_score_rel": self.can_score_rel,
            "can_score_pair": self.can_score_pair,
            "can_explain": self.can_explain,
            "needs_activation": self.needs_activation,
            "datasets": None if self.datasets is None else sorted(self.datasets),
        }


class Registry:
    """Ordered name->Engine map with a designated default."""

    def __init__(self):
        self._engines = {}
        self._default = None

    def register(self, engine: Engine, default: bool = False) -> Engine:
        self._engines[engine.name] = engine
        if default or self._default is None:
            self._default = engine.name
        return engine

    def get(self, name, fallback: bool = False):
        """Look up an engine by name. With ``fallback=True``, an unknown name
        returns the default engine instead of ``None``."""
        if name and name in self._engines:
            return self._engines[name]
        if fallback and self._default:
            return self._engines[self._default]
        return None

    @property
    def default(self):
        return self._default

    def names(self):
        return list(self._engines)

    def all(self):
        return list(self._engines.values())

    def for_capability(self, capability: str, dataset=None):
        """Engines advertising ``capability`` (e.g. ``"can_explain"``) that also
        support ``dataset`` (when given)."""
        return [
            e for e in self._engines.values()
            if getattr(e, capability, False)
            and (dataset is None or e.supports_dataset(dataset))
        ]


# Process-wide singleton. app.py registers its engines into this at import time.
REGISTRY = Registry()


def register(engine: Engine, default: bool = False) -> Engine:
    return REGISTRY.register(engine, default=default)


def get_engine(name, fallback: bool = False):
    return REGISTRY.get(name, fallback=fallback)
