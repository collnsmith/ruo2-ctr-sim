"""Measured structure factors |F|(H, K, L) with errors, grouped by rod.

sigma_eff = sqrt(sigma^2 + (sys_floor * F)^2): the systematic error floor (relative) is added in
quadrature to the counting errors, e.g. as estimated from symmetry-equivalent rods.
"""
import json

import numpy as np

META_KEYS = ("name", "energy_kev", "potential_V", "thickness_nm", "notes")


class Dataset:
    def __init__(self, H, K, L, F, sigma, rod=None, name="data", sys_floor=0.0, scope="", **meta):
        self.L = np.atleast_1d(np.asarray(L, float))
        n = self.L.size
        self.H = np.broadcast_to(np.asarray(H), (n,)).astype(int).copy()
        self.K = np.broadcast_to(np.asarray(K), (n,)).astype(int).copy()
        if np.any(np.asarray(H) != np.round(np.asarray(H))) or np.any(np.asarray(K) != np.round(np.asarray(K))):
            raise ValueError("H and K must be whole numbers (rod indices)")
        self.F = np.asarray(F, float).reshape(n).copy()
        self.sigma = np.asarray(sigma, float).reshape(n).copy()
        if np.any(self.sigma <= 0) or not np.all(np.isfinite(self.sigma)):
            raise ValueError("all sigma must be positive and finite")
        if not np.all(np.isfinite(self.F)):
            raise ValueError("F must be finite")
        self.rod = (np.array([f"{h} {k}" for h, k in zip(self.H, self.K)]) if rod is None
                    else np.asarray(rod, str).reshape(n).copy())
        self.name = name
        self.sys_floor = float(sys_floor)
        self.scope = scope
        self.meta = {k: v for k, v in meta.items() if v is not None}

    def __len__(self):
        return self.L.size

    def __repr__(self):
        return f"Dataset({self.name!r}, {len(self)} points, rods {', '.join(self.rod_labels)})"

    @property
    def sigma_eff(self):
        return np.sqrt(self.sigma ** 2 + (self.sys_floor * self.F) ** 2)

    @property
    def rod_labels(self):
        _, idx = np.unique(self.rod, return_index=True)
        return [str(self.rod[i]) for i in sorted(idx)]

    def rods(self):
        """{label: index array} in order of first appearance."""
        return {lab: np.flatnonzero(self.rod == lab) for lab in self.rod_labels}

    def rod_hk(self, label):
        i = np.flatnonzero(self.rod == label)[0]
        return int(self.H[i]), int(self.K[i])

    def subset(self, mask, name=None):
        mask = np.asarray(mask)
        return Dataset(self.H[mask], self.K[mask], self.L[mask], self.F[mask], self.sigma[mask], self.rod[mask],
                       name=name or self.name, sys_floor=self.sys_floor, scope=self.scope, **self.meta)

    def select_rods(self, rods, name=None):
        """Keep only these rods, given as labels '0 1' or (H, K) tuples."""
        want = {r if isinstance(r, str) else f"{r[0]} {r[1]}" for r in rods}
        mask = np.array([lab in want or f"{h} {k}" in want for lab, h, k in zip(self.rod, self.H, self.K)])
        return self.subset(mask, name)

    # ------------------------------------------------------------------ io
    def to_dict(self):
        return dict(name=self.name, sys_floor=self.sys_floor, scope=self.scope, meta=self.meta,
                    H=self.H.tolist(), K=self.K.tolist(), L=self.L.tolist(), F=self.F.tolist(),
                    sigma=self.sigma.tolist(), rod=self.rod.tolist())

    @classmethod
    def from_dict(cls, d):
        return cls(d["H"], d["K"], d["L"], d["F"], d["sigma"], d.get("rod"), name=d.get("name", "data"),
                   sys_floor=d.get("sys_floor", 0.0), scope=d.get("scope", ""), **d.get("meta", {}))

    def save(self, path):
        """Save as .json (everything) or .csv (points, with metadata in # comment lines)."""
        path = str(path)
        if path.lower().endswith(".csv"):
            from .io import write_csv
            write_csv(self, path)
            return
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=0)

    @classmethod
    def load(cls, path, **kw):
        """Load .json (from save), .csv or whitespace .dat files."""
        p = str(path).lower()
        if p.endswith(".json"):
            with open(path, encoding="utf-8") as fh:
                return cls.from_dict(json.load(fh))
        from .io import read_csv, read_dat
        return read_csv(path, **kw) if p.endswith(".csv") else read_dat(path, **kw)
