"""A project file: model, datasets, fit results and notes in one JSON file, so a fit can be
reproduced exactly. It also keeps the loaded files and the choice per scan (fitted or left out, its
group; see fit.scans): `datasets` are the fitted datasets built from them.

    proj = Project(model, [data], name="RuO2 6 nm, 1.0 V")
    proj.results.append(res.to_dict())
    proj.save("fit.ctrproj.json")
    proj = Project.load("fit.ctrproj.json")
"""
import json
from datetime import datetime, timezone

from .. import __version__
from ..data.dataset import Dataset
from ..fit.scans import ScanSelection
from ..model.model import Model

FORMAT = "ctrfit-project"
VERSION = 1


class Project:
    def __init__(self, model=None, datasets=None, results=None, name="untitled", notes="", scans=None,
                 options=None):
        self.model = model if model is not None else Model.from_settings()
        self.datasets = list(datasets or [])
        self.results = list(results or [])
        self.name, self.notes = name, notes
        self.options = dict(options or {})      # fit choices kept with the project, e.g. skip_bragg (± L)
        self.scans = scans if scans is not None else ScanSelection.from_datasets(self.datasets)

    def to_dict(self):
        return dict(format=FORMAT, version=VERSION, ctrfit=__version__, name=self.name, notes=self.notes,
                    saved=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    model=self.model.to_dict(), datasets=[ds.to_dict() for ds in self.datasets],
                    scans=self.scans.to_dict(), results=self.results, options=self.options)

    @classmethod
    def from_dict(cls, d):
        if d.get("format") != FORMAT:
            raise ValueError("not a ctrfit project file")
        if d.get("version", 0) > VERSION:
            raise ValueError(f"project file version {d['version']} is newer than this ctrfit")
        datasets = [Dataset.from_dict(x) for x in d.get("datasets", [])]
        scans = None
        if "scans" in d:                  # the fitted datasets again from the files and choices (same scopes)
            scans = ScanSelection.from_dict(d["scans"])
            datasets, _ = scans.build(previous=datasets)
        return cls(Model.from_dict(d["model"]), datasets, d.get("results", []), d.get("name", "untitled"),
                   d.get("notes", ""), scans=scans, options=d.get("options"))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1, default=float)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
