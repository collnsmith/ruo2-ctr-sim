"""Physics. Everything that computes a structure factor lives here."""
from .ctrmodel import CTRModel, parse_inputs, relaxation_study  # noqa: F401
from .lattice import film_spacing, trilayers  # noqa: F401
from .structures import ADSORBATE_H, oh_h2o  # noqa: F401
