"""In-situ electrochemistry: stationary SPEC scans (loopscan, or a phi scan that does not move)
recorded during a CV, lined up with the potentiostat file (EC-Lab .mpr / .mpt).

spec.py      SPEC files: scans, columns, point times on the local clock
ec.py        EC-Lab .mpr (galvani), .mpt and plain tables
sync.py      clock alignment (absolute, manual, events), potential at each point, sweep direction, cycles
analysis.py  relaxation background, cycle averaging, sigmoidal transitions, hysteresis
predict.py   CTR-model intensity at one HKL along a composition path vs potential; path fit; coverage
session.py   the whole chain in one object (used by the window and the command line)
"""
