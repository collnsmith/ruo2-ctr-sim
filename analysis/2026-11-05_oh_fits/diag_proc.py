import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))   # the repository; sys.modules["xraydb"] = None
import numpy as np
from ctrfit.project.project import Project
from ctrfit.fit.fit import Fit
from ctrfit.fit.global_fit import film_of
from ctrfit.fit.workflow import near_bragg

def main():
    p = Project.load(sys.argv[1])
    m, D = p.model, p.datasets
    f = Fit(m, D)
    print("per rod: slope of ln(F_data/F_model) on ln(F_model) (0: errors independent of intensity; < 0: the data's"
          " range is compressed), and the largest steps of F_data/F_model between neighbouring points")
    for ds, F in zip(D, f.model_F()):
        o = np.argsort(ds.L)
        L, Fd, Fm = ds.L[o], ds.F[o], F[o]
        keep = ~near_bragg(ds, 0.05, m)[o]
        y, x = np.log(Fd / Fm), np.log(Fm)
        slope = np.polyfit(x[keep], y[keep], 1)[0]
        r = np.exp(y)
        jump = np.abs(np.diff(y))                    # neighbouring ratio change (the model's own fringes cancel)
        big = np.argsort(jump)[-2:][::-1]
        js = ", ".join(f"{np.exp(y[i + 1] - y[i]):.2f}x at L {0.5 * (L[i] + L[i + 1]):.3f}" for i in big)
        print(f"  {film_of(ds).split('|')[0]:6s} {ds.meta.get('potential_V')!s:6s} ({ds.rod_labels[0]:4s}) slope {slope:+.2f}  "
              f"ratio range {np.percentile(r, 5):.2f}-{np.percentile(r, 95):.2f}  steps: {js}")

if __name__ == "__main__":
    main()
