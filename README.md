# MicroBooNE decay study 2024

This code is an extension of the one used in [2111.10359](https://arxiv.org/abs/2111.10359) to include decaying-sterile neutrinos. It also inherits some functionality from the antineutrino analysis in [2301.12573](https://arxiv.org/abs/2301.12573).

If you are using this, please cite: 

[2406.04401](https://arxiv.org/abs/2406.04401)

# Short-baseline sterile neutrino exclusion study 2025

This code is also used in [2503.13594](https://arxiv.org/abs/2503.13594).

For verification purposes, figure 1,2,3,5 are from MH_jointfit.ipynb. Figure 6 is generated in event_rate_plots.ipynb. Figure 4 is from KJK_MutualCompatibility.ipynb. Figure 7 is from MH_oscillation_scans.ipynb.
---

This repository contains data files provided by the MicroBooNE collaboration in their data releases:
* inclusive (Wire-Cell) [data release](https://www.hepdata.net/record/ins1953539) (including v3)
* CCQE (Deep Learning) [data release](https://www.hepdata.net/record/ins1953568)

Also used are the files in the [2018 MiniBooNE data release](https://arxiv.org/abs/2110.15055?context=nucl-ex#:~:text=The%20MiniBooNE%20experiment%20has%20provided,of%20the%20MiniBooNE%20data%20releases) as well as the 2020 data provided by Austin Schneider.

PROSPECT chi square and sensitivity maps in ProspectTools are provided by Bryce Littlejohn and Ohana Benevides Rodrigues. The datafiles are also publicly accessible in [1806.02784](https://arxiv.org/abs/1806.02784) and [2006.11210](https://arxiv.org/abs/2006.11210).

All other data files are obtained by digitizing plots available in the corresponding papers.

---

The MINOS files come from data release and are run using ROOT.


# MH explanations

Each experiment has its own folder (BestTools, MicroTools, InclusiveTools...).
Magic happens in param_scan.py
You can turn off decays so it is only oscillations.
MiniBooNE is inside MicroTools, functions return both chi2 (check DecayReturnMicroBooNEChi2).

This is how it works. ``theta'' are physical parameters
1. Computes the rates (dict has several flags for how to do the fit)
2. nue_app_for_unfolding is used to get the template which then is passed to uB (miniTomicro would become miniToSBND ...)
3. We need to re-write NuMuReps, NueReps, etc.

MH_decay_scans.ipynb has working examples
