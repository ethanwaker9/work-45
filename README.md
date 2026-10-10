# Deterministic Reductions to Unique-SVP for All Gaps

This repository contains our implementation of experiments of our research
*Derandomizing Unique-SVP Lattice Reductions Without Gap Restrictions*, which consists of:

* **TSE** (tube-calibrated sparsified embedding), the reduction from bounded distance
  decoding (BDD) to the unique shortest vector problem (uSVP) of our research, together with
  * `ATSE`, the interleaved adaptive variant of TSE,
  * `TSEF`, TSE with the dimension-free prime of our work,
* the competing reductions from BDD to uSVP:
  * `KAN`: Kannan's embedding with the scaling `M = d/sqrt(n)` used by the primal attack,
  * `LM09`: the embedding of Lyubashevsky and Micciancio (CRYPTO 2009) with `M = d`,
  * `FOLK`: the folklore embedding analyzed in the appendix of Bai, Stehle and Wen (ICALP 2016),
  * `BSW16`: the sparsified embedding of Bai, Stehle and Wen (ICALP 2016),
  * `ABSW16`: the adaptive schedule of ATSE applied to the scaling and prime of BSW16,
* the deterministic reductions from BDD to uSVP with the pencil family
  (`bdd_deterministic`) and with shifts (`bdd_shift_deterministic`),
* the deterministic reductions from ZSVP (unit vectors in rotations of Z^n) to uSVP with
  the pencil family and the thin pencil family, the randomized counterpart with the same
  prime, and the sampler of Bennett, Ganju, Peetathawatchai and Stephens-Davidowitz
  (EUROCRYPT 2023) with its own prime,
* the exact unwanted set of an embedding and the exact success probability of one
  sparsified trial (`unwanted_exact`, `residue_lines`, `avoid_probability`),
* the empirical count of lattice points in tubes (the tube lemma of our work),
* a promise oracle for uSVP that decides the promise `lambda_2 >= gamma * lambda_1`
  exactly by enumeration and returns a failure symbol when the promise does not hold.

## Files and Contents

```
tubesparse/
  lattice.py      exact integer vector arithmetic, LLL and enumeration through fpylll
  oracle.py       PromiseUSVPOracle: exact uSVP_gamma promise check, failure on violation;
                  HelpfulUSVPOracle: returns a shortest vector without checking the promise
  isolation.py    primes, tube counts N(n, eps) and N'(eps), sparsified bases,
                  pencil family and thin pencil family
  reductions.py   KAN, LM09, FOLK, BSW16, TSE, TSEF, ATSE, ABSW16, deterministic BDD
                  reductions, deterministic and randomized ZSVP reductions
  instances.py    random q-ary lattices, adversarial and random BDD targets, bases of Z^n,
                  simplex lattices and the lower bound instances of the appendix
  tube.py         lattice points near a segment, tube lemma bound, exact unwanted sets
experiments/
  common.py       paths, gap values, regions of the decoding radius
  exp_tube.py     lattice points in tubes
  exp_zsvp.py     deterministic ZSVP reduction with the pencil family
  exp_thin.py     thin pencil family against the pencil family and random sparsification
  exp_bgps.py     sampler of Bennett et al. for ZSVP with its prime
  exp_bdd.py      main comparison of the five BDD reductions
  exp_sweep.py    success and calls as a function of the decoding radius
  exp_gamma.py    oracle calls as a function of the gap
  exp_det.py      deterministic BDD reduction with pencils on the hard instances
  exp_dets.py     deterministic BDD reduction with shifts on the hard instances
  exp_atse.py     adaptive variants on the main and gap instances, and n = 32, 36, 40
  exp_fold.py     one call over a grid of radii and scalings (optimality of the folklore radius)
  exp_calib.py    success per trial against the exact prediction, and repeated runs of TSE
  exp_mult.py     exact unwanted sets as a function of the scaling
  exp_simplex.py  tightness of the tube lemma and the lower bound instances
  exp_extra.py    slack ablation, oracle without the promise, other lattice families
  exp_uniq.py     gap lambda_2/lambda_1 of the oracle inputs on the hard instances
  exp_beyond.py   hard instances beyond the proven radius
  make_tables.py  results/summary.json with the main numbers
results/          JSON lines of all runs, logs and summary.json
```


## Running the Experiments

The requirements are Python 3.12, fplll 5.5.0, fpylll 0.6.4, numpy 2.3.1, matplotlib 3.10.8, and epstopdf.

`fpylll` needs the fplll library; on macOS it is available through Homebrew
(`brew install fplll`), and on Linux through the package manager or conda forge.

```bash
./run_all.sh
```
runs all experiments with all but one core, writes every run to `results/*.jsonl`, and
produces the figures in `figures/` and `results/summary.json`. The drivers of the long
experiments skip runs that are already present in their result files, so an interrupted
run can be resumed by starting it again. All random choices are derived from fixed seeds,
so a second run reproduces the same instances, coins and numbers of oracle calls, while
the measured times depend on the machine.

On M2 with 8 cores, the main comparison takes about 70
minutes, the gap experiment about three hours, the two deterministic BDD experiments about
30 minutes each, the radius sweep with 70 radii and ten lattices per radius, the adaptive
variants in dimensions 32 to 40 and the sampler of Bennett et al. about an hour each, and
every other driver a few minutes. 

Individual experiments can be run from `experiments/`, for example
```bash
cd experiments
python3 exp_bdd.py --dims 12,16 --gammas 1.5,2 --instances 3 --workers 4
python3 exp_atse.py --parts bdd --workers 4
python3 make_plots.py
python3 make_plots.py --llncs
```

## The Oracle Model

A reduction to the promise problem uSVP_gamma may only rely on answers to instances
that satisfy the promise. `PromiseUSVPOracle` LLL-reduces its input, enumerates the
`2*ceil(gamma)+3` shortest vectors within radius `gamma * ||b_1||` with fpylll, and
checks with exact integer arithmetic whether a vector that is not parallel to the
shortest one is shorter than `gamma` times the minimum. If the promise holds it returns
a shortest vector; otherwise it returns `None`, which models the worst case allowed by
the definition of a promise problem. All reductions use the same oracle, receive the
same instances and the same distance `d` of the target, and stop at the first candidate
at distance `d`. The number of oracle calls, the wall-clock time, the peak memory of the
Python process measured with `tracemalloc`, and the largest bit length of an entry of an
oracle input are recorded for every run. `HelpfulUSVPOracle` is used only to show what
an oracle that ignores the promise would provide.

## Instances

`random_lattice(n, 10, seed)` generates a q-ary lattice with fpylll and LLL-reduces it.
`bdd_instance` scales the lattice by `2^12`, picks a random lattice point `c`, and places
the target at distance `alpha * lambda_1` from `c`, either in a uniformly random direction
(`rand`) or along a shortest vector (`adv`), which realizes the worst case of the
single call embeddings because multiples of the offset then approach lattice points.
The regions of `alpha` are R1 (below the radius of LM09), R2 (between LM09 and the
folklore embedding, only for non-integer gaps), and R3a, R3b (between the folklore
radius and the radius `0.9/(sqrt(2)*gamma)` that TSE with `eps = 0.1` guarantees).
`zn_instance` produces random bases of Z^n, `simplex_rows` the lattices L_{n,b} of the
appendix, and `simplex_bdd_instance` the instances of the lower bound for sparsification.
