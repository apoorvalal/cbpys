# `cbpys` : Covariate Balancing Propensity Scores in Python

Balancing scores for causal inference / covariate shift / domain adaptation.
Uses exponential tilting / implied regularized logistic pscore for exact balance.

## Installation:

__First__, install torch for your system (depending on whether you have a GPU or not) by following the instructions [here](https://pytorch.org/get-started/locally/).

Then, run
```
pip install git+https://github.com/apoorvalal/cbpys
```

or clone the repo and run
```
pip install -e .
```

(the latter is recommended since the code is still in development and you may want to pull updates)

## Examples

+ `examples/example.ipynb` for an example using Lalonde data.
+ `examples/ks.ipynb` for an example using the Kang/Schafer simulation dgp.



## References:

+ [Imai and Ratkovic](https://imai.fas.harvard.edu/research/files/CBPS.pdf)
+ [Zhao](https://www.statslab.cam.ac.uk/~qz280/publication/balancing-loss/)
+ [Wang and Zubizarreta](http://jrzubizarreta.com/minimal.pdf)
+ [Hainmueller](https://web.stanford.edu/~jhain/Paper/PA2012.pdf)

Reference R implementation is lightly edited version of published implementation [here](https://github.com/wxwx1993/wildfire_mitigation/blob/main/balancing/cbps_ATT.R) (courtesy Wager/Sverdrup)
