# CBPS via empirical loss minimization

Solves ATT analogue of the covariate-balancing propensity scores problem 
![image](https://github.com/apoorvalal/covariate_balancing_propensity_scores/assets/12086926/7076542a-d177-4dcf-83c8-5b5e95772d8d)

R implementation lightly edited version of implementation [here](https://github.com/wxwx1993/wildfire_mitigation/blob/main/balancing/cbps_ATT.R) (courtesy Wager/Svedrup)

Uses exponential tilting for exact balance. Approximate balance tuning forthcoming.  

Refs:

+ [Imai and Ratkovic](https://imai.fas.harvard.edu/research/files/CBPS.pdf)
+ [Zhao](https://www.statslab.cam.ac.uk/~qz280/publication/balancing-loss/)
+ [Wang and Zubizarreta](http://jrzubizarreta.com/minimal.pdf)
+ [Hainmueller](https://web.stanford.edu/~jhain/Paper/PA2012.pdf)
