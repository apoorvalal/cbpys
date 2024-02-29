"""Examples"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from cbps_scipy import cbps_att
from src.cbps_torch import cbps_retarget, cbps_torch

## From the scipy implementation


df = pd.read_csv("../df/lalonde_psid.csv")
df.head()

w, y = df.treat.values, df.re78.values
X = df.drop(columns=["treat", "re78"]).values

cbps_wt = cbps_att(X, w, method="Powell")
y[w == 1].mean() - np.average(y[w == 0], weights=cbps_wt["weights_0"][w == 0])
y[w == 1].mean() - y[w == 0].mean()

## From the pytorch implementation

# testing
df = pd.read_csv("../df/lalonde_psid.csv")
w, y = df.treat.values, df.re78.values
# naive estimate is garbage - true effect is ~ 1800 (1794)
y[w == 1].mean() - y[w == 0].mean()
# -15204.775555988717
# stabilize by scaling covariates
X = df.drop(columns=["treat", "re78"]).values
X = MinMaxScaler().fit_transform(X)
#  cbps estimate of ATT - comes pretty close to true effect
cbps_wt = cbps_torch(
    X,
    w,
    niter=10_000,
    lr=0.001,
    noi=False,
)
y[w == 1].mean() - np.average(y[w == 0], weights=cbps_wt)
# 1671.9811205765518
# test aggregated data - same loss as above
X0, X1 = X[w == 0], X[w == 1].mean(axis=0)
wgt = cbps_retarget(
    X0,
    X1,
    niter=10_000,
    lr=0.001,
)
y[w == 1].mean() - np.average(y[w == 0], weights=wgt)
# 1951.5708059253102
