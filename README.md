# Forecasting
Repository containing time-series analysis methods, such as HiPPO and neural Koopman operators.

1. The lkis.py file contains an implementation of [Learning Koopman Invariant Subspaces for Dynamic Mode Decomposition](https://arxiv.org/pdf/1710.04340.pdf), a neural network approach for learning the measurement operators in Koopman theory / Dynamic Mode Decomposition.
2. The koopman_neural_forcaster.py file contains an implementation of [Koopman Neural Forecaster for Time Series with Temporal Distribution Shifts](https://arxiv.org/pdf/2210.03675).
3. The hippo.py file contains an implementation of [HiPPO-LegS](https://arxiv.org/abs/2008.07669). Note that the official version has better speed and stability, so you should [compile the official C++ kernel from here](https://github.com/HazyResearch/hippo-code)
