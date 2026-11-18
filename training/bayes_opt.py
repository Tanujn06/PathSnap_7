"""bayes_opt.py - Bayesian optimisation (BO-01).

Owner: Siddharth. Branch: feature/bayes-opt.
Optuna TPE with SQLite storage; objective = mean 5-fold GroupKFold macro F1
on train trips; at least 20 trials, 40 minute timeout.
"""
