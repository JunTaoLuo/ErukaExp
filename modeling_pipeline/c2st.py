# Source: https://gist.github.com/oddskool/409018f61d432f10fe00223e2b93cb51

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss

def c2st(X, y, clf=LogisticRegression(max_iter=1000), loss=hamming_loss, bootstraps=300):
    """
    Perform Classifier Two Sample Test (C2ST) [1].

    This test estimates if a target is predictable from features by comparing the loss of a classifier learning
    the true target with the distribution of losses of classifiers learning a random target with the same average.

    The null hypothesis is that the target is independent of the features - therefore the loss a classifier learning
    to predict the target should not be different from the one of a classifier learning independent, random noise.

    Input:
        - `X` : (n,m) matrix of features
        - `y` : (n,) vector of target - for now only supports binary target
        - `clf` : instance of sklearn compatible classifier (default: `LogisticRegression`)
        - `loss` : sklearn compatible loss function (default: `hamming_loss`)
        - `bootstraps` : number of resamples for generating the loss scores under the null hypothesis

    Return: (
        loss value of classifier predicting `y`,
        loss values of bootstraped random targets,
        p-value of the test
    )

    Usage:
    >>> emp_loss, random_losses, pvalue = c2st(X, y)

    Plotting H0 and target loss:
    >>>bins, _, __ = plt.hist(random_losses)
    >>>med = np.median(random_losses)
    >>>plt.plot((med,med),(0, max(bins)), 'b')
    >>>plt.plot((emp_loss,emp_loss),(0, max(bins)), 'r--')

    [1] Lopez-Paz, D., & Oquab, M. (2016). Revisiting classifier two-sample tests. arXiv preprint arXiv:1610.06545.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    emp_loss = loss(y_test, y_pred)
    bs_losses = []
    y_bar = np.mean(y)
    for b in range(bootstraps+1):
        y_random = np.random.binomial(1, y_bar, size=y.shape[0])
        X_train, X_test, y_train, y_test = train_test_split(X, y_random)
        y_pred_bs = clf.fit(X_train, y_train).predict(X_test)
        bs_losses += [loss(y_test, y_pred_bs)]
    pc = stats.percentileofscore(sorted(bs_losses), emp_loss) / 100.
    pvalue = pc if pc < y_bar else 1 - pc
    return emp_loss, np.array(bs_losses), pvalue