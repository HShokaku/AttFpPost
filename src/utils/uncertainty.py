import numpy as np
import torch
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import norm

def ranking_evaluation(y_true,
                       y_pred,
                       var_estimated):

    y_true, y_pred, var_estimated = np.array(y_true), np.array(y_pred), np.array(var_estimated)
    if (y_pred.shape == y_true.shape == var_estimated.shape) is False:
        raise TypeError("The data you entered is wrong.")

    data = np.vstack((y_true, y_pred, var_estimated)).T
    data = data[data[:, 2].argsort()[::-1]]
    y_ordered = data[:, 0:2]

    MAE        =  np.array([mae(y_ordered[i:, 0], y_ordered[i:, 1]) for i in range(len(y_ordered))])
    ERROR      =  np.sort(np.abs(y_ordered[:, 0] - y_ordered[:, 1]))[::-1]
    MAE_ORACLE =  np.array([np.nanmean(ERROR[i:]) for i in range(len(ERROR))])

    # compute AUCO
    AUC_OF_MAE = MAE.sum() / len(MAE)
    AUC_OF_ORACLE = MAE_ORACLE.sum() / len(MAE)
    AUCO = AUC_OF_MAE - AUC_OF_ORACLE

    # compute Error Drop
    ED = MAE[0] / MAE[-1]

    # compute Decreasing Coefficient
    count = 0
    for i in range(len(MAE) - 1):
        if MAE[i + 1] >= MAE[i]:
            count += 1
        else:
            continue
    DC = count / (len(MAE) - 1)

    return {'AUCO': AUCO,
            'ED': ED,
            'DC': DC,
            'MAE': MAE,
            'MAE_ORACLE': MAE_ORACLE}

def cbc_evaluation(y_true,
                   y_pred,
                   var_estimated):

    y_true, y_pred, var_estimated = np.array(y_true), np.array(y_pred), np.array(var_estimated)
    if (y_pred.shape == y_true.shape == var_estimated.shape) is False:
        raise TypeError("The data you entered is wrong.")

    std_mean = var_estimated ** 0.5
    data = np.vstack((y_true, y_pred, std_mean)).T

    PERCENT = []
    for p in range(100):
        p = p + 1
        count = 0
        for i in range(len(data)):
            upper = norm.ppf((0.5 + p / 200), data[i, 1], data[i, 2])
            if (2 * data[i, 1] - upper) <= data[i, 0] <= upper:
                count += 1
            else:
                continue
        percent = count / len(data)
        PERCENT.append(percent)

    PERCENT = np.array(PERCENT)
    P = np.linspace(0.01, 1.00, 100)

    AUCE = np.sum(abs(PERCENT - P))
    ECE = np.nanmean(abs(PERCENT - P))
    MCE = np.max(abs(PERCENT - P))

    return {'AUCE': AUCE,
            'ECE': ECE,
            'MCE': MCE,
            'P'  : P,
            'PERCENT': PERCENT}

def ebc_evaluation(y_true,
                   y_pred,
                   var_estimated):

    y_true, y_pred, var_estimated = np.array(y_true), np.array(y_pred), np.array(var_estimated)
    if (y_pred.shape == y_true.shape == var_estimated.shape) is False:
        print("The data you entered is wrong.")

    data = np.vstack((y_true, y_pred, var_estimated)).T
    data = data[data[:, 2].argsort()]  # Sort the data according to the var_estimated
    # data = data[::-1]
    # var_estimated_ordered = data[:,2].T.squeeze()
    K = len(data) // 20  # the number of bins
    bins = np.array_split(data, K, axis=0)

    RMSE = []
    ERROR = []
    for i in range(len(bins)):
        rmse = np.sqrt(mse(bins[i][:, 0], bins[i][:, 1]))
        error = np.sqrt(np.nanmean(bins[i][:, 2]))
        RMSE.append(rmse)
        ERROR.append(error)
    RMSE = np.array(RMSE)
    ERROR = np.array(ERROR)

    # compute expected normalized calibration error
    ENCE = np.nanmean((abs(RMSE - ERROR)) / ERROR)
    return {'ENCE': ENCE,
            'RMSE': RMSE,
            'ERROR': ERROR}

def diff_entropy(alpha):
    eps = 1e-6
    alpha = alpha + eps
    alpha0 = alpha.sum(-1)

    log_term = torch.sum(torch.lgamma(alpha), axis=1) - torch.lgamma(alpha0)
    digamma_term = torch.sum((alpha - 1.0) * ( torch.digamma(alpha) - torch.digamma((alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha))), axis=1)
    differential_entropy = log_term - digamma_term
    return differential_entropy

