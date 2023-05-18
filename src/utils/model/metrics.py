import math
import torch
from typing import Any, Callable, List, Tuple, Union
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score, roc_auc_score, accuracy_score, log_loss, matthews_corrcoef, f1_score, brier_score_loss, recall_score, precision_score
import numpy as np
import torch.nn as nn
import torch
from torch.distributions.dirichlet import Dirichlet

def UCE_loss(targets, alpha):
    targets = torch.tensor(targets, dtype=torch.long).reshape(-1, 1)
    alpha   = torch.tensor(alpha, dtype=torch.float64)
    alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, 2)
    entropy_reg = Dirichlet(alpha).entropy()

    targets_hot = torch.zeros(targets.shape[0], 2)
    targets_hot.scatter_(1, targets, 1)

    UCE_loss = torch.sum(targets * (torch.digamma(alpha_0) - torch.digamma(alpha))) - 1e-5 * torch.sum(entropy_reg)

    return UCE_loss.detach().cpu().item()


def EF1(y_test, y_pred, threshold=0.01):
    '''calculte the enrichment factors'''
    y_test = np.array(y_test).squeeze()
    y_pred = np.array(y_pred).squeeze()
    sorted_inds = np.argsort(y_pred, kind="mergesort")[::-1]
    y_test = y_test[sorted_inds]
    y_pred = y_pred[sorted_inds]

    N_total = y_test.shape[0]
    N_actives = len(np.where(y_test == 1)[0])

    N_topx_total = int(np.ceil(N_total * threshold))
    topx_total = y_test[:N_topx_total]
    N_topx_actives = len(np.where(topx_total == 1)[0])

    return (N_topx_actives / N_topx_total) / (N_actives / N_total)

def bedroc_score(y_test, y_pred, alpha=80.5):
    y_test = np.array(y_test).squeeze()
    y_pred = np.array(y_pred).squeeze()
    '''calculte the BEDROC'''
    assert len(y_test) == len(y_pred), \
        'The number of scores must be equal to the number of labels'
    big_n = len(y_test)
    n = sum(y_test == 1)
    order = np.argsort(-y_pred)
    m_rank = (y_test[order] == 1).nonzero()[0]
    s = np.sum(np.exp(-alpha * m_rank / big_n))
    r_a = n / big_n
    rand_sum = r_a * (1 - np.exp(-alpha)) / (np.exp(alpha / big_n) - 1)
    fac = r_a * np.sinh(alpha / 2) / (np.cosh(alpha / 2) -
                                      np.cosh(alpha / 2 - alpha * r_a))
    cte = 1 / (1 - np.exp(alpha * (1 - r_a)))
    return s * fac / rand_sum + cte

def prc_auc(targets: List[int],
            preds: List[float]) -> float:

    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def rmse(targets: List[float],
         preds: List[float]) -> float:

    return math.sqrt(mean_squared_error(targets, preds))


def mse(targets: List[float],
        preds: List[float]) -> float:

    return mean_squared_error(targets, preds)


def accuracy(targets: List[int],
             preds: Union[List[float], List[List[float]]],
             threshold: float = 0.5) -> float:
    """
    Computes the accuracy of a binary prediction task using a given threshold for generating hard predictions.

    Alternatively, computes accuracy for a multiclass prediction task by picking the largest probability.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0.
    :return: The computed accuracy.
    """
    if type(preds[0]) == list:  # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds]  # binary prediction

    return accuracy_score(targets, hard_preds)

def MCC(targets: List[int],
        preds: List[float],
        threshold = 0.5) -> float:

    if 0 in targets:
        targets = [-1 if l == 0 else 1 for l in targets]
    hard_preds = [1 if p > threshold else -1 for p in preds]
    return matthews_corrcoef(targets, hard_preds)

def F1(targets: List[int],
       preds: Union[List[float], List[List[float]]],
       threshold: float = 0.5) -> float:

    if type(preds[0]) == list:  # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds]  # binary prediction

    return f1_score(targets, hard_preds)


def bce(targets: List[int],
        preds: List[float]) -> float:
    """
    Computes the binary cross entropy loss.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed binary cross entropy.
    """
    # Don't use logits because the sigmoid is added in all places except training itself
    bce_func = nn.BCELoss(reduction='mean')
    loss = bce_func(target=torch.Tensor(targets), input=torch.Tensor(preds)).item()

    return loss


def get_metric_func(metric: str) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    r"""
    Gets the metric function corresponding to a given metric name.

    Supports:

    * :code:`auc`: Area under the receiver operating characteristic curve
    * :code:`prc-auc`: Area under the precision recall curve
    * :code:`rmse`: Root mean squared error
    * :code:`mse`: Mean squared error
    * :code:`mae`: Mean absolute error
    * :code:`r2`: Coefficient of determination R\ :superscript:`2`
    * :code:`accuracy`: Accuracy (using a threshold to binarize predictions)
    * :code:`cross_entropy`: Cross entropy
    * :code:`binary_cross_entropy`: Binary cross entropy

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """
    if metric == 'roc-auc':
        return roc_auc_score

    if metric == 'prc-auc':
        return prc_auc

    if metric == 'rmse':
        return rmse

    if metric == 'mse':
        return mse

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score

    if metric == 'accuracy':
        return accuracy

    if metric == 'cross_entropy':
        return log_loss

    if metric == 'binary_cross_entropy':
        return bce
    
    if metric == 'EF1':
        return EF1

    if metric == 'UCE_loss':
        return UCE_loss

    if metric == "F1":
        return F1

    if metric == "MCC":
        return MCC

    if metric == "BEDROC":
        return bedroc_score

    if metric == "UCE":
        return UCE_loss

    raise ValueError(f'Metric "{metric}" not supported.')


def Gaussian_NLL(targets: np.ndarray,
                 preds  : np.ndarray,
                 unc    : np.ndarray,
                 reduce=True):
    if np.sum(unc < 0) > 0:
        return None
    unc[unc==0] = 1e-7
    log_likelihood = 0.5 * np.log(unc) + \
                     0.5 * np.log(2*np.pi) + \
                     0.5 * ((preds-targets)**2/unc)
    return np.mean(log_likelihood) if reduce else log_likelihood


def bin_predictions_and_accuracies(probabilities, ground_truth, bins=10):
  counts, bin_edges = np.histogram(probabilities, bins=bins, range=[0., 1.])
  indices = np.digitize(probabilities, bin_edges, right=True)
  accuracies = np.array([np.mean(ground_truth[indices == i])
                         for i in range(1, bins + 1)])
  return bin_edges, accuracies, counts

def bin_centers_of_mass(probabilities, bin_edges):
  probabilities = np.where(probabilities == 0, 1e-8, probabilities)
  indices = np.digitize(probabilities, bin_edges, right=True)
  return np.array([np.mean(probabilities[indices == i])
                   for i in range(1, len(bin_edges))])

def expected_calibration_error(ground_truth, probabilities, bins=15):

  probabilities = probabilities.flatten()
  ground_truth = ground_truth.flatten()
  bin_edges, accuracies, counts = bin_predictions_and_accuracies(
      probabilities, ground_truth, bins)
  bin_centers = bin_centers_of_mass(probabilities, bin_edges)
  num_examples = np.sum(counts)

  ece = np.sum([(counts[i] / float(num_examples)) * np.sum(
      np.abs(bin_centers[i] - accuracies[i]))
                for i in range(bin_centers.size) if counts[i] > 0])
  return ece

def confusion_matrix(predictions, labels, threshold=0.5):
  tp = 0
  tn = 0
  fp = 0
  fn = 0
  for prediction, label in zip(predictions, labels):
    if prediction >= threshold:
      if label == 1:
        tp += 1
      else:
        fp += 1
    else:
      if label == 0:
        tn += 1
      else:
        fn += 1
  return (tp, tn, fp, fn)

def OverconfidentFalseNegatives(predictions, labels, threshold=0.1):
    tp, tn, fp, fn = confusion_matrix(predictions, labels, threshold)
    OFN = fn/(tn+fn)
    return OFN

def OverconfidentFalseRate(predictions, labels):
    error_count = 0
    count = 0
    for prediction, label in zip(predictions, labels):
        pred_label = 1 if prediction > 0.5 else 0
        if(prediction > 0.9 or prediction < 0.1) and (pred_label != label):
            error_count += 1
        if(prediction > 0.9 or prediction < 0.1):
            count += 1
    OFR = error_count/count if count > 0 else 0

    return OFR

def precision(targets: List[int],
             preds: Union[List[float], List[List[float]]],
             threshold: float = 0.5) -> float:
    
    if type(preds[0]) == list:  # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds]  # binary prediction
    
    return precision_score(targets, hard_preds)

def recall(targets: List[int],
             preds: Union[List[float], List[List[float]]],
             threshold: float = 0.5) -> float:
    
    if type(preds[0]) == list:  # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds]  # binary prediction
    
    return recall_score(targets, hard_preds)

def Brier(targets: List[int],
             preds: Union[List[float], List[List[float]]],
             threshold: float = 0.5) -> float:
    
    return (brier_score_loss(targets, preds))