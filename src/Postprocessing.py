from utils import *
from numpy import zeros
from operator import itemgetter

#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: # accuracy
#######################################################################################################################
""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""
def enforce_demographic_parity(categorical_results, epsilon):
    demographic_parity_data = {}
    thresholds = {}

    # Must complete this function!
    # return demographic_parity_data, thresholds

    return None, None

#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""
def enforce_equal_opportunity(categorical_results, epsilon):

    thresholds = {}
    equal_opportunity_data = {}

    # Must complete this function!
    # return equal_opportunity_data, thresholds

    return None, None

#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):
    maximum_profit_data = dict()
    thresholds = dict()

    accuracy_dict = dict()
    threshold = 0.00
    while threshold <= 1.00:
        for race, pairs in categorical_results.items():
            threshed = apply_threshold(pairs, threshold)
            maximum_profit_data[race] = threshed
            if race not in accuracy_dict:
                accuracy_dict[race] = [(threshold, get_accuracy(threshed))]
            else:
                accuracy_dict[race].append((threshold, get_accuracy(threshed)))
        threshold += 0.01

    for race, pairs in accuracy_dict.items():
        max_pair = max(pairs, key=lambda x:x[1])
        thresholds[race] = round(max_pair[0], 2)

    return maximum_profit_data, thresholds

#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):
    maximum_profit_data = dict()
    thresholds = dict()

    threshold = 0.00
    race_dict = dict()
    accuracy_dict = dict()
    for key, pairs in categorical_results.items():
        threshed = apply_threshold(pairs, threshold)
        ppv = get_positive_predictive_value(threshed)

        if ppv not in race_dict:
            race_dict[ppv] = [(key, threshold)]
        else:
            race_dict[ppv].append((key, threshold))
        accuracy_dict[key] = threshed

    # Must complete this function!
    # return predictive_parity_data, thresholds

    return None, None


def predictive_parity(results):
    thresh = 0.00
    thresh_ppv = []

    for i in range(0, 101):
        thresh_ppv.append((i/100, utils.get_true_positive_rate(utils.apply_threshold(results, thresh))))
        thresh += 0.01

    return thresh_ppv

    ###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):
    single_threshold_data = {}
    thresholds = {}

    # Must complete this function!
    # return single_threshold_data, thresholds

    return None, None
