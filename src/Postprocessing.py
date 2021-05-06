import utils
from numpy import zeros

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
    thresholds = dict()

    for e in list(categorical_results.items()):
        thresholds[e[0]] = max_profit(e[1])

    print()
    print('%16s %10s' % ('RACE', 'THRESHOLD'))
    for key in thresholds:
        print('%16s: %.2f' % (key, thresholds[key]))
    print()

    return None, None


# determines the threshold that maximizes accuracy for a single category
def max_profit(results):
    thresh = 0.00
    accuracies = zeros(101)

    for i in range(0, 101):
        accuracies[i] = utils.get_accuracy(utils.apply_threshold(results, thresh))
        thresh += 0.01

    return accuracies.argmax()/100


#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):
    predictive_parity_data = {}
    thresholds = {}

    # Must complete this function!
    # return predictive_parity_data, thresholds

    return None, None

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
