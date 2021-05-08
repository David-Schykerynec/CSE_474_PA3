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


    threshold = 0
    threshold_eta = .01
    parity_eta = .01
    # Must complete this function!
    #return demographic_parity_data, threshold
    race_dict  = {}
    parity = 0
    while threshold <= 1:
        accuracy_dict = {}
        for key, pairs in categorical_results.items():
            threshed = apply_threshold(pairs, threshold)
            ppv = get_num_predicted_positives(threshed)/ len(threshed)
            acc = get_num_correct(threshed) / len(threshed)
            if ppv not in race_dict:
                race_dict[ppv] = [(key,threshold,acc,ppv)]
            else:
                race_dict[ppv].append((key,threshold,acc,ppv))
            #race_dict.setdefault(ppv,[]).append((key, threshold))
            accuracy_dict[key] = threshed

        threshold += threshold_eta
    # find different threshes with same ppv
    sorted_keys_list = race_dict.keys()
    parity_dict = {}
    while parity <= 1:
        for ppvs in sorted_keys_list:
            if abs(parity - ppvs) <= epsilon:
                lists = race_dict[ppvs]
                if parity not in parity_dict:
                    parity_dict[parity] = lists
                else:
                    parity_dict[parity] = parity_dict[parity] + lists
        parity += parity_eta
    new_parity_dict = {}
    for key,pair in parity_dict.items():
        dictionary_race = {}
        for tup in pair:
            if tup[0] in dictionary_race:
                acc_ = dictionary_race[tup[0]][2]
                if tup[2] > acc_:
                    dictionary_race[tup[0]] = (tup[1],tup[2],tup[3]) # threshold, accuracy, ppv
            else: # dictionary_not_instanstiated
                dictionary_race[tup[0]] = (tup[1],tup[2],tup[3])
        if len(dictionary_race.keys()) == 4:
            new_parity_dict[key] = dictionary_race
    # out of these parities; we choose the highest accuracy
    highest_acc, highest_par = 0,0
    for par, race_pairs in new_parity_dict.items():
        total_correct = 0
        total_length = 0
        for k,p in race_pairs.items():
            length  = len(categorical_results[k])
            accuracy = p[1]
            num_correct = length * accuracy
            total_length += length
            total_correct += num_correct
        total_accuracy = total_correct / total_length
        if highest_acc < total_accuracy:
            highest_acc, highest_par = total_accuracy, par
    threshold_dict = {}
    demographic_parity_data = {}
    for key,pair in new_parity_dict[highest_par].items():
        thresh = pair[0]
        demographic_parity_data[key] = apply_threshold(categorical_results[key], thresh)
        threshold_dict[key] = thresh
    return demographic_parity_data, threshold_dict


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
        max_pair = max(pairs, key=lambda x: x[1])
        thresholds[race] = round(max_pair[0], 2)

    return maximum_profit_data, thresholds


#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""


def enforce_predictive_parity(categorical_results, epsilon):
    demographic_parity_data = {}
    thresholds = {}

    # Must complete this function!
    # return demographic_parity_data, thresholds

    threshold = 0
    threshold_eta = .01
    parity_eta = .01
    # Must complete this function!
    # return demographic_parity_data, threshold
    race_dict = {}
    parity = 0
    while threshold <= 1:
        accuracy_dict = {}
        for key, pairs in categorical_results.items():
            threshed = apply_threshold(pairs, threshold)
            tpr = get_true_positive_rate(threshed)
            acc = get_num_correct(threshed) / len(threshed)
            if tpr not in race_dict:
                race_dict[tpr] = [(key, threshold, acc, tpr)]
            else:
                race_dict[tpr].append((key, threshold, acc, tpr))
            # race_dict.setdefault(ppv,[]).append((key, threshold))
            accuracy_dict[key] = threshed

        threshold += threshold_eta
    # find different threshes with same ppv
    sorted_keys_list = race_dict.keys()
    parity_dict = {}
    while parity <= 1:
        for ppvs in sorted_keys_list:
            if abs(parity - ppvs) <= epsilon:
                lists = race_dict[ppvs]
                if parity not in parity_dict:
                    parity_dict[parity] = lists
                else:
                    parity_dict[parity] = parity_dict[parity] + lists
        parity += parity_eta
    new_parity_dict = {}
    for key, pair in parity_dict.items():
        dictionary_race = {}
        for tup in pair:
            if tup[0] in dictionary_race:
                acc_ = dictionary_race[tup[0]][2]
                if tup[2] > acc_:
                    dictionary_race[tup[0]] = (tup[1], tup[2], tup[3])  # threshold, accuracy, ppv
            else:  # dictionary_not_instanstiated
                dictionary_race[tup[0]] = (tup[1], tup[2], tup[3])
        if len(dictionary_race.keys()) == 4:
            new_parity_dict[key] = dictionary_race
    # out of these parities; we choose the highest accuracy
    highest_acc, highest_par = 0, 0
    for par, race_pairs in new_parity_dict.items():
        total_correct = 0
        total_length = 0
        for k, p in race_pairs.items():
            length = len(categorical_results[k])
            accuracy = p[1]
            num_correct = length * accuracy
            total_length += length
            total_correct += num_correct
        total_accuracy = total_correct / total_length
        if highest_acc < total_accuracy:
            highest_acc, highest_par = total_accuracy, par
    threshold_dict = {}
    demographic_parity_data = {}
    for key, pair in new_parity_dict[highest_par].items():
        thresh = pair[0]
        demographic_parity_data[key] = apply_threshold(categorical_results[key], thresh)
        threshold_dict[key] = thresh
    return demographic_parity_data, threshold_dict

    ###################################################################################################################


""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""


def enforce_single_threshold(categorical_results):
    single_threshold_data = {}
    thresholds = {}

    threshold = 0
    threshold_eta = .01
    accuracy_list = []
    while threshold <= 1:
        race_dict = {}
        for key, pairs in categorical_results.items():
            threshed = apply_threshold(pairs, threshold)
            race_dict[key] = threshed
        accuracy = get_total_accuracy(race_dict)
        accuracy_list.append(accuracy)
        threshold += threshold_eta
    max_value = max(accuracy_list)
    max_index = accuracy_list.index(max_value)
    optimal_threshold  = threshold_eta * (max_index +1)
    race_dict_output = {}
    race_dict_thesholds = {}
    for key, pairs in categorical_results.items():
        threshed = apply_threshold(pairs, optimal_threshold)
        race_dict_output[key] = threshed
        race_dict_thesholds[key] = optimal_threshold

    return race_dict_output, race_dict_thesholds
