import sys
from collections import Counter
import pandas as pd

from pm4py import util as pmutil
from pm4py.algo.discovery.inductive.util import shared_constants, get_tree_repr
from pm4py.algo.discovery.inductive.util.petri_el_count import Counts
from pm4py.algo.discovery.inductive.versions.log.data_structures.subtree_basic import SubtreeBasic
from pm4py.objects.conversion.process_tree import factory as tree_to_petri
from pm4py.objects.conversion.log import factory as log_conversion
from pm4py.statistics.variants.pandas import get as variants_get
from pm4py.util import xes_constants as xes_util
from pm4py.objects.process_tree import util as tree_util

sys.setrecursionlimit(shared_constants.REC_LIMIT)

PARAMETER_VARIANT_SEP = pmutil.constants.PARAMETER_VARIANT_SEP
DEFAULT_VARIANT_SEP = pmutil.constants.DEFAULT_VARIANT_SEP


def apply(log, parameters=None):
    """
    Apply the IMDF algorithm to a log obtaining a Petri net along with an initial and final marking

    Parameters
    -----------
    log
        Log
    parameters
        Parameters of the algorithm, including:
            pmutil.constants.PARAMETER_CONSTANT_ACTIVITY_KEY -> attribute of the log to use as activity name
            (default concept:name)

    Returns
    -----------
    net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    """
    if parameters is None:
        parameters = {}
    if pmutil.constants.PARAMETER_CONSTANT_ACTIVITY_KEY not in parameters:
        parameters[pmutil.constants.PARAMETER_CONSTANT_ACTIVITY_KEY] = xes_util.DEFAULT_NAME_KEY
    if pmutil.constants.PARAMETER_CONSTANT_TIMESTAMP_KEY not in parameters:
        parameters[pmutil.constants.PARAMETER_CONSTANT_TIMESTAMP_KEY] = xes_util.DEFAULT_TIMESTAMP_KEY
    if pmutil.constants.PARAMETER_CONSTANT_CASEID_KEY not in parameters:
        parameters[pmutil.constants.PARAMETER_CONSTANT_CASEID_KEY] = pmutil.constants.CASE_ATTRIBUTE_GLUE
    if type(log) is pd.DataFrame:
        return apply_variants(variants_get.get_variants_set(log, parameters=parameters), parameters=parameters)
    else:
        log = log_conversion.apply(log, parameters, log_conversion.TO_EVENT_LOG)
        tree = apply_tree(log, parameters=parameters)
        net, initial_marking, final_marking = tree_to_petri.apply(tree)
        return net, initial_marking, final_marking


def apply_tree(log, parameters=None):
    """
    Apply the IMDF algorithm to a log obtaining a process tree

    Parameters
    ----------
    log
        Log
    parameters
        Parameters of the algorithm, including:
            pmutil.constants.PARAMETER_CONSTANT_ACTIVITY_KEY -> attribute of the log to use as activity name
            (default concept:name)

    Returns
    ----------
    tree
        Process tree
    """
    if parameters is None:
        parameters = {}
    if pmutil.constants.PARAMETER_CONSTANT_ACTIVITY_KEY not in parameters:
        parameters[pmutil.constants.PARAMETER_CONSTANT_ACTIVITY_KEY] = xes_util.DEFAULT_NAME_KEY
    if pmutil.constants.PARAMETER_CONSTANT_TIMESTAMP_KEY not in parameters:
        parameters[pmutil.constants.PARAMETER_CONSTANT_TIMESTAMP_KEY] = xes_util.DEFAULT_TIMESTAMP_KEY
    if pmutil.constants.PARAMETER_CONSTANT_CASEID_KEY not in parameters:
        parameters[pmutil.constants.PARAMETER_CONSTANT_CASEID_KEY] = pmutil.constants.CASE_ATTRIBUTE_GLUE
    if type(log) is pd.DataFrame:
        return apply_tree_variants(variants_get.get_variants_set(log, parameters=parameters), parameters=parameters)
    else:
        activity_key = parameters[pmutil.constants.PARAMETER_CONSTANT_ACTIVITY_KEY]
        traces = Counter([tuple(y[activity_key] for y in x if activity_key in y) for x in log])
        return apply_internal(traces)


def apply_variants(variants, parameters=None):
    """
    Apply the IMDF algorithm to a dictionary/list/set of variants obtaining a Petri net along with an initial and final marking

    Parameters
    -----------
    variants
        Dictionary/list/set of variants in the log
    parameters
        Parameters of the algorithm, including:
            pmutil.constants.PARAMETER_CONSTANT_ACTIVITY_KEY -> attribute of the log to use as activity name
            (default concept:name)

    Returns
    -----------
    net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    """
    if parameters is None:
        parameters = {}
    tree = apply_tree_variants(variants, parameters=parameters)
    net, initial_marking, final_marking = tree_to_petri.apply(tree)
    return net, initial_marking, final_marking


def apply_tree_variants(variants, parameters=None):
    """
    Apply the IMDF algorithm to a dictionary/list/set of variants a log obtaining a process tree

    Parameters
    ----------
    variants
        Dictionary/list/set of variants in the log
    parameters
        Parameters of the algorithm, including:
            pmutil.constants.PARAMETER_CONSTANT_ACTIVITY_KEY -> attribute of the log to use as activity name
            (default concept:name)

    Returns
    ----------
    tree
        Process tree
    """
    if parameters is None:
        parameters = {}

    variant_sep = parameters[PARAMETER_VARIANT_SEP] if PARAMETER_VARIANT_SEP in parameters else DEFAULT_VARIANT_SEP
    if type(variants) is set or type(variants) is list:
        traces = Counter([tuple(x.split(variant_sep)) for x in variants])
    else:
        traces = Counter({tuple(x.split(variant_sep)): len(y) if type(y) is list else y for x, y in variants.items()})
    return apply_internal(traces)


def apply_dfg(dfg, parameters=None, activities=None, contains_empty_traces=False, start_activities=None,
              end_activities=None):
    raise Exception("apply_dfg not implemented in the log version!")


def apply_tree_dfg(dfg, parameters=None, activities=None, contains_empty_traces=False, start_activities=None,
                   end_activities=None):
    raise Exception("apply_tree_dfg not implemented in the log version!")


def apply_internal(traces, parameters=None):
    if parameters is None:
        parameters = {}

    noise_threshold = shared_constants.NOISE_THRESHOLD

    c = Counts()
    s = SubtreeBasic(traces, None, c, rec_depth=0, noise_threshold=noise_threshold)
    s.initialize_tree()
    s = s.detect_cut()

    tree_repr = get_tree_repr.get_repr(s, 0, contains_empty_traces=s.contains_empty_traces)
    #tree_repr = tree_util.fold(tree_repr)

    return tree_repr
