import networkx as nx
import numpy as np

from copy import copy

from pm4py.objects.dfg.utils.dfg_utils import get_all_activities_connected_as_input_to_activity
from pm4py.objects.dfg.utils.dfg_utils import get_all_activities_connected_as_output_to_activity
from pm4py.objects.dfg.utils.dfg_utils import filter_dfg_on_act, negate, get_activities_dirlist, \
    get_activities_self_loop, get_activities_direction
from pm4py.objects.dfg.utils.dfg_utils import get_ingoing_edges, get_outgoing_edges, get_activities_from_dfg, \
    infer_start_activities, infer_end_activities
from pm4py.objects.dfg.filtering.dfg_filtering import clean_dfg_based_on_noise_thresh
from pm4py.objects.dfg.utils.dfg_utils import infer_start_activities_from_prev_connections_and_current_dfg, \
    infer_end_activities_from_succ_connections_and_current_dfg
from pm4py.algo.discovery.inductive.util import detection_utils, cut_detection


class SubtreeDFGBasedOld():
    def __init__(self, dfg, master_dfg, initial_dfg, activities, counts, rec_depth, noise_threshold=0,
                 start_activities=None, end_activities=None, initial_start_activities=None,
                 initial_end_activities=None):
        """
        Constructor

        Parameters
        -----------
        dfg
            Directly follows graph of this subtree
        master_dfg
            Original DFG
        initial_dfg
            Referral directly follows graph that should be taken in account adding hidden/loop transitions
        activities
            Activities of this subtree
        counts
            Shared variable
        rec_depth
            Current recursion depth
        """
        self.master_dfg = copy(master_dfg)
        self.initial_dfg = copy(initial_dfg)
        self.counts = counts
        self.rec_depth = rec_depth
        self.noise_threshold = noise_threshold
        self.start_activities = start_activities
        if self.start_activities is None:
            self.start_activities = []
        self.end_activities = end_activities
        if self.end_activities is None:
            self.end_activities = []
        self.initial_start_activities = initial_start_activities
        if self.initial_start_activities is None:
            self.initial_start_activities = infer_start_activities(master_dfg)
        self.initial_end_activities = initial_end_activities
        if self.initial_end_activities is None:
            self.initial_end_activities = infer_end_activities(master_dfg)

        self.second_iteration = None
        self.activities = None
        self.dfg = None
        self.outgoing = None
        self.ingoing = None
        self.self_loop_activities = None
        self.initial_ingoing = None
        self.initial_outgoing = None
        self.activities_direction = None
        self.activities_dir_list = None
        self.negated_dfg = None
        self.negated_activities = None
        self.negated_outgoing = None
        self.negated_ingoing = None
        self.detected_cut = None
        self.children = None
        self.must_insert_skip = False
        self.need_loop_on_subtree = False

        self.initialize_tree(dfg, initial_dfg, activities)

    def initialize_tree(self, dfg, initial_dfg, activities, second_iteration=False):
        """
        Initialize the tree


        Parameters
        -----------
        dfg
            Directly follows graph of this subtree
        initial_dfg
            Referral directly follows graph that should be taken in account adding hidden/loop transitions
        activities
            Activities of this subtree
        second_iteration
            Boolean that indicates if we are executing this method for the second time
        """

        self.second_iteration = second_iteration

        if activities is None:
            self.activities = get_activities_from_dfg(dfg)
        else:
            self.activities = copy(activities)

        if second_iteration:
            self.dfg = clean_dfg_based_on_noise_thresh(self.dfg, self.activities, self.noise_threshold)
        else:
            self.dfg = copy(dfg)

        self.initial_dfg = initial_dfg

        self.outgoing = get_outgoing_edges(self.dfg)
        self.ingoing = get_ingoing_edges(self.dfg)
        self.self_loop_activities = get_activities_self_loop(self.dfg)
        self.initial_outgoing = get_outgoing_edges(self.initial_dfg)
        self.initial_ingoing = get_ingoing_edges(self.initial_dfg)
        self.activities_direction = get_activities_direction(self.dfg, self.activities)
        self.activities_dir_list = get_activities_dirlist(self.activities_direction)
        self.negated_dfg = negate(self.dfg)
        self.negated_activities = get_activities_from_dfg(self.negated_dfg)
        self.negated_outgoing = get_outgoing_edges(self.negated_dfg)
        self.negated_ingoing = get_ingoing_edges(self.negated_dfg)
        self.detected_cut = None
        self.children = []

        self.detect_cut(second_iteration=second_iteration)

    def detect_loop_cut(self):
        """
        Detect loop cut
        """
        start_activities = self.start_activities
        if len(start_activities) == 0:
            start_activities = infer_start_activities_from_prev_connections_and_current_dfg(self.initial_dfg, self.dfg,
                                                                                            self.activities)
        end_activities = self.end_activities

        end_activities = list(set(end_activities) - set(start_activities))

        if len(end_activities) == 0:
            end_activities = infer_end_activities_from_succ_connections_and_current_dfg(self.initial_dfg, self.dfg,
                                                                                        self.activities)
            end_activities = list(set(end_activities) - set(start_activities))
            if len(end_activities) == 0:
                end_activities = infer_end_activities_from_succ_connections_and_current_dfg(self.initial_dfg, self.dfg,
                                                                                            self.activities,
                                                                                            include_self=False)
        all_end_activities = copy(end_activities)
        end_activities = list(set(end_activities) - set(start_activities))
        end_activities_that_are_also_start = list(set(all_end_activities) - set(end_activities))

        do_part = []
        redo_part = []
        dangerous_redo_part = []
        exit_part = []

        for sa in start_activities:
            do_part.append(sa)
        for ea in end_activities:
            exit_part.append(ea)

        for act in self.activities:
            if act not in start_activities and act not in end_activities:
                input_connected_activities = get_all_activities_connected_as_input_to_activity(self.dfg, act)
                output_connected_activities = get_all_activities_connected_as_output_to_activity(self.dfg, act)
                if set(output_connected_activities).issubset(start_activities) and set(start_activities).issubset(
                        output_connected_activities):
                    if len(input_connected_activities.intersection(exit_part)) > 0:
                        dangerous_redo_part.append(act)
                    redo_part.append(act)
                else:
                    do_part.append(act)

        if len(do_part) > 0 and (len(redo_part) > 0 or len(exit_part)) > 0:
            if len(redo_part) > 0:
                return [True, [do_part + exit_part, redo_part], len(end_activities_that_are_also_start) > 0]
            else:
                return [True, [do_part, redo_part + exit_part], len(end_activities_that_are_also_start) > 0]

        return [False, [], []]

    def detect_sequential_cut(self, dfg, strongly_connected_components):
        """
        Detect sequential cut in DFG graph

        Parameters
        --------------
        dfg
            DFG
        strongly_connected_components
            Strongly connected components
        """
        if len(strongly_connected_components) > 1:
            orig_conn_comp = copy(strongly_connected_components)
            conn_matrix = detection_utils.get_connection_matrix(strongly_connected_components, dfg)
            something_changed = True
            while something_changed:
                something_changed = False
                i = 0
                while i < len(strongly_connected_components):
                    idx_i_comp = orig_conn_comp.index(strongly_connected_components[i])
                    j = i + 1
                    while j < len(strongly_connected_components):
                        idx_j_comp = orig_conn_comp.index(strongly_connected_components[j])
                        if conn_matrix[idx_i_comp][idx_j_comp] > 0:
                            copyel = copy(strongly_connected_components[i])
                            strongly_connected_components[i] = strongly_connected_components[j]
                            strongly_connected_components[j] = copyel
                            something_changed = True
                            break
                        j = j + 1
                    i = i + 1
            ret_connected_components = []
            ignore_comp = set()
            i = 0
            while i < len(strongly_connected_components):
                if i not in ignore_comp:
                    idx_i_comp = orig_conn_comp.index(strongly_connected_components[i])
                    comp = copy(strongly_connected_components[i])
                    j = i + 1
                    is_component_mergeable = True
                    while j < len(strongly_connected_components):
                        idx_j_comp = orig_conn_comp.index(strongly_connected_components[j])
                        if conn_matrix[idx_i_comp][idx_j_comp] < 0 or conn_matrix[idx_i_comp][idx_j_comp] > 0:
                            is_component_mergeable = False
                            break
                        j = j + 1
                    if is_component_mergeable:
                        j = i + 1
                        while j < len(strongly_connected_components):
                            idx_j_comp = orig_conn_comp.index(strongly_connected_components[j])
                            if conn_matrix[idx_i_comp][idx_j_comp] == 0:
                                comp = comp + strongly_connected_components[j]
                                ignore_comp.add(j)
                            else:
                                break
                            j = j + 1
                    else:
                        j = i + 1
                        while j < len(strongly_connected_components):
                            idx_j_comp = orig_conn_comp.index(strongly_connected_components[j])
                            # two components that have exactly the same inputs/outputs are merged
                            if np.array_equal(conn_matrix[idx_i_comp], conn_matrix[idx_j_comp]):
                                comp = comp + strongly_connected_components[j]
                                ignore_comp.add(j)
                            j = j + 1
                    ret_connected_components.append(comp)
                i = i + 1

            if len(ret_connected_components) > 1:
                return [True, ret_connected_components]
        return [False, [], []]

    def detect_parallel_cut(self):
        """
        Detects parallel cut
        """
        conn_components = detection_utils.get_connected_components(self.negated_ingoing, self.negated_outgoing, self.activities)

        if len(conn_components) > 1:
            conn_components = detection_utils.check_par_cut(conn_components, self.ingoing, self.outgoing)
            if conn_components is not None:
                for comp in conn_components:
                    comp_ok = False
                    for el in self.initial_dfg:
                        if (el[0][0] in comp and el[0][1] not in self.activities) or (
                                el[0][1] in comp and el[0][0] not in self.activities):
                            comp_ok = True
                            break
                    if self.rec_depth == 0:
                        for sa in self.start_activities:
                            if sa in comp:
                                comp_ok = True
                                #print("siii")
                                break
                    if not comp_ok:
                        return [False, conn_components]
                return [True, conn_components]

        return [False, []]

    def __str__(self):
        return "subtree rec_depth="+str(self.rec_depth)+" dfg="+str(self.dfg)+" activities="+str(self.activities)

    def __repr__(self):
        return "subtree rec_depth="+str(self.rec_depth)+" dfg="+str(self.dfg)+" activities="+str(self.activities)

    def put_skips_in_seq_cut(self):
        """
        Puts the skips in sequential cut
        """
        # first, put skips when in some cut there is an ending activity
        in_end_act = set(self.initial_end_activities)
        i = 0
        while i < len(self.children) - 1:
            activities_set = set(self.children[i].activities)
            intersection = activities_set.intersection(in_end_act)
            if len(intersection) > 0:
                j = i + 1
                while j < len(self.children):
                    self.children[j].must_insert_skip = True
                    j = j + 1
            i = i + 1

        # second, put skips when in some cut you are not sure to pass through
        i = 0
        while i < len(self.children) - 1:
            act_i = self.children[i].activities
            act_i_output_appearences = {}
            max_value = i
            for act in act_i:
                for out_act in self.outgoing[act]:
                    act_i_output_appearences[out_act] = len(self.children) - 1
            j = i + 1
            while j < len(self.children):
                act_children = self.children[j].activities
                for act in act_children:
                    if act in act_i_output_appearences and act_i_output_appearences[act] == len(self.children) - 1:
                        act_i_output_appearences[act] = j
                        if j > max_value:
                            max_value = j
                j = j + 1
            j = i + 1
            while j < max_value:
                self.children[j].must_insert_skip = True
                j = j + 1
            i = i + 1

        this_start_activities = set(infer_start_activities(self.dfg))
        # third, put skips when some input activities do not pass there
        out_start_activities = infer_start_activities_from_prev_connections_and_current_dfg(self.initial_dfg, self.dfg,
                                                                                            self.activities,
                                                                                            include_self=False)
        out_start_activities_diff = out_start_activities - set(self.activities)
        for act in out_start_activities_diff:
            out_act_here = set()
            for el in self.initial_dfg:
                if el[0][0] == act and el[0][1] in self.activities:
                    out_act_here.add(el[0][1])
            i = 0
            while i < len(self.children):
                child_act = set(self.children[i].activities)
                inte = child_act.intersection(out_act_here)
                if inte:
                    for el in inte:
                        out_act_here.remove(el)
                if len(out_act_here) > 0:
                    self.children[i].must_insert_skip = True
                i = i + 1

        # fourth, put skips until all start activities are reached
        remaining_act = (out_start_activities - this_start_activities).intersection(self.activities)
        i = 0
        while i < len(self.children):
            child_act = set(self.children[i].activities)
            inte = child_act.intersection(remaining_act)
            if inte:
                for el in inte:
                    remaining_act.remove(el)
            if len(remaining_act) > 0:
                self.children[i].must_insert_skip = True
            i = i + 1

    def put_skips_in_loop_cut(self):
        """
        Puts the skips in loop cuts
        """
        all_start_activities = infer_start_activities_from_prev_connections_and_current_dfg(self.initial_dfg, self.dfg,
                                                                                            self.activities)
        if not all_start_activities:
            self.children[0].must_insert_skip = True
            self.children[1].must_insert_skip = True
            return

    def detect_cut(self, second_iteration=False):
        """
        Detect generally a cut in the graph (applying all the algorithms)
        """
        if self.dfg:
            conn_components = detection_utils.get_connected_components(self.ingoing, self.outgoing, self.activities)
            this_nx_graph = detection_utils.transform_dfg_to_directed_nx_graph(self.activities, self.dfg)
            strongly_connected_components = [list(x) for x in nx.strongly_connected_components(this_nx_graph)]

            xor_cut = cut_detection.detect_xor_cut(self.dfg, conn_components)

            if xor_cut[0]:
                for comp in xor_cut[1]:
                    new_dfg = filter_dfg_on_act(self.dfg, comp)
                    self.detected_cut = "xor"
                    self.children.append(
                        SubtreeDFGBasedOld(new_dfg, self.master_dfg, self.initial_dfg, comp, self.counts, self.rec_depth + 1,
                                           noise_threshold=self.noise_threshold,
                                           initial_start_activities=self.initial_start_activities,
                                           initial_end_activities=self.initial_end_activities))
            else:
                seq_cut = self.detect_sequential_cut(self.dfg, strongly_connected_components)
                if seq_cut[0]:
                    self.detected_cut = "sequential"
                    for child in seq_cut[1]:
                        dfg_child = filter_dfg_on_act(self.dfg, child)
                        self.children.append(
                            SubtreeDFGBasedOld(dfg_child, self.master_dfg, self.initial_dfg, child, self.counts,
                                               self.rec_depth + 1,
                                               noise_threshold=self.noise_threshold,
                                               initial_start_activities=self.initial_start_activities,
                                               initial_end_activities=self.initial_end_activities))
                    self.put_skips_in_seq_cut()
                else:
                    par_cut = self.detect_parallel_cut()
                    if par_cut[0]:
                        self.detected_cut = "parallel"
                        for comp in par_cut[1]:
                            new_dfg = filter_dfg_on_act(self.dfg, comp)
                            self.children.append(
                                SubtreeDFGBasedOld(new_dfg, self.master_dfg, new_dfg, comp, self.counts,
                                                   self.rec_depth + 1,
                                                   noise_threshold=self.noise_threshold,
                                                   initial_start_activities=self.initial_start_activities,
                                                   initial_end_activities=self.initial_end_activities))
                    else:
                        loop_cut = self.detect_loop_cut()
                        if loop_cut[0]:
                            # print(self.rec_depth, "loop_cut", self.activities, loop_cut)
                            self.detected_cut = "loopCut"
                            for index_enum, child in enumerate(loop_cut[1]):
                                dfg_child = filter_dfg_on_act(self.dfg, child)
                                next_subtree = SubtreeDFGBasedOld(dfg_child, self.master_dfg, self.initial_dfg, child,
                                                                  self.counts, self.rec_depth + 1,
                                                                  noise_threshold=self.noise_threshold,
                                                                  initial_start_activities=self.initial_start_activities,
                                                                  initial_end_activities=self.initial_end_activities)
                                if loop_cut[2] and index_enum > 0:
                                    next_subtree.force_loop_hidden = True
                                self.children.append(next_subtree)
                            self.put_skips_in_loop_cut()
                        else:
                            if self.noise_threshold > 0:
                                if not second_iteration:
                                    self.initialize_tree(self.dfg, self.initial_dfg, None, second_iteration=True)
                            else:
                                pass
                            self.detected_cut = "flower"
        else:
            self.detected_cut = "base_xor"
