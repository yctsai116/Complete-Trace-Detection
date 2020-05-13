import networkx as nx

from pm4py.objects.dfg.utils.dfg_utils import negate, get_activities_dirlist, get_activities_self_loop, \
    get_activities_direction
from pm4py.objects.dfg.utils.dfg_utils import get_ingoing_edges, get_outgoing_edges, get_activities_from_dfg
from pm4py.algo.discovery.inductive.util import detection_utils, cut_detection
from copy import copy
from collections import Counter


class SubtreeBasic():
    def __init__(self, traces, activities, counts, rec_depth, parent=None, noise_threshold=0, second_iteration=False,
                 rec_must_insert_skip=False):
        """
        Constructor

        Parameters
        -----------

        """
        self.traces = traces
        self.activities = activities
        self.counts = counts
        self.rec_depth = rec_depth
        self.parent = parent
        self.noise_threshold = noise_threshold
        self.second_iteration = second_iteration
        self.rec_must_insert_skip = rec_must_insert_skip

        self.start_activities = None
        self.end_activities = None
        self.activities_occurrences = None
        self.dfg = None
        self.initial_dfg = None
        self.self_loop_activities = None
        self.activities_direction = None
        self.activities_dir_list = None
        self.outgoing = None
        self.ingoing = None
        self.negated_dfg = None
        self.negated_activities = None
        self.negated_outgoing = None
        self.negated_ingoing = None
        self.detected_cut = None
        self.detected_cut_add_info = None
        self.children = None
        self.second_tree = None
        self.cut_value = 0

        self.contains_empty_traces = False
        self.must_insert_skip = False
        self.need_loop_on_subtree = False

    def initialize_tree(self):
        """
        Initialize the tree
        """
        if self.activities is None:
            self.activities = list(set(y for x in self.traces for y in x))
        else:
            if self.parent is not None and self.parent.detected_cut_add_info == "loop":
                self.get_traces_loop()
            else:
                self.traces = self.get_traces_general()

            if self.second_iteration:
                self.traces, self.activities = self.clean_traces_noise()

        self.start_activities = list(set(x[0] for x in self.traces if x))
        self.end_activities = list(set(x[-1] for x in self.traces if x))
        self.activities_occurrences = Counter([y for x in self.traces for y in x])
        self.dfg = Counter((x[i - 1], x[i]) for x in self.traces for i in range(1, len(x)))
        self.dfg = [(x, y) for x, y in self.dfg.items()]
        self.initial_dfg = self.dfg

        self.outgoing = get_outgoing_edges(self.dfg)
        self.ingoing = get_ingoing_edges(self.dfg)
        self.self_loop_activities = get_activities_self_loop(self.dfg)
        self.activities_direction = get_activities_direction(self.dfg, self.activities)
        self.activities_dir_list = get_activities_dirlist(self.activities_direction)
        self.negated_dfg = negate(self.dfg)
        self.negated_activities = get_activities_from_dfg(self.negated_dfg)
        self.negated_outgoing = get_outgoing_edges(self.negated_dfg)
        self.negated_ingoing = get_ingoing_edges(self.negated_dfg)

        self.contains_empty_traces = min(len(x) for x in self.traces) == 0 if len(self.traces) > 0 else False
        self.must_insert_skip = self.contains_empty_traces
        if self.parent is not None and self.parent.detected_cut == "xor":
            self.must_insert_skip = False
        self.must_insert_skip = self.rec_must_insert_skip or self.must_insert_skip

        if not self.second_iteration:
            self.second_tree = self.clone_second_it()

        self.detected_cut = None
        self.children = []

    def clean_traces_noise(self):
        """
        Clean noisy traces as the specified noise threshold
        """
        traces_val_ordered = sorted([y for y in self.traces.values()], reverse=True)
        total_sum = sum(traces_val_ordered) if traces_val_ordered else 0
        total_sum_wo_noise = total_sum * (1.0 - self.noise_threshold)
        if total_sum > 0:
            partial_sum = 0
            i = 0
            while i < len(traces_val_ordered):
                partial_sum += traces_val_ordered[i]
                if partial_sum >= total_sum_wo_noise:
                    break
                i = i + 1
            min_allowed = traces_val_ordered[i]
        allowed_traces = Counter({x: y for x, y in self.traces.items() if y >= min_allowed})
        activities = list(set(y for x in allowed_traces for y in x))
        return allowed_traces, activities

    def clone_second_it(self):
        """
        Clone the tree in order to perform noise cleaning
        on a second iteration
        """
        tree = SubtreeBasic(copy(self.traces), copy(self.activities), copy(self.counts), self.rec_depth,
                            noise_threshold=self.noise_threshold, second_iteration=True, parent=self.parent,
                            rec_must_insert_skip=self.rec_must_insert_skip)
        tree.need_loop_on_subtree = self.need_loop_on_subtree
        return tree

    def get_traces_general(self):
        """
        Project the traces on the activities of this tree
        """
        red_traces = Counter()
        red_dic = {x: tuple(y for y in x if y in self.activities) for x in self.traces}
        for x in red_dic:
            red_traces[red_dic[x]] += self.traces[x]
        return red_traces

    def get_traces_loop(self):
        """
        Get traces of the subcomponents of a loop cut
        (whether the parent has 2 childs, DO(EXIT) and REDO)
        """
        do_part, redo_part = self.parent.children[0], self.parent.children[1]
        if self == do_part:
            red_traces_do, red_traces_redo = Counter(), Counter()
            for x in self.traces:
                do_tr, redo_tr = [], []
                prev_read_do, prev_read_redo, prev_do_sa, prev_do_ea = False, False, False, True
                i = 0
                while i < len(x):
                    reading_do, reading_redo = x[i] in do_part.activities, x[i] in redo_part.activities
                    do_sa, do_ea = x[i] in self.parent.start_activities, x[i] in self.parent.end_activities
                    if do_sa and (prev_do_ea or prev_read_redo):
                        do_tr.append([])
                    if reading_redo and prev_read_do:
                        redo_tr.append([])
                    if reading_do:
                        do_tr[-1].append(x[i])
                    if reading_redo:
                        redo_tr[-1].append(x[i])
                    prev_read_do, prev_read_redo, prev_do_sa, prev_do_ea = reading_do, reading_redo, do_sa, do_ea
                    i = i + 1
                if len(do_tr) == 0:
                    do_tr.append([])
                if len(redo_tr) == 0 and (self.parent.detected_cut == "sequential" or len(do_tr) > 1):
                    redo_tr.append([])
                for subt in do_tr:
                    red_traces_do[tuple(subt)] += self.traces[x]
                for subt in redo_tr:
                    red_traces_redo[tuple(subt)] += self.traces[x]
            do_part.traces = red_traces_do
            redo_part.traces = red_traces_redo

    def check_sa_ea_for_each_branch(self, conn_components):
        """
        Checks if each branch of the parallel cut has a start
        and an end node of the subgraph

        Parameters
        --------------
        conn_components
            Parallel cut

        Returns
        -------------
        boolean
            True if each branch of the parallel cut has a start and an end node
        """
        parallel_cut_sa = self.start_activities
        parallel_cut_ea = self.end_activities

        if conn_components is None:
            return False

        for comp in conn_components:
            comp_sa_ok = False
            comp_ea_ok = False

            for sa in parallel_cut_sa:
                if sa in comp:
                    comp_sa_ok = True
                    break
            for ea in parallel_cut_ea:
                if ea in comp:
                    comp_ea_ok = True
                    break

            if not (comp_sa_ok and comp_ea_ok):
                return False

        return True

    def detect_parallel_cut(self):
        """
        Detects parallel cut
        """
        conn_components = detection_utils.get_connected_components(self.negated_ingoing, self.negated_outgoing,
                                                                   self.activities)

        if len(conn_components) > 1:
            conn_components = detection_utils.check_par_cut(conn_components, self.ingoing, self.outgoing)

            if self.check_sa_ea_for_each_branch(conn_components):
                return [True, conn_components]

        return [False, []]

    def detect_cut(self):
        """
        Detect generally a cut in the graph (applying all the algorithms)
        """
        if not self.second_iteration:
            self.second_tree.initialize_tree()
            self.second_tree.detect_cut()

        if self.dfg and len(self.activities) > 1:
            if self.contains_empty_traces:
                self.traces = Counter({x: y for x, y in self.traces.items() if len(x) > 0})
            conn_components = detection_utils.get_connected_components(self.ingoing, self.outgoing, self.activities)
            this_nx_graph = detection_utils.transform_dfg_to_directed_nx_graph(self.activities, self.dfg)
            strongly_connected_components = [list(x) for x in nx.strongly_connected_components(this_nx_graph)]
            xor_cut = cut_detection.detect_xor_cut(self.dfg, conn_components)

            if xor_cut[0]:
                for comp in xor_cut[1]:
                    self.detected_cut = "xor"
                    self.cut_value = 4
                    self.children.append(SubtreeBasic(self.traces, comp, self.counts, self.rec_depth + 1,
                                                      noise_threshold=self.noise_threshold, parent=self,
                                                      rec_must_insert_skip=self.rec_must_insert_skip))
            else:
                seq_cut = cut_detection.detect_sequential_cut(self.dfg, strongly_connected_components)
                if seq_cut[0]:
                    self.detected_cut = "sequential"
                    self.cut_value = 3
                    for child in seq_cut[1]:
                        self.children.append(SubtreeBasic(self.traces, child, self.counts, self.rec_depth + 1,
                                                          noise_threshold=self.noise_threshold, parent=self,
                                                          rec_must_insert_skip=self.rec_must_insert_skip))
                else:
                    par_cut = self.detect_parallel_cut()
                    if par_cut[0]:
                        self.detected_cut = "parallel"
                        self.cut_value = 2
                        for comp in par_cut[1]:
                            self.children.append(SubtreeBasic(self.traces, comp, self.counts, self.rec_depth + 1,
                                                              noise_threshold=self.noise_threshold, parent=self,
                                                              rec_must_insert_skip=self.rec_must_insert_skip))
                    else:
                        loop_cut = cut_detection.detect_loop_cut(self.dfg, self.activities, self.start_activities,
                                                                 self.end_activities)
                        if loop_cut[0]:
                            self.detected_cut = "loopCut"
                            self.detected_cut_add_info = "loop"
                            self.cut_value = 1
                            for index_enum, child in enumerate(loop_cut[1]):
                                next_subtree = SubtreeBasic(self.traces, child, self.counts, self.rec_depth + 1,
                                                            noise_threshold=self.noise_threshold, parent=self,
                                                            rec_must_insert_skip=self.rec_must_insert_skip)
                                self.children.append(next_subtree)
                        else:
                            self.detected_cut = "flower"
        else:
            self.detected_cut = "base_xor"

        kept_tree = self
        if not self.second_iteration:
            if self.cut_value >= self.second_tree.cut_value:
                kept_tree = self
            else:
                kept_tree = self.second_tree
            kept_tree.detect_cut_in_children()

        return kept_tree

    def detect_cut_in_children(self):
        """
        Applies the cut detection in each child
        """
        i = 0
        while i < len(self.children):
            self.children[i].initialize_tree()
            self.children[i] = self.children[i].detect_cut()
            i = i + 1
