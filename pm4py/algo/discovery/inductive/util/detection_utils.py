from copy import copy
import numpy as np
import networkx as nx


def get_connected_components(ingoing, outgoing, activities):
    """
    Get connected components in the DFG graph

    Parameters
    -----------
    ingoing
        Ingoing attributes
    outgoing
        Outgoing attributes
    activities
        Activities to consider
    """
    activities_considered = set()

    connected_components = []

    for act in ingoing:
        ingoing_act = set(ingoing[act].keys())
        if act in outgoing:
            ingoing_act = ingoing_act.union(set(outgoing[act].keys()))

        ingoing_act.add(act)

        if ingoing_act not in connected_components:
            connected_components.append(ingoing_act)
            activities_considered = activities_considered.union(set(ingoing_act))

    for act in outgoing:
        if act not in ingoing:
            outgoing_act = set(outgoing[act].keys())
            outgoing_act.add(act)
            if outgoing_act not in connected_components:
                connected_components.append(outgoing_act)
            activities_considered = activities_considered.union(set(outgoing_act))

    for activ in activities:
        if activ not in activities_considered:
            added_set = set()
            added_set.add(activ)
            connected_components.append(added_set)
            activities_considered.add(activ)

    max_it = len(connected_components)
    for it in range(max_it - 1):
        something_changed = False

        old_connected_components = copy(connected_components)
        connected_components = []

        for i in range(len(old_connected_components)):
            conn1 = old_connected_components[i]

            if conn1 is not None:
                for j in range(i + 1, len(old_connected_components)):
                    conn2 = old_connected_components[j]
                    if conn2 is not None:
                        inte = conn1.intersection(conn2)

                        if len(inte) > 0:
                            conn1 = conn1.union(conn2)
                            something_changed = True
                            old_connected_components[j] = None

            if conn1 is not None and conn1 not in connected_components:
                connected_components.append(conn1)

        if not something_changed:
            break

    return connected_components


def perform_list_union(lst):
    """
    Performs the union of a list of sets

    Parameters
    ------------
    lst
        List of sets

    Returns
    ------------
    un_set
        United set
    """
    ret = set()
    for s in lst:
        ret = ret.union(s)
    return ret


def get_connection_matrix(strongly_connected_components, dfg):
    """
    Gets the connection matrix between connected components

    Parameters
    ------------
    strongly_connected_components
        Strongly connected components
    dfg
        DFG

    Returns
    ------------
    connection_matrix
        Matrix reporting the connections
    """
    act_to_scc = {}
    for index, comp in enumerate(strongly_connected_components):
        for act in comp:
            act_to_scc[act] = index
    conn_matrix = np.zeros((len(strongly_connected_components), len(strongly_connected_components)))
    for el in dfg:
        comp_el_0 = act_to_scc[el[0][0]]
        comp_el_1 = act_to_scc[el[0][1]]
        if not comp_el_0 == comp_el_1:
            conn_matrix[comp_el_1][comp_el_0] = 1
            if conn_matrix[comp_el_0][comp_el_1] == 0:
                conn_matrix[comp_el_0][comp_el_1] = -1
    return conn_matrix


def check_if_comp_is_completely_unconnected(conn1, conn2, ingoing, outgoing):
    """
    Checks if two connected components are completely unconnected each other

    Parameters
    -------------
    conn1
        First connected component
    conn2
        Second connected component
    ingoing
        Ingoing dictionary
    outgoing
        Outgoing dictionary

    Returns
    -------------
    boolean
        Boolean value that tells if the two connected components are completely unconnected
    """
    for act1 in conn1:
        for act2 in conn2:
            if ((act1 in outgoing and act2 in outgoing[act1]) and (
                    act1 in ingoing and act2 in ingoing[act1])):
                return False
    return True


def merge_connected_components(conn_components, ingoing, outgoing):
    """
    Merge the unconnected connected components

    Parameters
    -------------
    conn_components
        Connected components
    ingoing
        Ingoing dictionary
    outgoing
        Outgoing dictionary

    Returns
    -------------
    conn_components
        Merged connected components
    """
    i = 0
    while i < len(conn_components):
        conn1 = conn_components[i]
        j = i + 1
        while j < len(conn_components):
            conn2 = conn_components[j]
            if check_if_comp_is_completely_unconnected(conn1, conn2, ingoing, outgoing):
                conn_components[i] = set(conn_components[i]).union(set(conn_components[j]))
                del conn_components[j]
                continue
            j = j + 1
        i = i + 1
    return conn_components


def transform_dfg_to_directed_nx_graph(activities, dfg):
    """
    Transform DFG to directed NetworkX graph

    Parameters
    ------------
    activities
        Activities of the graph
    dfg
        DFG

    Returns
    ------------
    G
        NetworkX digraph
    nodes_map
        Correspondence between digraph nodes and activities
    """
    G = nx.DiGraph()
    for act in activities:
        G.add_node(act)
    for el in dfg:
        act1 = el[0][0]
        act2 = el[0][1]
        G.add_edge(act1, act2)
    return G


def check_par_cut(conn_components, ingoing, outgoing):
    """
    Checks if in a parallel cut all relations are present

    Parameters
    -----------
    conn_components
        Connected components
    ingoing
        Ingoing edges to activities
    outgoing
        Outgoing edges to activities
    """
    conn_components = merge_connected_components(conn_components, ingoing, outgoing)
    conn_components = sorted(conn_components, key=lambda x: len(x))
    sthing_changed = True
    while sthing_changed:
        sthing_changed = False
        i = 0
        while i < len(conn_components):
            ok_comp_idx = []
            partly_ok_comp_idx = []
            not_ok_comp_idx = []
            conn1 = conn_components[i]
            j = i + 1
            while j < len(conn_components):
                count_good = 0
                count_notgood = 0
                conn2 = conn_components[j]
                for act1 in conn1:
                    for act2 in conn2:
                        if not ((act1 in outgoing and act2 in outgoing[act1]) and (
                                act1 in ingoing and act2 in ingoing[act1])):
                            count_notgood = count_notgood + 1
                            if count_good > 0:
                                break
                        else:
                            count_good = count_good + 1
                            if count_notgood > 0:
                                break
                if count_notgood == 0:
                    ok_comp_idx.append(j)
                elif count_good > 0:
                    partly_ok_comp_idx.append(j)
                else:
                    not_ok_comp_idx.append(j)
                j = j + 1
            if not_ok_comp_idx or partly_ok_comp_idx:
                if partly_ok_comp_idx:
                    conn_components[i] = set(conn_components[i]).union(set(conn_components[partly_ok_comp_idx[0]]))
                    del conn_components[partly_ok_comp_idx[0]]
                    sthing_changed = True
                    continue
                else:
                    return False
            if sthing_changed:
                break
            i = i + 1
    if len(conn_components) > 1:
        return conn_components
    return None
