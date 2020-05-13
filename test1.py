import pm4py
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.algo.discovery.inductive.util import shared_constants

# print(shared_constants.NOISE_THRESHOLD)
# log = xes_importer.import_log('/home/tsai/pm4py_mod/test1.xes')
# log = xes_importer.import_log('/home/tsai/pm4py_mod/test2.xes')


from pm4py.algo.discovery.inductive import factory as inductive_miner
from pm4py.algo.discovery.inductive.versions.dfg import dfg_based
# tree_dfg=dfg_based.apply_tree(log,parameters={'noiseThreshold':0.2})
# print(tree_dfg)

# net, initial_marking, final_marking = inductive_miner.apply(log,parameters={'noiseThreshold':0.2},variant=inductive_miner.DFG_BASED)

from pm4py.algo.discovery.inductive.versions.log import basic
# net, initial_marking, final_marking=basic.apply(log,parameters={'noiseThreshold':0.2})
# tree_basic=basic.apply_tree(log,parameters={'noiseThreshold':0.2})
# print(tree_basic)

# net, initial_marking, final_marking = inductive_miner.apply(log,parameters={'noiseThreshold':0.2},variant=inductive_miner.LOG_BASIC)


from pm4py.visualization.petrinet import factory as pn_vis_factory
# gviz_pn = pn_vis_factory.apply(net, initial_marking, final_marking)
# pn_vis_factory.view(gviz_pn)