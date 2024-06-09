import pickle
import os
import abc_py
import numpy as np
import torch
from tqdm import tqdm

def get_data(state):
    _abc = abc_py.AbcInterface()
    _abc.start()
    _abc.read(state)
    data = {}
    numNodes = _abc.numNodes()
    data['node_type'] = np.zeros(numNodes, dtype = int)
    data['num_inverted_predecessors'] = np.zeros(numNodes, dtype=int)
    edge_src_index = []
    edge_target_index = []
    for nodeIdx in range(numNodes):
        aigNode = _abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()
        data['num_inverted_predecessors'][nodeIdx] = 0
        if nodeType == 0 or nodeType == 2:
            data['node_type'][nodeIdx] = 0
        elif nodeType == 1:
            data['node_type'][nodeIdx] = 1
        else:
            data['node_type'][nodeIdx] = 2
            if nodeType == 4:
                data['num_inverted_predecessors'][nodeIdx] = 1
            if nodeType == 5:
                data['num_inverted_predecessors'][nodeIdx] = 2
        if(aigNode.hasFanin0()):
            fanin = aigNode.fanin0()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
        if(aigNode.hasFanin1()):
            fanin = aigNode.fanin1()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
    data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype=torch.long)
    data['node_type'] = torch.tensor([data['node_type']])
    data['num_inverted_predecessors'] = torch.tensor([data['num_inverted_predecessors']])
    data['nodes'] = numNodes
    return data

def get_aig(state):
    circuitName, actions = state.split('_')
    circuitPath = './InitialAIG/train/' + circuitName + '.aig'
    libFile = './lib/7nm/7nm.lib'
    logFile = './log/' + state + '.log'
    nextState = './aig/' + state + '.aig' # current AIG file
    if os.path.exists(nextState):
        return get_data(nextState)
    synthesisOpToPosDic = {
        0: "refactor",
        1: "refactor -z",
        2: "rewrite",
        3: "rewrite -z",
        4: "resub",
        5: "resub -z",
        6: "balance"
    }
    actionCmd = ''
    for action in actions:
        actionCmd += (synthesisOpToPosDic[int(action)] + ';')
    abcRunCmd = "../yosys-main/yosys-abc -c \"read " + circuitPath + ";" + actionCmd + "; read_lib " + libFile + "; write " + nextState + "; print_stats\" > " + logFile
    os.system(abcRunCmd)

def main():
    train_data = []
    i = 0
    data_folder = "./project_data" # ./project_data2
    for file in tqdm(os.listdir(data_folder)):
        n = int(file.split('_')[1].split('.')[0])
        if n % 30 == 0:
            i += 1
            with open(os.path.join(data_folder, file), "rb") as f:
                state_targets = pickle.load(f)
                states = state_targets['input']
                targets = state_targets['target']
                for state, target in zip(states, targets):
                    circuit = state.split('_')[0]
                    get_aig(state)
                    data = get_data('./aig/' + state + '.aig')
                    data['target'] = target
                    train_data.append(data)
            if i % 500 == 0:
                torch.save(train_data, 'task1_set_{0}.pt'.format(i))
                train_data = []
if __name__ == "__main__":
    main()
