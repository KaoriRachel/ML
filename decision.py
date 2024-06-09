import torch
import os
import numpy as np
import abc_py
from matplotlib import pyplot as plt
from tqdm import *
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch.nn import Dropout
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import argparse
import torch.optim as optim

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search', default='greedy', type=str, required=False)
    parser.add_argument('--initial', type=str, required=True)
    return parser.parse_args()

def get_next_data(circuitName,actions):
    circuitPath = './InitialAIG/train/' + circuitName + '.aig'
    libFile = './lib/7nm/7nm.lib'
    logFile = './log/' + circuitName + '_' + actions + '.log'
    nextState = './aig/' + circuitName + '_' + actions + '.aig'
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
    action_cmd = ""
    for action in actions:
        action_cmd += (synthesisOpToPosDic[int(action)] + ';')
    abcRunCmd = f"../yosys-main/yosys-abc -c \"read {circuitPath}; {action_cmd} read_lib {libFile}; write {nextState}; print_stats\" > {logFile}"
    os.system(abcRunCmd)
    return get_data(nextState)

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

# Search algorithm
def greedy_search(circuitName, model, max_steps=10, cur_actions=''):
    for step in range(max_steps):
        print(step)
        best_reward = -float('inf')
        best_action = 0
        for action in range(7):  # 7 possible actions
            data = get_next_data(circuitName, cur_actions+str(action))
            x = torch.cat([data['node_type'], data['num_inverted_predecessors']], dim=1)
            x = x.reshape(2, -1).T
            x = torch.tensor(x, dtype=torch.float32)
            edge_index = data['edge_index']
            new_data = Data(x=x, edge_index=edge_index)
            reward = model(new_data)
            if reward > best_reward:
                best_reward = reward
                best_action = action
        cur_actions += str(best_action)
    return circuitName + '_' + cur_actions


# Search algorithm
def beam_search(circuitName, model, beam_width=3, max_steps=10, cur_actions=''):
    # Initialize the beam with the initial state
    beam = [(circuitName, cur_actions)]

    for step in range(max_steps):
        print(step)
        candidates = []

        for circuitName, actions in beam:
            for action in range(7):  # 7 possible actions
                new_actions = actions + str(action)
                data = get_next_data(circuitName, new_actions)
                x = torch.cat([data['node_type'], data['num_inverted_predecessors']], dim=1)
                x = x.reshape(2, -1).T
                x = torch.tensor(x, dtype=torch.float32)
                edge_index = data['edge_index']
                new_data = Data(x=x, edge_index=edge_index)

                with torch.no_grad():
                    reward = model(new_data).item()

                candidates.append((circuitName, new_actions, reward))

        # Sort the candidates by reward and select the top `beam_width` candidates
        candidates.sort(key=lambda x: x[2], reverse=True)
        beam = [(circuitName, actions) for circuitName, actions, reward in candidates[:beam_width]]

    # Return the best sequence of actions from the beam
    best_circuitName, best_actions = max(beam, key=lambda x: x[1])
    return best_circuitName + '_' + best_actions
 
# Search algorithm
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.reward = 0.0

    def add_child(self, child_node):
        self.children.append(child_node)

    def update(self, reward):
        self.visits += 1
        self.reward += reward


def select_node(node):
    best_child = max(node.children, key=lambda c: c.reward / c.visits + np.sqrt(2 * np.log(node.visits) / c.visits))
    return best_child


def expand_node(node, model):
    for action in range(7):
        new_state = node.state + str(action)
        new_node = MCTSNode(new_state, parent=node, action=action)
        node.add_child(new_node)


def simulate(node, model):
    actions = node.state.split('_')[1]
    data = get_next_data(circuitName, actions)
    x = torch.cat([data['node_type'], data['num_inverted_predecessors']], dim=1)
    x = x.reshape(2, -1).T
    x = torch.tensor(x, dtype=torch.float32)
    edge_index = data['edge_index']
    new_data = Data(x=x, edge_index=edge_index)

    with torch.no_grad():
        reward = model(new_data).item()
    return reward


def backpropagate(node, reward):
    while node is not None:
        node.update(reward)
        node = node.parent


def mcts(circuitName, model, iterations=1000, max_steps=10, cur_actions=''):
    root = MCTSNode(circuitName + '_')
    for _ in range(iterations):
        node = root
        depth = 0

        # Selection
        while node.children and depth < max_steps:
            node = select_node(node)
            depth += 1

        # Expansion
        if depth < max_steps:
            expand_node(node, model)

        # Simulation
        reward = simulate(node, model)

        # Backpropagation
        backpropagate(node, reward)

    best_child = max(root.children, key=lambda c: c.reward / c.visits)
    return best_child.state

# Search algorithm
def dfs_search(circuitName, model, max_depth=10, cur_actions=''):
    stack = [(cur_actions, 0)]  # 初始化栈，存储动作序列和当前深度
    best_actions = ''
    best_reward = -float('inf')

    while stack:
        actions, depth = stack.pop()

        if depth == max_depth:
            reward = model_eval(circuitName, actions)
            if reward > best_reward:
                best_actions = actions
                best_reward = reward
            continue

        for action in range(7):  # 7 possible actions
            new_actions = actions + str(action)
            reward = model_eval(circuitName, new_actions)
            if reward > best_reward:
                best_actions = new_actions
                best_reward = reward
            stack.append((new_actions, depth + 1))

    return circuitName + '_' + best_actions

class GNNModel(nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(2, 32)
        self.dropout1 = Dropout(p=0.4)  # 添加Dropout
        self.conv2 = GCNConv(32, 64)
        self.dropout2 = Dropout(p=0.4)  # 添加Dropout
        self.conv3 = GCNConv(64, 32)
        self.dropout3 = Dropout(p=0.4)  # 添加Dropout
        self.fc = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout1(x)  # 应用Dropout
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout2(x)  # 应用Dropout
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.dropout3(x)  # 应用Dropout
        x = torch.mean(x, dim=0)  # global pooling
        x = self.fc(x)
        return x

def main():
    args = set_args()

    # Initialize model, optimizer, and loss function
    model = GNNModel()
    # the model file is on gpu, change it according to your need
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('task1_model_weights.pth', map_location=device))
    model.eval()
    '''
    circuitNames = ['adder','alu2','apex3','apex5','arbiter','b2','c1355','c2670','c5315','c6288',
                    'ctrl','frg1','i7','i8','int2float','log2','m3','max','max512',
                    'multiplier','priority','prom2','table5']
    
    for circuitName in circuitNames:
        print(circuitName)
        if args == 'greedy':
            print(greedy_search(circuitName, model))
        if args == 'beam':
            print(beam_search(circuitName, model))
    
    circuitName = args.initial.split
    '''
    if '_' in args.initial:
        circuitName, cur_actions = args.initial.split('_')
    else:
        circuitName, cur_actions = args.initial, ''
    if args.search == 'greedy':
        print(greedy_search(circuitName, model, cur_actions=cur_actions))
    if args.search == 'beam':
        print(beam_search(circuitName, model, cur_actions=cur_actions))
    
if __name__ == '__main__':
    main()
