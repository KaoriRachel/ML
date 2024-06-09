# Quick Start
process decision making
```
python3 decision.py --initial alu_34
```
# Introduction
- Initial AIG: store the initial 23 aig files
  - train: useful initial aig files
  - test: unused in this project
- aig: store the aig files generated in the data geting process
- log: store the log files generated in the data geting precess
- project\_data: the training data of task1
- project\_data2: the training data of task2
- lib: lib file used in the project
- get\_data.py: Read the pkl data, generate the aig file, and transform it to GNN input
- train.py: use the data from get\_data.py to train the GNN model
- decision.py: to decision the next steps of logic synthesis
  - --search: the search way, greedy or beam
  - --initial: the initial state to predict, e.g.alu2\_532
- task1\_model\_weights: the parameters of model in task1
- task2\_model\_weights: the parameters of model in task2
