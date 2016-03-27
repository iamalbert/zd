require 'nn'
require 'rnn'
require 'optim'
require 'nngraph'

zd = {
    __version = 0.1
}

zdnn = {
    __version = 0.1
}

local modules = {
    'util', 

    'optims/util',
    'optims/rmsprop',
    'optims/adagrad',

    'Data',
    'Evaluator',
    'Trainer',
    'Expr',

    'FSM',
    'Tree',
    'TreeNN',

    'zdnn/init',
    'zdnn/FrozenLookupTable',
    'zdnn/BatchWeight',
    'zdnn/JoinTableFixed',
    'zdnn/FilterTarget',
    'zdnn/Slice',
    'zdnn/ReplicateAs',
    'zdnn/Sequencer',
    'zdnn/BiSequencer',
    'zdnn/GeneralLinear',
    'zdnn/Transpose',
    'zdnn/PairwiseReplicate',
    'zdnn/SetDiagonal',
}

for _, file in ipairs( modules ) do
	torch.include('zd', file.. '.lua')
end
