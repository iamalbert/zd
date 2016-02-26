zd = {
    __version = 0.1
}

zdnn = {
    __version = 0.1
}

require 'nn'

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

    'zdnn/FrozenLookupTable',
    'zdnn/AdaptiveWeight',
}

for _, file in ipairs( modules ) do
	torch.include('zd', file.. '.lua')
end
