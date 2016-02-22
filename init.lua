zd = {
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

    'FSM',
    'Tree',
    'TreeNN'
}

for _, file in ipairs( modules ) do
	torch.include('zd', file.. '.lua')
end
