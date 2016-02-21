zd = {
    version = 0.1
}

require 'nn'

local modules = {
    'util', 

    'Data',
    'Evaluator',

    'FSM',
    'Tree',
    'TreeNN'
}

for _, file in ipairs( modules ) do
	torch.include('zd', file.. '.lua')
end
