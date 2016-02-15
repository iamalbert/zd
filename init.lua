zd = {
    version = 0.1
}

local modules = {
    'util', 

    'FSM',
    'Tree',
}

for _, file in ipairs( modules ) do
    torch.include('zd', file .. '.lua')
end
