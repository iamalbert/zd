local Class, Parent = torch.class('zdnn.BiSequencer', 'nn.Sequential')

function Class:__init( module, module_rev, nInputDim )
    Parent.__init(self)
    self.module = module
    if module_rev then
        self.module_rev = module_rev
    else
        self.module_rev = module:clone():reset()
    end

    self:add( nn.SplitTable(1) )
        :add( nn.BiSequencer(self.module, self.module_rev) )
        :add( nn.Sequencer(nn.Unsqueeze(1)) )
        :add( nn.JoinTable(1) )
end


function Class:updateOutput(input)
    local dim = input:dim()
    assert( dim == 2 or dim == 3, 
        " input size must be [Time x Batch x Vector] or [Time x Vector]," ..
        " given [" .. table.concat(input:size():totable(),'x') .. "]"
    )

    
    if dim == 2 then
        local i = input:view( input:size(1), 1, input:size(2) )
        local o = Parent.updateOutput(self, i)
        self.output = o:squeeze(2)
    else
        self.output = Parent.updateOutput(self, input)
    end

    return self.output
end

-- Wrapper is not necessary in backward process, I don't know why but
-- it works fine, look at tests/test-zdnn-sequencer-back.lua
-- maybe nn.Sequencer/nn.AbstractRecurrent do handle different size?

--[[
function Class:updateGradInput(input, gradOutput)
    local dim = input:dim()
    assert( dim == 2 or dim == 3, 
        " input size must be [Time x Batch x Vector] or [Time x Vector]," ..
        " given [" .. table.concat(input:size():totable(),'x') .. "]"
    )
    
    if dim == 2 then
        local i  = input:view( input:size(1), 1, input:size(2) )
        local go = gradOutput:view(gradOutput:size(1), 1, gradOutput:size(2) )

        local gI = Parent.updateGradInput(self, i, go)
        self.gradInput = gI:squeeze(2)
    else
        self.gradInput = Parent.updateGradInput(self, input, gradOutput)
    end

    return self.gradInput
end

function Class:accGradParameters(input, gradOutput, scale)
    local dim = input:dim()
    assert( dim == 2 or dim == 3, 
        " input size must be [Time x Batch x Vector] or [Time x Vector]," ..
        " given [" .. table.concat(input:size():totable(),'x') .. "]"
    )
    
    if dim == 2 then
        local i  = input:view(input:size(1), 1, input:size(2) )
        local go = gradOutput:view(gradOutput:size(1), 1, gradOutput:size(2) )
        return Parent.accGradParameters(self, i, go, scale)
    else
        return Parent.accGradParameters(self, gradOutput, scale)
    end
end

function Class:backward(input, gradOutput, scale)
    local dim = input:dim()
    assert( dim == 2 or dim == 3, 
        " input size must be [Time x Batch x Vector] or [Time x Vector]," ..
        " given [" .. table.concat(input:size():totable(),'x') .. "]"
    )
    
    if dim == 2 then
        local i  = input:view(input:size(1), 1, input:size(2) )
        local go = gradOutput:view(gradOutput:size(1), 1, gradOutput:size(2) )
        return Parent.backward(self, i, go, scale):resizeAs(input)
    else
        return Parent.backward(self, gradOutput, scale)
    end
end
--]]

function Class:__tostring()
	return torch.type(self) .. " @ " .. tostring(self.module)
end
