local Class, Parent = torch.class('zdnn.Sequencer', 'nn.Module')

function Class:__init( module, nInputDim )
    Parent.__init(self)
    self.module = module

    self.seq = nn.Sequential()
        :add( nn.SplitTable(1) )
        :add( nn.Sequencer(module) )
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
        assert(i:eq(i):all())
        local o = self.seq:forward(i)
        assert(o:eq(o):all())
        self.output = o:squeeze(2)
        assert(self.output:eq(self.output):all())
    else
        self.output = self.seq:forward(input)
    end

    return self.output
end

function Class:updateGradInput(input, gradOutput)
    local dim = input:dim()
    assert( dim == 2 or dim == 3, 
        " input size must be [Time x Batch x Vector] or [Time x Vector]," ..
        " given [" .. table.concat(input:size():totable(),'x') .. "]"
    )
    
    if dim == 2 then
        local i  = input:view( input:size(1), 1, input:size(2) )
        local go = gradOutput:view(gradOutput:size(1), 1, gradOutput:size(2) )

        local gI = self.seq:updateGradInput(i, go)
        self.gradInput = gI:squeeze(2)
    else
        self.gradInput = self.seq:updateGradInput(input, gradOutput)
    end

    return self.gradInput
end

function Class:accGradParameters(input, gradOutput)
    local dim = input:dim()
    assert( dim == 2 or dim == 3, 
        " input size must be [Time x Batch x Vector] or [Time x Vector]," ..
        " given [" .. table.concat(input:size():totable(),'x') .. "]"
    )
    
    if dim == 2 then
        local i  = input:view(input:size(1), 1, input:size(2) )
        local go = gradOutput:view(gradOutput:size(1), 1, gradOutput:size(2) )
        self.seq:accGradParameters(i, go)
    else
        self.seq:accGradParameters(input, gradOutput)
    end

    return self.gradInput
end

function Class:__tostring()
	return torch.type(self) .. " @ " .. tostring(self.module)
end
