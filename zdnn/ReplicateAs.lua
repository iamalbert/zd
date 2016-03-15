local Class,Parent = torch.class('zdnn.ReplicateAs', 'nn.Module')

local unpack = unpack or table.unpack

function Class:__init()
	Parent.__init(self)
end

function Class:updateOutput(input)
    local source, n = input[1], input[2]:size(1)
    self.output = torch.expand(source, n, source:size(2) )
    return self.output
end

function Class:updateGradInput(input, gradOutput)
    self.gradInput = { gradOutput:sum(1), input[2]:clone():zero() }
    return self.gradInput
end

function Class:__tostring()
	return torch.type(self) 
end
