local Class,Parent = torch.class('zdnn.Slice', 'nn.Module')

local unpack = unpack or table.unpack

function Class:__init(...)
	Parent.__init(self)
    self.indices = {...}
end

function Class:updateOutput(input)
    self.output = input:sub( unpack(self.indices) )
    return self.output
end

function Class:updateGradInput(input, gradOutput)
    self.gradInput = input:clone()
    self.gradInput:sub( unpack(self.indices) ):copy(gradOutput)
    return self.gradInput
end

function Class:__tostring()
	return torch.type(self) ..
		string.format( "(%s)", table.concat(self.indices, ", ") )
end
