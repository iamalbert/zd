require 'nn'

local Class, Parent = torch.class('zdnn.FrozenLookupTable', 'nn.Module')

function Class:__init( source, out_policy)
	Parent.__init(self)

	assert( zd.util.isTensor(source), "source must be a Tensor" )
	assert( source:dim() >= 2, "source must have at least 2 dimension")

	self.source = source
	self.policy = out_policy
end

function Class:updateOutput(input)
	self.output = self.source:index( 1, input )
	return self.output
end

function Class:updateGradInput(input,gradInput)
end

function Class:updateGradParameters(input, gradInput)
end

function Class:__tostring()
	local sz = self.source:size():totable()
	local insz = table.remove( sz, 1 )
	return torch.type(self) .. 
		string.format("( [%d] -> %s )", 
			insz,
			table.concat(sz, " x ")
		)
end
