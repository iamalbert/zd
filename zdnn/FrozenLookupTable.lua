require 'nn'

local Class, Parent = torch.class('zdnn.FrozenLookupTable', 'nn.LookupTable')

function Class:__init( source, out_policy)
	nn.Module.__init(self)

	assert( zd.util.isTensor(source), "source must be a Tensor" )
	assert( source:dim() >= 2, "source must have at least 2 dimension")

	self.weight = source:clone()
	-- self.gradWeight = torch.Tensor( source:size(1), source:size(2) )
	self.paddingValue =  0
end

function Class:reset()
end

function Class:accGradParameters()
end

function Class:__tostring()
	local sz = self.weight:size():totable()
	local insz = table.remove( sz, 1 )
	return torch.type(self) .. 
		string.format("( [%d] -> %s )", 
			insz,
			table.concat(sz, " x ")
		)
end
