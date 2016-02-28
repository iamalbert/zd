local Class, Parent = torch.class('zdnn.FrozenLookupTable', 'nn.Module')

function Class:__init( source, maskzero)
	Parent.__init(self)

	assert( zd.util.isTensor(source), "source must be a Tensor" )
	assert( source:dim() >= 2, "source must have at least 2 dimension")

	self.maskzero = maskzero
	if maskzero then
		self.source = source:sub(1,-1,1,-1):resize( source:size(1)+1, source:size(2) )
		self.source:sub( -1,-1 ):fill(0)
	else
		self.source = source
	end
end


function Class:updateOutput(input)
	if self.maskzero then
		input = input:clone()
		input:maskedFill( input:eq(0), self.source:size(1) )
	end

	input = input:long()

	if input:dim() == 1 then
	    self.output:index(self.source, 1, input)
	elseif input:dim() == 2 then
	    self.output:index(self.source, 1, input:view(-1))
	    self.output = self.output:view(input:size(1), input:size(2), self.source:size(2))
	else
	    error("input must be a vector or matrix")
	end
	return self.output
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
