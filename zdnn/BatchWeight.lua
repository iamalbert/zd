
local Class,Parent = torch.class('zdnn.BatchWeight', 'nn.Sequential')

function Class:__init( dim, bias )
	Parent.__init(self)

	self.dim = dim

	local module = nn.Linear(dim,1,bias)
	self :add( module )              -- N x 1
		 :add(nn.View(-1))           -- N
		 :add(nn.SoftMax())          -- N
		 :add(nn.View(-1,1))         -- N x 1
end

function Class:__tostring()
	return torch.type(self) .. string.format( "( [n,%d] -> [n,1] )",
		self.dim
	)
end
