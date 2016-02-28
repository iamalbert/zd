local Class,Parent = torch.class('zdnn.BatchWeight', 'nn.Sequential')

function Class:__init(dim, n_layers)
	Parent.__init(self)

	self.dim = dim

	n_layers = n_layers or 0

	for i=1,n_layers do
		self:add( nn.Linear(dim,dim) ):add( nn.Tanh() )
	end

	self.mod = nn.Linear(dim,1, false)

							        -- N x dim [input]
	self:add( self.mod )            -- N x 1

	self :add(nn.View(-1))       -- N
		 :add(nn.SoftMax())      -- N
		 :add(nn.View(-1,1))     -- N x 1
end


function Class:__tostring()
	return torch.type(self) ..
		string.format( "( [n,%d] -> [n,1] )", self.dim )
end
