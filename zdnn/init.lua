assert( nn, "should require 'nn' first")

do
    local Module = torch.getmetatable('nn.Module')
    function Module:cancelGradInput()
        self.updateGradInput = function() end
        return self
    end
end

assert( nngraph, "should require 'nngraph' first")
do
    local Module = torch.getmetatable('nngraph.Node')
    Module.__sub = function( prev, next )
        return next(prev)
    end
end
