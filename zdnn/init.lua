assert( nn, "should require 'nn' first")
do
    local noop = function() end

    local Module = torch.getmetatable('nn.Module')
    function Module:cancelGradInput()
        self.updateGradInput = noop
        return self
    end

    Module.__unm__ = function( obj )
        return obj()
    end

    Module.__sub__ = function( prev, next )
        return next(prev)
    end

    function Module:forwardbackward(input, gradOutput)
        local output = self:forward(input)
        local gradInput = self:backward(input, gradOutput)
        return output, gradInput
    end
end

assert( nngraph, "should require 'nngraph' first")
do
    local Node = torch.getmetatable('nngraph.Node')
    Node.__sub__ = function( prev, next )
        return next(prev)
    end
end
