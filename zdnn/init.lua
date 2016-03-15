assert( nn, "should require 'nn' first")

local Module = nn.Module
function Module:cancelGradInput()
    self.updateGradInput = function() end
    return self
end
