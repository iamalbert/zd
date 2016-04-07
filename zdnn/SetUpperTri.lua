local Class, Parent = torch.class('zdnn.SetUpperTri', 'nn.Module')

function Class:__init(value, k)
    Parent.__init(self)
    self.value = value
    self.k     = k or 0
end

function Class:makeMask(input)
    local mask = torch.triu(torch.ones(input:size()), self.k)
    if input:type() == 'torch.CudaTensor' then
        mask = mask:cuda()
    else
        mask = mask:byte()
    end
    return mask
end

function Class:updateOutput(input)
    local mask = self:makeMask(input)
    self.output = input:clone()
    self.output:maskedFill( mask, self.value )
    return self.output
end
function Class:updateGradInput(input, gradOutput)
    local mask = self:makeMask(input)
    self.gradInput = gradOutput:clone()
    self.gradInput:maskedFill( mask, 0 )
    return self.gradInput
end
function Class:__tostring()
    return torch.type(self) .. string.format("(%d,%d)", self.value, self.k)
end
