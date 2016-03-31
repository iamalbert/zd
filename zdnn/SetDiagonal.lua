local Class, Parent = torch.class('zdnn.SetDiagonal', 'nn.Module')


function Class:__init(value)
    Parent.__init(self)
    self.value = value
end

function Class:makeMask(input)
    if input:type() == 'torch.CudaTensor' then
        self.mask = self.mask or torch.CudaTensor()
    else
        self.mask = self.mask or torch.ByteTensor()
    end
    return torch.eye(self.mask, input:size(1) )
end

function Class:updateOutput(input)
    local mask = self:makeMask(input)

    self.output = input:clone()
    self.output:maskedFill(mask, self.value)

    return self.output
end


function Class:updateGradInput(input, gradOutput)
    local mask = self:makeMask(input)
    self.gradInput = gradOutput:clone()
    self.gradInput:maskedFill(mask, 0)
    return self.gradInput
end

