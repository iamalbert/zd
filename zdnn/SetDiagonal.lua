local Class, Parent = torch.class('zdnn.SetDiagonal', 'nn.Module')


function Class:__init(value)
    Parent.__init(self)
    self.value = value
end

function makeMask(input)
    return torch.ByteTensor():eye(input:size(1))
end

function Class:updateOutput(input)
    local mask = makeMask(input)

    self.output = input:clone()
    self.output:maskedFill(mask, self.value)

    return self.output
end


function Class:updateGradInput(input, gradOutput)
    local mask = makeMask(input)
    self.gradInput = gradOutput:clone()
    self.gradInput:maskedFill(mask, 0)
    return self.gradInput
end

