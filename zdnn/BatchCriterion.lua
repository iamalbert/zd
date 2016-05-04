local Class, Parent = torch.class('zdnn.BatchCriterion', 'nn.Criterion')

function Class:__init(crit, batch_size, to_shuffle)
    Parent.__init(self)

    assert( crit, "must provide an criterion")
    assert( batch_size, "must provide batch size")
    self.criterion  = crit
    self.batch_size = batch_size
    self.to_shuffle = not not to_shuffle or false
end

function Class:updateOutput(input, target)

    assert( torch.isTensor(input), "input must be a tensor")
    assert( torch.isTensor(target), "target must be a tensor")

    self.output = 0
    if self.to_shuffle then
        error "not implemented"
    else
        local inputs  = input:split(self.batch_size)
        local targets = target:split(self.batch_size)

        for i = 1, #inputs do
            local Input  = inputs[i]
            local Target = targets[i]
            --print(Input, Target)

            local loss = self.criterion:forward(Input, Target)
            self.output = self.output + loss
        end
    end
    return self.output
end

function Class:updateGradInput(input, target)
    assert( torch.isTensor(input), "input must be a tensor")
    assert( torch.isTensor(target), "target must be a tensor")

    self.gradInput = self.gradInput or input.new()
    self.gradInput:typeAs(input):resizeAs(input):zero()

    if self.to_shuffle then
        error "not implemented"
    else

        local inputs = input:split(self.batch_size)
        local targets = target:split(self.batch_size)
        local gradInputs = self.gradInput:split(self.batch_size)

        for i = 1, #inputs do
            local Input  = inputs[i]
            local Target = targets[i]

            local gradInput = self.criterion:backward(Input, Target)

            gradInputs[i]:copy(gradInput)
        end
    end
    return self.gradInput
end
