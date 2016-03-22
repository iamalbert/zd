local Class, Parent = torch.class('zdnn.GeneralLinear', 'nn.Linear')

function Class:updateOutput(input)
    local nDim = input:dim()
    if nDim == 1 or nDim == 2 then
        return Parent.updateOutput(self, input)
    elseif nDim > 2 then
        local inDim = input:size( nDim )

        local transInput = input:view(-1, inDim)
        Parent.updateOutput(self, transInput)

        local outSize = input:size()
        outSize[ nDim ] = self.weight:size(1)

        self.output:resize( outSize )

        return self.output
    else
        error "input must have dimension > 0"
    end
end

function Class:updateGradInput(input, gradOutput)
    local nDim = input:dim()
    if nDim == 1 or nDim == 2 then
        return Parent.updateGradInput(self, input, gradOutput)
    elseif nDim > 2 then
        local inDim = input:size( nDim )
        local outDim = self.weight:size(1)

        local transInput = input:view(-1, inDim)
        local transGradOutput = gradOutput:view(-1, outDim)

        local transGradInput = Parent.updateGradInput(
            self, transInput, transGradOutput
        )

        self.gradInput:resizeAs(input)

        return self.gradInput
    else
        error "input must have dimension > 0"
    end
end

function Class:accGradParameters(input, gradOutput, scale)
    local nDim = input:dim()
    if nDim == 1 or nDim == 2 then
        return Parent.accGradParameters(self, input, gradOutput, scale)
    elseif nDim > 2 then
        local inDim = input:size( nDim )
        local outDim = self.weight:size(1)

        local transInput = input:view(-1, inDim)
        local transGradOutput = gradOutput:view(-1, outDim)

        Parent.accGradParameters(
            self, transInput, transGradOutput, scale
        )

    else
        error "input must have dimension > 0"
    end
end
