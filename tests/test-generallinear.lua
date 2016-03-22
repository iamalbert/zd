#!/usr/bin/env th

require 'totem'

require 'zd'

local test = totem.TestSuite()

local tester = totem.Tester()

local inDim, outDim = 10, 30

local lin = nn.Linear(inDim, outDim)
local glin = zdnn.GeneralLinear(inDim, outDim)

glin.weight:copy( lin.weight )
glin.bias:copy( lin.bias)

test['GeneralLinear: forward'] = function ()
    lin:evaluate()
    glin:evaluate()
    for i=1,10 do
        local input = torch.rand(inDim)

        local outlin = lin:forward(input)
        local outglin = lin:forward(input)

        tester:assertTensorEq(outglin, outlin, 1e-8, "forward is not equal")
    end
end

test['GeneralLinear: forward (batch)'] = function ()
    lin:evaluate()
    glin:evaluate()
    for i=1,10 do
        local input = torch.rand(i, inDim)

        local outlin = lin:forward(input)
        local outglin = lin:forward(input)

        tester:assertTensorEq(outglin, outlin, 1e-8, "forward is not equal")
    end
end


test['GeneralLinear: forward (multi-dim batch)'] = function ()
    lin:evaluate()
    glin:evaluate()
    for i=1,10 do
        local input = torch.rand(i, i*2, i*3, inDim)

        local outStorage = input:size()
        outStorage[ input:dim() ] = outDim

        local outglin = glin:forward(input)

        tester:assertGeneralEq(outglin:size(), outStorage, 1e-8, 
            "size incorrect\n" 
                .. table.concat(outStorage:totable(), 'x') .. "\n"
                .. table.concat(outglin:size():totable(), 'x') 
        )

        local outlin  = lin:forward(input:view(-1, inDim)):view( outStorage )

        tester:assertTensorEq(outglin, outlin, 1e-8, "forward is not equal")
    end
end

test['GeneralLinear: backward'] = function ()
    lin:training()
    glin:training()

    for i=1,10 do
        local input = torch.rand(inDim)
        local gradOutput = torch.rand(outDim)

        lin:zeroGradParameters()
        glin:zeroGradParameters()

        local gI_lin = lin:backward(input, gradOutput)
        local gI_glin = glin:backward(input, gradOutput)

        tester:assertTensorEq(
            gI_glin, gI_lin, 1e-8, "gradInput is not equal")

        tester:assertTensorEq(
            glin.gradWeight,glin.gradWeight, 1e-8, "gradWeight unequal"
        )
        tester:assertTensorEq(
            glin.gradBias,glin.gradBias, 1e-8, "gradBias unequal"
        )
    end
end

test['GeneralLinear: backward (batch)'] = function ()
    lin:training()
    glin:training()

    for i=1,10 do
        local input = torch.rand(i, inDim)
        local gradOutput = torch.rand(i, outDim)

        lin:zeroGradParameters()
        glin:zeroGradParameters()

        local gI_lin = lin:backward(input, gradOutput)
        local gI_glin = glin:backward(input, gradOutput)

        tester:assertTensorEq(
            gI_glin, gI_lin, 1e-8, "gradInput is not equal")

        tester:assertTensorEq(
            glin.gradWeight,glin.gradWeight, 1e-8, "gradWeight unequal"
        )
        tester:assertTensorEq(
            glin.gradBias,glin.gradBias, 1e-8, "gradBias unequal"
        )
    end
end
test['GeneralLinear: backward (multi-dim batch)'] = function ()
    lin:training()
    glin:training()

    for i=1,10 do
        local input = torch.rand(i, i*2, i*3, inDim)
        local gradOutput = torch.rand(i, i*2, i*3, outDim)

        local input_flat = input:view(-1, inDim)
        local gradOutput_flat = gradOutput:view(-1, outDim)

        lin:zeroGradParameters()
        glin:zeroGradParameters()

        local gI_lin = lin:backward(input_flat, gradOutput_flat)
        local gI_glin = glin:backward(input, gradOutput)

        tester:assertTensorEq(
            gI_glin:view(-1, inDim), gI_lin, 1e-8, "gradInput is not equal")

        tester:assertTensorEq(
            glin.gradWeight,glin.gradWeight, 1e-8, "gradWeight unequal"
        )
        tester:assertTensorEq(
            glin.gradBias,glin.gradBias, 1e-8, "gradBias unequal"
        )
    end
end


return tester:add(test):run()
