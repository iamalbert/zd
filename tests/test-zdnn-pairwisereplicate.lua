#!/usr/bin/env th

require 'totem'

require 'zd'
local test = totem.TestSuite()

local tester = totem.Tester()


test['PairwiseReplicate: forward'] = function()
    local A = torch.Tensor {
        {1,2,3,4},
        {5,6,7,8}
    }
    local B = torch.Tensor {
        {11,13,15,17},
        {12,14,16,18} 
    }
    local C1 = torch.Tensor(2,2,4)
    local C2 = torch.Tensor(2,2,4)

    for i=1,2 do
        for j=1,2 do
            for k=1,4 do
                C1[{i,j,k}] = A[{i,k}]
                C2[{i,j,k}] = B[{j,k}]
            end
        end
    end

    local m = zdnn.PairwiseReplicate()
    m:evaluate()

    local pred = m:forward {A,B}
    tester:assertGeneralEq( pred[1], C1, 1e-8, "out incorrect")
    tester:assertGeneralEq( pred[2], C2, 1e-8, "out incorrect")
    return m, {A,B},{C1,C2}
end

test['PairwiseReplicate: backward'] = function()
    local m, input, target = test['PairwiseReplicate: forward']()

    local gradOutput = {
        torch.rand(2,2,4),
        torch.rand(2,2,4)
    }
    local targetGradInput = {
    }
    m:training()
    local gradInput = m:backward(input, gradOutput)

    tester:assertGeneralEq( gradInput[1]:size(), input[1]:size(),
        1e-8, "size A mismatch")
    tester:assertGeneralEq( gradInput[2]:size(), input[2]:size(),
        1e-8, "size B mismatch")
end

-- add more tests by adding to the 'test' table


return tester:add(test):run()
