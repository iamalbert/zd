#!/usr/bin/env th

local test = torch.TestSuite()

local tester = torch.Tester()

require 'nn'
require 'zd'

local target = torch.DoubleTensor{
    1,2,3,4,5,6,7,8,9,10,
    4,7,1,6,5,7,9,3,2,5,
   10,5,8,6,2,3,7,2,7,3,
    7,2,6,5,1,5,2
}

local input = torch.rand(target:size(1), 10)

local batch_size = 8

local inputs = input:split(batch_size)
local targets = target:split(batch_size)


local C = nn.MultiMarginCriterion

local criterion = C(2)
local bcriterion = zdnn.BatchCriterion( C(2), batch_size  )

function test.forward()
    local loss = 0
    for i = 1, #inputs do
        loss = loss + criterion:forward(inputs[i], targets[i])
    end

    local bloss = bcriterion:forward(input, target)

    tester:assertGeneralEq( bloss, loss, 1e-8, "loss is wrong")
end

function test.backward()

    local bgradInput = bcriterion:backward(input, target)
    tester:assertGeneralEq( bgradInput:size(), input:size(), 1e-8, "size is wrong")

    local bgIs = bgradInput:split(batch_size)

    for i = 1, #inputs do
        local gradInput = criterion:backward(inputs[i], targets[i])
        tester:assertGeneralEq(bgIs[i], gradInput, 1e-8, "gradient wrong")
    end

end


-- add more tests by adding to the 'test' table


return tester:add(test):run()
