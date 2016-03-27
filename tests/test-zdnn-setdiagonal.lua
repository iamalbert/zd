#!/usr/bin/env th

require 'totem'
require 'zd'

local test = totem.TestSuite()

local tester = totem.Tester()

local value = 20.9098
local m = zdnn.SetDiagonal(value)

test['SetDiagonal: forward'] = function()
    m:evaluate()
    for i = 1,10 do
        local dim = i * i 
        local input = torch.rand(dim,dim)

        local target = input:clone()
        for j=1,dim do target[ {j,j} ] = value end

        local pred = m:forward(input)

        tester:assertGeneralEq( pred, target,
            1e-7, "prediction incorrect")
    end
end


test['SetDiagonal: backward'] = function()
    m:training()
    for i = 1,10 do
        local dim = i * i 

        local input = torch.rand(dim,dim)
        local gradOutput = torch.rand(dim,dim)

        local target = gradOutput:clone()
        for j=1,dim do target[ {j,j} ] = 0 end


        local gradInput = m:backward( input, gradOutput )

        tester:assertGeneralEq(gradInput, target,
            1e-8, "gradInput incorrect")

    end
end
-- add more tests by adding to the 'test' table


return tester:add(test):run()
