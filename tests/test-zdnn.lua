#!/usr/bin/env th

require 'totem'
require 'nn'
require 'zd'

local test = {}

local tester = totem.Tester()


function test.FrozenLookupTable()
    local dim = 40
    local db = torch.rand( 30, dim )
    
    local indices_list = {
        {1,21,30,24,16,2,1,9,10}, 
        {25,12,7,24,5,8,6,19,15,1},
        {6,2,11,30,29,3,14,8,8,23,25,17,15,19,25,14,3,4,17,14,20,23,6,18,16,24,10,10,19,30,6,18,27,24,16,20,5,2,4,5,14,19,23,20,16,18,4,6,13,5,24,18,10,30,16,16,11,16,18,17,3,29,21,26,10,15,13,8,4,9,19,18,3,16,30,11,22,14,14,22,11,3,4,5,3,2,3,26,5,5,21,18,25,21,19,26,22,11,16,15},
    }

    local model = zdnn.FrozenLookupTable(db)

    for _, indices in ipairs(indices_list) do
        local out = model:forward( torch.LongTensor(indices) )

        tester:assertTableEq( 
            out:size():totable(), 
            { #indices, dim },
            1e-20,
            "output tensor size mismatch with input"
        )
        for i = 1,#indices do
            tester:assertTensorEq(
                out[i],
                db[ indices[i] ],
                1e-7,
                "output tensor error"
            )
        end
    end
    -- add test code here, using tester:asserteq methods
end




-- add more tests by adding to the 'test' table


return tester:add(test):run()
