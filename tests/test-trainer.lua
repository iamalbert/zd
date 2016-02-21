#!/usr/bin/env th

require 'nn'
require 'totem'

require 'zd'

local test = {}

local tester = totem.Tester()


function test.TrainPropogate()
    -- add test code here, using tester:asserteq methods
end


-- add more tests by adding to the 'test' table


return tester:add(test):run()
