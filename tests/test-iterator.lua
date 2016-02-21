#!/usr/bin/env th

require 'totem'
require 'zd'

local test = {}

local tester = totem.Tester()

local inputs  = torch.rand( 883, 40, 500 )
local targets = torch.rand( 883, 1)

function test.no_shuffle()
    local iter = zd.Iterator {
        source = {
            input = inputs,
            target = targets,
        }
    }
    iter:reset()
    repeat
        local datum, i = iter:next()
        tester:assertTensorEq(datum.input, inputs[i], 1e-7,
            "entry.input not equal inputs[i]")
    until  iter:finished()
end

function test.no_shuffle_batch()
    local bs = 17
    local iter = zd.Iterator {
        source = {
            input = inputs,
            target = targets
        },
        batch = bs
    }
    iter:reset()
    repeat
        local datum, i = iter:next()
        start = (i-1) * bs + 1

        local len = bs
        if start + len - 1 > inputs:size(1) then
            len = inputs:size(1) - start + 1
        end
        tester:assertTensorEq(
            datum.input, 
            inputs:narrow(1, start, len),
            1e-7)
    until iter:finished()
end


-- add more tests by adding to the 'test' table

return tester:add(test):run()
