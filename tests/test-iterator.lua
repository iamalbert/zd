#!/usr/bin/env th

require 'totem'
require 'zd'

local test = {}

local tester = totem.Tester()


local data = torch.rand( 88, 40, 500 )

function test.no_shuffle()
    local iter = zd.Iterator {
        source = data
    }
    iter:reset()
    repeat
        local datum, i = iter:next()
        tester:assertTensorEq(datum, data[i], 1e-7)
    until  iter:finished()
end

function test.no_shuffle_batch()
    local bs = 17
    local iter = zd.Iterator {
        source = data,
        batch = bs
    }
    iter:reset()
    repeat
        local datum, i = iter:next()
        start = (i-1) * bs + 1

        local len = bs
        if start + len - 1 > data:size(1) then
            len = data:size(1) - start + 1
        end
        tester:assertTensorEq(
            datum, 
            data:narrow(1, start, len),
            1e-7)
    until iter:finished()
end


-- add more tests by adding to the 'test' table

return tester:add(test):run()
