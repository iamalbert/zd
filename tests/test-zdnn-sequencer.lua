#!/usr/bin/env th

require 'zd'


local test = torch.TestSuite()

local tester = torch.Tester()


local m = zdnn.Sequencer( nn.FastLSTM(3,5) )

test["Sequencer: 1-batch"] = function()
    local out = m:forward( torch.rand(4, 1, 3) )
    tester:assertTableEq(
        out:size():totable(), {4, 1, 5},
        1e-8, "output size mismatch " .. 
            table.concat(out:size():totable(), 'x')
    )

    local gI = m:backward( torch.rand(4,1,3), torch.rand(4,1,5) )
    tester:assertTableEq(
        gI:size():totable(), {4, 1, 3},
        1e-8, "gradInput size mismatch " .. 
            table.concat(gI:size():totable(), 'x')
    )
end

test["Sequencer: n-batch"] = function()
    local out = m:forward( torch.rand(4, 7, 3) )
    tester:assertTableEq(
        out:size():totable(), {4, 7, 5},
        1e-8, "output size mismatch " .. table.concat(out:size():totable(), 'x')
    )

    local gI = m:backward( torch.rand(4,7,3), torch.rand(4,7,5) )
    tester:assertTableEq(
        gI:size():totable(), {4, 7, 3},
        1e-8, "gradInput size mismatch " .. 
            table.concat(gI:size():totable(), 'x')
    )
end

test["Sequencer: error input"] = function()
    local input  = torch.rand(4)
    tester:assertError( function()
        m:forward( input )
    end)

    local input  = torch.rand(4,54,1,1,2,4)
    tester:assertError( function()
        m:forward( input )
    end)
end


test["Sequencer: no-batch"] = function ()
    local input  = torch.rand(4, 3)
    local output = m:forward( input )

    tester:assertTableEq(
        output:size():totable(), {4, 5},
        1e-8, "output size mismatch " .. 
            table.concat(output:size():totable(), 'x')
    )
    local gI = m:backward( torch.rand(4,3), torch.rand(4,5) )
    tester:assertTableEq(
        gI:size():totable(), {4, 3},
        1e-8, "gradInput size mismatch " .. 
            table.concat(gI:size():totable(), 'x')
    )
end


--]]


return tester:add(test):run()
