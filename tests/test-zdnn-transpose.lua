#!/usr/bin/env th

require 'totem'

require 'zd'

local test = totem.TestSuite()

local tester = totem.Tester()


test['Transpose: forward 1'] = function()
    local m = zdnn.Transpose( {1,3} )
    m:evaluate()
    -- [ 1, 2, 3, 4 ] => [3, 2, 1, 4]

    for i = 1,5 do
        local input = torch.rand       ( i  , i+1, i*2, i*3)
        local outSg = torch.LongStorage{ i*2, i+1, i  , i*3}

        local output = m:forward(input)

        tester:assertGeneralEq( output:size(), outSg,
            1e-8, "size is wrong"
        )
    end
end

test['Transpose: forward 2'] = function()
    local m = zdnn.Transpose( {1,3}, {2,-2} )
    m:evaluate()
    -- [1, 2, 3, 4, 5] => [3, 2, 1, 4, 5] => [3, 4, 1, 2, 5]

    for i = 1,5 do
        local input = torch.rand       ( i  , i+1, i*2, i*3, i)
        local outSg = torch.LongStorage{ i*2, i*3, i  , i+1, i}

        local output = m:forward(input)

        tester:assertGeneralEq( output:size(),
            outSg,
            1e-8, "size is wrong"
        )
    end
end


test['Transpose: backward 1'] = function()
    local m = zdnn.Transpose( {1,3} )
    m:evaluate()
    -- [ 1, 2, 3, 4 ] => [3, 2, 1, 4]

    for i = 1,5 do
        local input = torch.rand( i  , i+1, i*2, i*3)
        local gradO = torch.rand( i*2, i+1, i  , i*3)

        local gradI = m:backward(input, gradO)

        tester:assertGeneralEq( 
            gradI:size(), input:size(),
            1e-8, "size is wrong"
        )
    end
end
test['Transpose: backward 2'] = function()
    local m = zdnn.Transpose( {1,3}, {2,-2} )
    m:evaluate()
    -- [1, 2, 3, 4, 5] => [3, 2, 1, 4, 5] => [3, 4, 1, 2, 5]

    for i = 1,5 do
        local input = torch.rand( i  , i+1, i*2, i*3, i)
        local gradO = torch.rand( i*2, i*3, i  , i+1, i)

        local gradI = m:backward(input, gradO)

        tester:assertGeneralEq( 
            gradI:size(), input:size(),
            1e-8, "size is wrong"
        )
    end
end



-- add more tests by adding to the 'test' table


return tester:add(test):run()
