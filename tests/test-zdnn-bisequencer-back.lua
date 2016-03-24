#!/usr/bin/env th

require 'zd'
require 'totem'

local test = totem.TestSuite()

local tester = totem.Tester()

local inDim, outDim = 300, 100
local inDim, outDim = 10, 3 
local seqLen = 3

local f_rec, b_rec = nn.GRU(inDim, outDim), nn.FastLSTM(inDim,outDim)

local zm = zdnn.BiSequencer(f_rec, b_rec)
local rm = nn.BiSequencer(f_rec:clone(), b_rec:clone() )

local _, zgp = zm:getParameters()
local _, rgp = rm:getParameters()

local split = nn.SplitTable(1)

--[[
test["Sequencer:backward: 1-batch"] = function()

    local input = torch.rand(seqLen, 1, inDim)
    local input_table = split:forward( input )

    local target = torch.rand(seqLen, 1, outDim*2)
    local target_table = split:forward( target )


    tester:assertGeneralEq( #input_table, input:size(1),
        1e-8, "split length incorrect")
    tester:assertGeneralEq( input_table[1], input[1],
        1e-8, "split incorrect")

    tester:assertGeneralEq( #input_table, input:size(1),
        1e-8, "split length incorrect")
    tester:assertGeneralEq( input_table[1], input[1],
        1e-8, "split incorrect")

    for _, module in pairs{zm,rm} do
        module:training()
        module:zeroGradParameters()
        module:forget()
    end

    local pred = zm:forward( input )
    local pred_table = rm:forward( input_table )

    tester:assertGeneralEq( 
        pred:size(), target:size(),
        1e-8, "size should be " .. seqLen .. " x 1 x" .. outDim*2
    )

    for i=1,seqLen do
        tester:assertGeneralEq(
            pred[i], pred_table[i],
            1e-8, "step " .. i .. " is different"
        )
    end

    local gradInput = zm:backward( input, target )
    local gradInput_table = rm:backward( input_table, target_table )

    tester:assertGeneralEq( 
        gradInput:size(), input:size(),
        1e-8, 
        "gradInput size should be " .. seqLen .. " x 1 x" .. inDim
    )

    for i=1,seqLen do
        tester:assertGeneralEq(
            gradInput[i], gradInput_table[i],
            1e-8, "gradInput step " .. i .. " is different"
        )
    end

    tester:assertGeneralEq( zgp, rgp, 1e-8, "gradParam not equal")

end
test["Sequencer:backward: n-batch"] = function()

    local bs = 12
    
    local input = torch.rand(seqLen, bs, inDim)
    local input_table = split:forward( input )

    local target = torch.rand(seqLen, bs, outDim*2)
    local target_table = split:forward( target )


    tester:assertGeneralEq( #input_table, input:size(1),
        1e-8, "split length incorrect")
    tester:assertGeneralEq( input_table[1], input[1],
        1e-8, "split incorrect")

    tester:assertGeneralEq( #input_table, input:size(1),
        1e-8, "split length incorrect")
    tester:assertGeneralEq( input_table[1], input[1],
        1e-8, "split incorrect")

    for _, module in pairs{zm,rm} do
        module:training()
        module:zeroGradParameters()
        module:forget()
    end

    local pred = zm:forward( input )
    local pred_table = rm:forward( input_table )

    tester:assertGeneralEq( 
        pred:size(), target:size(),
        1e-8, 
        "size should be " .. seqLen .. "x" .. bs .."x" .. outDim *2 .. 
        ", got " .. table.concat( pred:size():totable(), 'x')
    )

    for i=1,seqLen do
        tester:assertGeneralEq(
            pred[i], pred_table[i],
            1e-8, "step " .. i .. " is different"
        )
    end

    local gradInput = zm:backward( input, target )
    local gradInput_table = rm:backward( input_table, target_table )

    tester:assertGeneralEq( 
        gradInput:size(), input:size(),
        1e-8, 
        "gradInput size should be " .. seqLen .. "x" .. bs .. "x" .. inDim
    )

    for i=1,seqLen do
        tester:assertGeneralEq(
            gradInput[i], gradInput_table[i],
            1e-8, "gradInput step " .. i .. " is different"
        )
    end
    tester:assertGeneralEq( zgp, rgp, 1e-8, "gradParam not equal")
end


test["Sequencer:backward: error input"] = function()
    local input  = torch.rand(4)
    tester:assertError( function()
        m:forward( input )
    end)

    local input  = torch.rand(4,54,1,1,2,4)
    tester:assertError( function()
        m:forward( input )
    end)
end
--]]

test["Sequencer:backward: no-batch"] = function ()
    local input = torch.rand(seqLen, inDim)
    local input_table = split:forward( input )

    local target = torch.rand(seqLen, outDim*2)
    local target_table = split:forward( target )

    tester:assertGeneralEq( #input_table, input:size(1),
        1e-8, "split length incorrect")
    tester:assertGeneralEq( input_table[1], input[1],
        1e-8, "split incorrect")

    tester:assertGeneralEq( #input_table, input:size(1),
        1e-8, "split length incorrect")
    tester:assertGeneralEq( input_table[1], input[1],
        1e-8, "split incorrect")

    for _, module in pairs{zm,rm} do
        module:training()
        module:zeroGradParameters()
        module:forget()
    end

    local pred = zm:forward( input )
    local pred_table = rm:forward( input_table )

    tester:assertGeneralEq( 
        pred:size(), target:size(),
        1e-8, "size should be " .. seqLen .. " x " .. outDim*2
    )

    for i=1,seqLen do
        print( pred[i], pred_table[i] )
        tester:assertGeneralEq(
            pred[i], pred_table[i],
            1e-8, "step " .. i .. " is different"
        )
    end

    local gradInput = zm:backward( input, target )
    local gradInput_table = rm:backward( input_table, target_table )

    tester:assertGeneralEq( 
        gradInput:size(), input:size(),
        1e-8, "gradInput size should be " .. seqLen .. " x " .. inDim
    )

    for i=1,seqLen do
        -- print( gradInput[i], gradInput_table[i] )
        tester:assertGeneralEq(
            gradInput[i], gradInput_table[i],
            1e-8, "gradInput step " .. i .. " is different"
        )
    end
    tester:assertGeneralEq( zgp, rgp, 1e-8, "gradParam not equal")
end

--]]


return tester:add(test):run()
