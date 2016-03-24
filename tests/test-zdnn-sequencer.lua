#!/usr/bin/env th

require 'zd'
require 'totem'

local test = totem.TestSuite()

local tester = totem.Tester()

local inDim, outDim = 300, 100
local seqLen = 956

local rec = nn.GRU(inDim, outDim)
local zm = zdnn.Sequencer(rec)
local rm = nn.Sequencer(rec:clone())

test["Sequencer:forward: 1-batch"] = function()
    local input = torch.rand(seqLen, 1, inDim)
    local input_table = nn.SplitTable(1):forward( input )

    tester:assertGeneralEq( #input_table, input:size(1),
        1e-8, "split length incorrect")
    tester:assertGeneralEq( input_table[1], input[1],
        1e-8, "split incorrect")

    zm:evaluate()
    rm:evaluate()
    local pred = zm:forward( input )
    local pred_table = rm:forward( input_table )

    tester:assertGeneralEq( 
        pred:size(), torch.LongStorage{seqLen, 1, outDim},
        1e-8, "size should be " .. seqLen .. " x 1 x " .. outDim
    )

    for i=1,seqLen do
        tester:assertGeneralEq(
            pred[i], pred_table[i],
            1e-8, "step " .. i .. " is different"
        )
    end
end

test["Sequencer:forward: n-batch"] = function()

    local batchSize = 12

    local input = torch.rand(seqLen, batchSize, inDim)
    local input_table = nn.SplitTable(1):forward( input )

    tester:assertGeneralEq( #input_table, input:size(1),
        1e-8, "split length incorrect")
    tester:assertGeneralEq( input_table[1], input[1],
        1e-8, "split incorrect")

    zm:evaluate()
    rm:evaluate()
    local pred = zm:forward( input )
    local pred_table = rm:forward( input_table )

    tester:assertGeneralEq( 
        pred:size(), torch.LongStorage{seqLen, batchSize, outDim},
        1e-8, 
        "size should be " .. seqLen .. " x " .. batchSize .. " x " .. outDim
    )

    for i=1,seqLen do
        tester:assertGeneralEq(
            pred[i], pred_table[i],
            1e-8, "step " .. i .. " is different"
        )
    end
end

--[[
test["Sequencer:forward: error input"] = function()
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

test["Sequencer:forward: no-batch"] = function ()
    local input = torch.rand(seqLen, inDim)
    local input_table = nn.SplitTable(1):forward( input )

    tester:assertGeneralEq( #input_table, input:size(1),
        1e-8, "split length incorrect")
    tester:assertGeneralEq( input_table[1], input[1],
        1e-8, "split incorrect")

    zm:evaluate()
    rm:evaluate()
    local pred = zm:forward( input )
    local pred_table = rm:forward( input_table )

    tester:assertGeneralEq( 
        pred:size(), torch.LongStorage{seqLen, outDim},
        1e-8, "size should be " .. seqLen .. " x " .. outDim
    )

    for i=1,seqLen do
        tester:assertGeneralEq(
            pred[i], pred_table[i],
            1e-8, "step " .. i .. " is different"
        )
    end
end


--]]


return tester:add(test):run()
