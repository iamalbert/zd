#!/usr/bin/env th

require 'totem'
require 'zd'

local test = {}

local tester = totem.Tester()

local in_dim, out_dim = 10, 30

local weight = torch

function test.A()
    
    local cnt = 0
    local max_epoch = 50
    local expr = zd.Expr {
        runners = {},
        max_epoch = max_epoch, 
        events = {
            after_epoch = function()
                cnt = cnt + 1
            end
        }
    }

    expr:run()

    tester:asserteq( cnt, max_epoch, "after_epoch not called correctly")
end


-- add more tests by adding to the 'test' table


return tester:add(test):run()
