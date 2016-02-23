#!/usr/bin/env th

require 'totem'

local test = {}

local tester = totem.Tester()

local in_dim, out_dim = 10, 30

local weight = torch

function test.A()
    local expr = zd.Expr {
        runners = {
            train,
            test,
        }
    }
end


-- add more tests by adding to the 'test' table


return tester:add(test):run()
