#!/usr/bin/env th

require 'nn'
require 'optim'
require 'totem'

require 'zd'

local test = {}

local tester = totem.Tester()

local n_data, inDim, outDim = 1000, 30, 3
local data = torch.rand( n_data, inDim )

local weight = torch.rand(outDim, inDim)
local ans  = data * (weight:t())

local model = nn.Linear(inDim, outDim, false)
local criterion = nn.MSECriterion()
local feedback = optim.ConfusionMatrix(outDim)

test['Trainer:Train y=Mx'] = function()
    local iter = zd.Iterator {
        source = {
            input  = data,
            target = ans
        }
    }

    local trainer = zd.Trainer {
        model = model,
        criterion = criterion, 
        optimizer = optim.sgd,
        optim_config = {
            learningRate = 0.03
        }
    }

    for e=1,50 do
        trainer:run(iter)
    end

    tester:assertTensorEq( model.weight, weight, 1e-7, 
        "failed to train y=Mx, where M: [" .. outDim .. "x" .. inDim .. "]")
end

function test.TrainerPropagate()
    local called = {}
    for _, event in ipairs(zd.Trainer._allow_event_list) do
        called[event] = 0
    end

    local iter = zd.Iterator {
        source = {
            input  = data,
            target = ans
        }
    }
    local target = 1

    local incre = function(name)
        return function()
            called[name] = called[name] + 1
        end
    end

    local evaluator = zd.Trainer {
        model = model,
        criterion = criterion,
        feedback = feedback,
        optimizer = optim.sgd,
        events = {
            before_epoch = incre 'before_epoch',
            before_feedback = incre 'before_feedback',
            before_loss = incre 'before_loss',
            
            after_epoch = incre 'after_epoch',

            after_example = function(self, state, example )

                tester:assertTensorEq(example.input, data[state.n_example],
                    1e-7, "wrong input for n_example=" .. state.n_example
                )
                tester:assertTensorEq(example.target, ans[state.n_example],
                    1e-7, "wrong target for n_example=" .. state.n_example
                )

                tester:asserteq( example.output:dim(), 1, 
                    "output dimension wrong"
                )
                tester:asserteq( example.output:size(1), outDim, 
                    "output size wrong" 
                )
                
                called.after_example = called.after_example + 1
            end,
            before_example = function(self, state, example)
                tester:assertTensorEq(example.input, data[state.n_example],
                    1e-7, "wrong input for n_example=" .. state.n_example
                )
                tester:assertTensorEq(example.target, ans[state.n_example],
                    1e-7, "wrong target for n_example=" .. state.n_example
                )
                called.before_example = called.before_example + 1
            end
        }
    }


    local state = evaluator:run(iter)
    tester:asserteq( called.before_epoch, 1 ,
        "`before_epoch' should be called once")
    tester:asserteq( called.after_epoch, 1 ,
        "`after_epoch' should be called once")

    tester:asserteq( called.before_feedback, n_data, 
        "`before_example' does not make every example")
    tester:asserteq( called.before_loss, n_data, 
        "`before_loss' does not make every example")
    tester:asserteq( called.before_feedback, n_data, 
        "`before_feedback' does not make every example")
    tester:asserteq( called.after_example, n_data, 
        "`after_example' does not make every example")

end


-- add more tests by adding to the 'test' table

return tester:add(test):run()
