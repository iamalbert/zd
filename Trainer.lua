local Trainer, parent = torch.class('zd.Trainer', 'zd.Evaluator')

Trainer._allow_event_list = {
    "before_epoch"      ,
        "before_example"    ,
            "before_feedback" ,
            "before_loss",
            "before_backward",
                "after_backward",
                "before_update",
                    "after_update",
            "after_example"     ,
      "after_epoch"       ,
}

function Trainer:__init(config)
    parent.__init(self, config)
end

--[[
function Trainer:_do_example(example, state)
    parent._do_example self, example, model, criterion, state)
end
--]]
--
function Trainer:_compute_loss(example, model, criterion, state )
    zd.Evaluator._compute_loss(self, example, model, criterion, state )

    if false ~= self:_trigger("before_backward", state, example ) then
        example.gradOutput = criterion:backward( example.output, example.target )
        example.gradInput = model:backward( example.input, example.gradOutput )
        self:_trigger("after_backward", state, example )

        if false ~= self:_trigger("before_update", state, example) then
            self:_update( example, model, state )
            self:_trigger("after_update", state, example)
        end
    end
end

function Trainer:_pre_propogate(model)
  local state = zd.Evaluator._pre_propogate(self, model)

  -- if self.params == nil or self.gradParams == nil then
  -- end

  model:training()
  model:forget()
  model:zeroGradParameters()

  self.params , self.gradParams = model:getParameters()

  state.optim_state = table.deepcopy( self._config.optim_state or {} )
  state.optim_config = self._config.optim_config

  return state
end

function Trainer:_update( example, model, state )

local params, gradParams = self.params, self.gradParams

local l2reg = self._config.l2reg or 0
local mom   = self._config.momentum or 0

if l2reg ~= 0 then
    gradParams:add( l2reg, params )
end

if mom ~= 0 then
    do
        local state = state.optim_state
        local damp  = mom
        local dfdx  = gradParams

        if not state.dfdx then
           state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
        else
           state.dfdx:mul(mom):add(1-damp, dfdx)
        end
        if self._config.nesterov then
            dfdx:add(mom, state.dfdx)
        else
            dfdx = state.dfdx
        end
   end
end

-- print(params:sum(), gradParams:sum() )
-- print(self._config.optim_config)

self._config.optimizer(
    function (p)
        if params ~= p then params:copy(p) end
        return example.loss, gradParams
    end,
    params, self._config.optim_config, state.optim_state
)
-- print(params:sum(), gradParams:sum() )
model:zeroGradParameters()
end
