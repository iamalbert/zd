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
    assert( config, torch.type(self) .. " requires a config table" )

    assert( config.model, "config requires field `model'" )
    assert( config.criterion, "config requires field `criterion'" )
    assert( config.optimizer, "config requires field `optimizer'" )

    parent.__init(self, config)
end

function Trainer:_setup(config)
    parent._setup(self, config)

    self.criterion = config.criterion
    self.optimizer = config.optimizer

    self.params , self.gradParams = self.model:getParameters()
end

--[[
function Trainer:_do_example(example, state)
    parent._do_example self, example, model, criterion, state)
end
--]]
--

function Trainer:_compute_loss(example, state )
    parent._compute_loss(self, example, state )

    if false ~= self:_trigger("before_backward", state, example ) then
        example.gradOutput = self.criterion:backward( 
            example.output, example.target )
        example.gradInput = self.model:backward(
            example.input, example.gradOutput )
        self:_trigger("after_backward", state, example )

        if false ~= self:_trigger("before_update", state, example) then
            self:_update( example, state )
            self:_trigger("after_update", state, example)
        end
    end

end

function Trainer:_pre_propogate()
    local state = parent._pre_propogate(self)

    -- if self.params == nil or self.gradParams == nil then
    -- end
    --
    local model = self.model

    model:training()
    if model.forget then model:forget() end

    model:zeroGradParameters()

    state.optim_state = table.deepcopy( self._config.optim_state or {} )

    return state
end

function Trainer:_update( example, state )

    local params, gradParams = self.params, self.gradParams

    if state.optim_state == nil then
        state.optim_state = {}
    end

    self:_perform_l2reg(self._config.l2reg)
    self:_perform_momentum(state.optim_state, self._config.momentum)

    -- parameter update
    self._config.optimizer(
        function (p)
            if params ~= p then params:copy(p) end
            return example.loss, gradParams
        end,
        params, self._config.optim_config, state.optim_state
    )

    self.model:zeroGradParameters()
end
