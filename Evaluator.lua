local Evaluator = torch.class('zd.Evaluator')

Evaluator._allow_event_list = {
    "before_epoch"      ,
        "before_example"    ,
             "before_feedback" ,
             "before_loss",
             "after_example"     ,
        "after_epoch"       ,
}

function Evaluator:__init(config)
    assert( config, torch.type(self) .. " requires a config table" )
    assert( config.model, "config requires field `model'" )
    self:_setup(config)
end

function Evaluator:run(sampler, criterion)

    local state = self:_pre_propogate()

    if false ~= self:_trigger("before_epoch", state) then
        self:_propogate(sampler, state)
        self:_trigger("after_epoch", state)
    end

    return state
end

function Evaluator:_setup(config)

    self._config = config

    self.feedback = config.feedback
    self.criterion = config.criterion
    -- self.sampler  = config.sampler
    self.events   = config.events or {}

    self.name     = config.name or torch.type(self)

    self.cuda     = config.cuda

    if self.cuda then
        self.model    = config.model:cuda()
    else
        self.model    = config.model
    end

    self.debug    = config.debug

    local allow_events = table.transpose( self._allow_event_list )

    local registered_events, unknown_events = {}, {}
    for event, handler in pairs(self.events) do 
        assert( allow_events[event], "no such event:" .. event )
    end
    --[[
        if allow_events[event] == nil then
            table.insert( unknown_events, event )
        else
            table.insert( registered_events, event )
        end
    end
    if self.debug then
        if #registered_events ~= 0 then
            print(self.name .. ': add listeners: '
                .. table.concat(registered_events, ", ") )
        end
        if #unknown_events ~= 0 then
            print(self.name .. ': known events: ' 
                .. table.concat(unknown_events,", "))
        end
    end
    --]]
end

function Evaluator:_trigger( event, ... )
    local func = self.events[event]
    if func ~= nil then
        return func( self, ...)
    else
        return nil
    end
end


function Evaluator:_pre_propogate()
    self.model:evaluate()
    if self.feedback then
        self.feedback:zero()
    end
    local state = {
      n_example = 0,
      loss = 0,
      feedback = self.feedback
    }
    return state
end

function Evaluator:_propogate(sampler, state)

    state.tot_example = sampler:reset()
    state.is_batch = sampler:batchSize() > 0

    repeat 
        state.n_example = state.n_example + 1

        local example = sampler:next()

        if false ~= self:_trigger("before_example", state, example) then
            self:_do_example(example, state)
            self:_trigger("after_example", state, example)
        end

    until sampler:finished()

end

function Evaluator:_do_example( example, state )
    example.output = self.model:forward(example.input)

    if false ~= self:_trigger("before_feedback",state,example) then
      self:_add_feedback(example, state)
    end

    if false ~= self:_trigger("before_loss", state, example ) then
        self:_compute_loss( example, state )
    end
end

function Evaluator:_compute_loss( example, state )
    if self.criterion ~= nil then
        example.loss = self.criterion:forward(example.output, example.target)
        state.loss = state.loss + example.loss
    end
end

function Evaluator:_add_feedback( example, state )
    if self.feedback then
        if state.is_batch then 
            self.feedback:batchAdd( example.output, example.target ) 
        else
            self.feedback:add( example.output, example.target )
        end
    end
    return self
end

