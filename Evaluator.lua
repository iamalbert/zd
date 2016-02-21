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
    local config = config or {}

    self._config = config

    self.feedback = config.feedback
    -- self.sampler  = config.sampler
    self.events   = config.events or {}

    self.name     = config.name

    local allow_events = table.transpose( self._allow_event_list )

    local registered_events, unknown_events = {}, {}
    for event, handler in pairs(self.events) do 
        if allow_events[event] == nil then
            table.insert( unknown_events, event )
        else
            table.insert( registered_events, event )
        end
    end
    if #registered_events ~= 0 then
        print(self.name .. ': add listeners: ' .. table.concat(registered_events, ", ") )
    end
    if #unknown_events ~= 0 then
        print(self.name .. ': known events: ' .. table.concat(unknown_events,", "))
    end
end

function Evaluator:_trigger( event, ... )
    local func = self.events[events]
    if func ~= nil then
        return func( self, ...)
    else
        return nil
    end
end

function Evaluator:run(model, criterion, sampler)
    local state = self:_pre_propogate(model)

    if false ~= self:_trigger("before_epoch", state) then
        self:_propogate(model, criterion, sampler, state)
        self:_trigger("after_epoch", state)
    end

    return state
end

function Evaluator:_pre_propogate(model)
    model:evaluate()
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

function Evaluator:_propogate(model, criterion, sampler, state)

    state.tot_example = sampler:reset()

    for _, example in zd.data_iter(data_view) do
        state.n_example = state.n_example + 1

        if false ~= self:_trigger("before_example", state, example) then
            self:_do_example(example, model, criterion, state)
            self:_trigger("after_example", state, example)
        end
    end
end

function Evaluator:_do_example( example, model, criterion, state )
    example.output = model:forward(example.input)

    if false ~= self:_trigger("before_feedback",state,example) then
      self:_add_feedback(example, state)
    end

    if false ~= self:_trigger("before_loss", state, example ) then
        self:_compute_loss( example, model, criterion, state )
    end
end

function Evaluator:_compute_loss( example, model, criterion, state )
    if criterion ~= nil then
        example.loss = criterion:forward(example.output, example.target)
        state.loss = state.loss + example.loss
    end
end
function Evaluator:_add_feedback( example, state )
    local sampler = self.sampler
    if sampler and self.feedback then
        if sampler.is_batch then 
            self.feedback:batchAdd( example.output, example.target ) 
        else
            self.feedback:add( example.output, example.target )
          end
    end
    return self
end

