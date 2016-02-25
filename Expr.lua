local Class, Parent = torch.class('zd.Expr')

function Class:__init(config)
    assert(config, torch.type(self) .. " requires a config table")
    assert(config.runners, "field `runners` are required")

    self.runners = config.runners or {}
    for k, value in ipairs(self.runners) do
        local runner, data_iter = value[1], value[2]

        assert( torch.isTypeOf(runner, zd.Evaluator),
            "runner " .. k .. " shall be an zd.Evaluator, got " 
                .. torch.type(runner) )
        assert( torch.isTypeOf(data_iter, zd.Sampler),
            "runner " .. k .. " shall be a zd.Sampler, got " 
                .. torch.type(data_iter) )
    end


    self.max_epoch = config.max_epoch or -1 

    self.events = config.events or {}
end

function Class:run()
    local state = {
        n_epoch = 0
    }
    while state.n_epoch ~= self.max_epoch do
        state.n_epoch = state.n_epoch + 1

        local reports = {}
        for k, value in ipairs(self.runners) do
            local runner, data_iter = value[1], value[2]
            local report = runner:run(data_iter)

            table.insert(reports, report)
            collectgarbage()
        end
        local cb = self.events.after_epoch 
        if cb ~= nil then
            local to_continue = cb(self, state, reports) 
            if to_continue == false then break end
        end
    end
end


