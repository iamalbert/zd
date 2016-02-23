local Class, Parent = torch.class('zd.Expr')

function Class:__init(config)
    assert(config, torch.type(self) .. " requires a config table")
    assert(config.runners, "field `runners` are required")

    self.runners = config.runners
    self.max_epoch = config.max_epoch or -1 

    self.events = {}
end

function Class:run()
    local n_epoch = 0
    while n_epoch ~= self.max_epoch do
        n_epoch = n_epoch + 1
        for k, value in ipairs(self.runners) do
            local runner, data_iter = value[1], value[2]

            assert( torch.isTypeOf(runner, zd.Evaluator),
                "runner " .. k .. " shall be an zd.Evaluator, got " 
                    .. torch.type(runner) )
            assert( torch.isTypeOf(data_iter, zd.Sampler),
                "runner " .. k .. " shall be a zd.Sampler, got " 
                    .. torch.type(data_iter) )

            local report = runner:run(data_iter)
            collectgarbage()
        end
        local to_continue = self.events.after_epoch and events.after_epoch(state, reports) 
        if to_continue == false then break end
        collectgarbage()
    end
end


