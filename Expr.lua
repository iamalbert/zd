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
    while n_epoch ~= config.max_epoch do
        n_epoch = n_epoch + 1
        for _, runner in ipairs(self.runners) do
            local report = runner:run()
            collectgarbage()
        end
        to_continue = self.events.after_epoch and events.after_epoch(state, reports) 
        if to_continue == false then break end
        collectgarbage()
    end
end


