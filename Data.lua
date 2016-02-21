do
    local Class = torch.class('zd.Sampler')

    function Class:__init(config)
        assert( config,  "require a config table"  )
        assert( config and zd.util.isTable(config.source), 
            "field `source' is required"  )
        self._source = config.source
        self._config = config
    end

    function Class:reset()
        self.state = {
            index = 1,
        }
        local max = self:rewind()
        assert( zd.util.isNumber(max), "::rewind() must return a number,"
            .. "got `" .. type(max) .. "' instead.")
        max = math.floor(max)
        self.state.max = max
        return max
    end

    function Class:batchSize()
        return math.floor( tonumber( self._config.batch or 0 ) )
    end

    function Class:rewind()
        error "not implemented"
    end

    function Class:next()
        error "not implemented"
    end

    function Class:finished(state)
        local S = self.state
        if S and (S.index > S.max) then
            return true
        else
            return false
        end
    end

    function Class:each( func )
        self:reset()
        repeat 
            func( self:next() )
        until self:finished()
    end
end

do 
    local Class, parent = torch.class('zd.Iterator', 'zd.Sampler')

    function Class:rewind()
        local source = self._source
        local config = self._config
        
        local state = self.state
        state.max = nil 

        for name, db in pairs(source) do
            if state.max == nil then
                state.max = zd.util.get_size( db )
            else
                if zd.util.get_size(db) ~= state.max then
                    error("inconsitent length of data sources")
                end
            end
        end
        assert( state.max, "no data given")

        local order

        if config.shuffle then
            order = torch.randperm( state.max ):long()
        else
            order = torch.range(1, state.max ):long()
        end

        if self:batchSize() > 0 then
            local bs = self:batchSize()
            local pad_length = math.ceil( state.max / bs ) * bs - state.max

            if pad_length > 0 then
                order:resize( state.max + pad_length )
                order:sub( state.max+1, state.max+pad_length):fill(0)
            end

            state._pad_length = pad_length

            state.max = (state.max + pad_length ) / bs
            order:resize( state.max, bs )
        else
            order:resize( state.max, 1 )
        end

        state.order = order
        return state.max
    end

    function Class:next()
        local state, source =  self.state, self._source
        local cur = state.order[ state.index ]

        if state.index == state.max and state._pad_length ~= nil then
            cur = cur:sub(1, - state._pad_length - 1)
        end

        local ret = {}
        for name, data in pairs(source) do
            local datum
            if zd.util.isTensor(data) then
                if self:batchSize() > 0 then
                    datum = data:index( 1, cur )
                else
                    datum = data:select( 1, cur[1] )
                end
            else -- data is table
                if self:batchSize() > 0 then
                    datum = table.index(data, cur)
                else
                    datum = data[ cur[1] ]
                end
            end
            ret[name] = datum
        end
        local oldindex = state.index

        state.index = state.index + 1

        return ret, oldindex
    end
end
