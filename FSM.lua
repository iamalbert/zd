do
	local FSM = torch.class('zd.FSM')

	local otherwise = {}
	FSM.OTHERWISE = otherwise
	function FSM.run(trans, conf)
		local i = 1
		-- print( i, #conf.data )
		while i <= #conf.data do
			conf.pos = i
			if conf.debug then
				print( i, conf.state, conf:currentStr() )
			end

			local t = trans[ conf.state ] 
			assert( t , "no such state: " .. conf.state )

			conf.newstate = nil
			local p 
			for wset , func in pairs(t) do
				if type(wset) == "table" then
					p = table.contains( wset, conf:currentStr() )
				else
					p = string.pos( wset, conf:currentStr() )
				end
				if p ~= nil then
					p = wset
					break
				end
			end

			if p == nil then
				p = otherwise
			end

			assert( t[p] ~= nil, "no valid transition" )

			local n, c = t[p]( conf, conf.stack )
			conf.newstate, conf.consume = n, c 

			assert( conf.newstate, "no return next state" )

			conf.state = conf.newstate
			if conf.consume == false then 
			else
				i = i + 1
			end
		end
		return conf
	end
end

do
	local Conf = torch.class('zd.FSMConfig')

	function Conf:__init ( data, init_state, payload )
		self.data = data
		self.pos = 1
		self.state = init_state
		self.stack = {}

		payload = payload or {}
		for k,v in pairs(payload) do
			self[k] = v
		end
	end
	function Conf:current()
		if type(self.data) == "string" then
			return string.at(self.data, self.pos)
		else
			return self.data[self.pos]
		end
	end
	function Conf:currentStr()
		if self.cur then
			return self:cur( self:current() )
		else
			return self:current()
		end
	end

end
