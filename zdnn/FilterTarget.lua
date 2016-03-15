if zdnn.FilterTarget then return end


local Class,Parent = torch.class('zdnn.FilterTarget', 'nn.Module')

function Class:__init(candidates, winsize)
	Parent.__init(self)
	self.candidates = candidates
	self.candidates_id = table.transpose(candidates)

	winsize = winsize or 0
	self.winsize = winsize
	self.none = torch.LongTensor(1+2*winsize):fill(0)
end

function Class:updateOutput(input)
	assert( input:dim() == 1, "current only 1D vector is supported")

	local filtered = {}
	for i=1,input:size(1) do
		local n = input[i]
		if self.candidates_id[n] ~= nil then
			table.insert(filtered, i)
		end
	end

	local winsize = self.winsize

	if #filtered == 0 then
		self.output = torch.LongTensor(1,winsize):zero()
	else
		local output = torch.LongTensor(#filtered, winsize*2+1):zero()
		for j = 1,#filtered do
			local i = filtered[j]

			local first,last = i-winsize, i+winsize
			if first < 1 then first = 1 end
			if last > input:size(1) then last = input:size(1) end

			local len = last - first + 1
			local offset = 2*winsize+1 - len + 1

			output:sub(j,j,offset,-1):copy( input:sub(first,last) )
		end
		self.output = output
	end

	return self.output
end
