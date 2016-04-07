require 'torch-rnn'
local Class, Parent = torch.class('zdnn.Bidirection', 'nn.Sequential')

function Class:__init( fmod, bmod, merge )
    Parent.__init(self)

    assert(fmod)
    self.fmod = fmod
    self.bmod = bmod or fmod:clone():setreverse(true)
    self.merge = merge or nn.JoinTable(2,2)
    self
        :add(
            nn.ConcatTable(3)
                :add(self.fmod)
                :add(self.bmod)
        )
        :add( self.merge )
end

function Class:__tostring()
	return torch.type(self) .. " { \n" .. 
        "  (1) " .. tostring(self.fmod) .. "\n" ..
        "  (2) " .. tostring(self.bmod) .. "\n" ..
        "  (+) " .. tostring(self.merge) .. "\n}"
end
