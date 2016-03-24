local Transpose, parent = torch.class('zdnn.Transpose', 'nn.Module')

-- copy from the undocumented nn.Transpose
-- https://raw.githubusercontent.com/torch/nn/6916775db4731b5c40656085471448be476a321d/Transpose.lua

-- transpose dimensions:
-- n = zdnn.Transpose({1,4},{1,-3})
-- will transpose dims 1 and 4, then 1 and 3...
-- accept negative index, the last dimension is -1
-- zdnn.Transpose( {1,-1] ):forward( torch.rand(2,3,4,5) ) -> [5,3,4,2]

function Transpose:__init(...)
   parent.__init(self)
   self.permutations = {...}
end

function Transpose:updateOutput(input)
   local dim = input:dim()
   for _,perm in ipairs(self.permutations) do
      local perm1 = (perm[1] < 0) and (dim+perm[1]+1) or perm[1]
      local perm2 = (perm[2] < 0) and (dim+perm[2]+1) or perm[2]

      input = input:transpose(perm1,perm2)
   end
   self.output:resizeAs(input):copy(input)
   return self.output
end

function Transpose:updateGradInput(input, gradOutput)
   local dim = input:dim()
   for i = #self.permutations,1,-1 do
      local perm = self.permutations[i]

      local perm1 = (perm[1] < 0) and (dim+perm[1]+1) or perm[1]
      local perm2 = (perm[2] < 0) and (dim+perm[2]+1) or perm[2]

      gradOutput = gradOutput:transpose(perm1,perm2)
   end
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   return self.gradInput
end

