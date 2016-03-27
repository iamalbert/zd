local Class, Parent = torch.class('zdnn.PairwiseReplicate', 'nn.Module')

--[[
This module accecpt two MxD matrices , A, B, and 
output two MxMxD matrices A', B' where

    A'[i][j][k] = A[i][k]
    B'[i][j][k] = B[j][k]

    for 1 <= i,j <= M and 1 <= k <= D
--]]

function Class:__init(...)
    Parent.__init(self)
end

function Class:updateOutput(input)
    local A,B = input[1], input[2]

    assert(A:dim() == 2 and B:dim() == 2)
    assert(A:size(1) == B:size(1) and A:size(2) == B:size(2))

    local M, D = A:size(1), A:size(2)
    local targetSize = torch.LongStorage { M, M, D}

    local nA = A:view( torch.LongStorage{M,1,D} ):expand( targetSize )
    local nB = B:view( torch.LongStorage{1,M,D} ):expand( targetSize )

    self.output =  {nA,nB}
    return self.output
end

function Class:updateGradInput(input, gradOutput)
    self.gradInput = {
        gradOutput[1]:sum(2):viewAs(input[1]),
        gradOutput[2]:sum(1):viewAs(input[2]),
    }
    return self.gradInput
end
