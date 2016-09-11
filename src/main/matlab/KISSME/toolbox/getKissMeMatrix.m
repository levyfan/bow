function [ M ] = getKissMeMatrix(ds, learn_algs, X, idxa, idxb, flag)
%   
% Input:
%
%  ds - data struct that stores the result
%  learn_algs - algorithms that are used for cross validation
%  X - input matrix, each column is an input vector [DxN*2]. N is the
%  number of pairs.
%  idxa - index of image A in X [1xN]
%  idxb - index of image B in X [1xN]
%
% Output:
%
%  ds - contains the result

c = 1;
clc;
% train 
for aC=1:length(learn_algs)
    cHandle = learn_algs{aC};
    fprintf('    training %s ',upper(cHandle.type));
    if aC > 3
        s = learnPairwise(cHandle,X,idxa(1:10:end),idxb(1:10:end),logical(flag(1:10:end)));
    else
        s = learnPairwise(cHandle,X,idxa,idxb,logical(flag));
    end
    if ~isempty(fieldnames(s))
        fprintf('... done in %.4fs\n',s.t);
        ds(c).(cHandle.type) = s;
    else
        fprintf('... not available');
    end
end

% test
names = fieldnames(ds(c));
nameCounter = 1;
M = ds(c).(names{nameCounter}).M;

end