function [ output ] = mySOPD( X, idxa, idxb )
output = zeros(size(X,1));
for i = 1:length(idxa)
    data = X(:,idxa(i))-X(:,idxb(i));
    output = data*data'+output;
end
end

