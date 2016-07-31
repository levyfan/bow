function [ ux,u,m ] = mypca( X )
[~,N]  = size(X);
m=mean(X,2);
X = X - m*ones(1,N);
[u,S,~] = svd(X,'econ');
[~,index] = sort(diag(S),'descend');
u = u(:,index);
ux=u'*X;
end

