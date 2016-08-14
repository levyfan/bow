clear all; clc; close all;
%% PCA
% Hist_train = importdata('market1501_hist_test_500_20.0.mat');
% [ ~,u,m ] = mypca(Hist_train);
% clear Hist_train;

%% search the database and calcuate re-id accuracy
Hist_query = importdata('market1501_hist_query_500_20_0_512.mat');
Hist_test = importdata('market1501_hist_test_500_20_0_512.mat');
nQuery = size(Hist_query, 2);
nTest = size(Hist_test, 2);

% Hist_query = u'*(Hist_query - m*ones(1,nQuery));
% Hist_test = u'*(Hist_test - m*ones(1,nTest));

ap = zeros(nQuery, 1); % average precision
CMC = zeros(nQuery, nTest);
r1 = 0; % rank 1 precision with single query

knn = 1; % number of expanded queries. knn = 1 yields best result
queryCAM = importdata('queryCam.mat'); % camera ID for each query
queryID = importdata('queryID.mat');
testCAM = importdata('testCam.mat'); % camera ID for each database image
testID = importdata('testID.mat');

dist =  -(Hist_test')*Hist_query;

for k = 1:nQuery
    k
    % load groud truth for each query (good and junk)
    good_index = intersect(find(testID == queryID(k)), find(testCAM ~= queryCAM(k)))';% images with the same ID but different camera from the query
    junk_index1 = find(testID == -1);% images neither good nor bad in terms of bbox quality
    junk_index2 = intersect(find(testID == queryID(k)), find(testCAM == queryCAM(k))); % images with the same ID and the same camera as the query
    junk_index = [junk_index1; junk_index2]';
    tic
    score = dist(:, k);
    
    % sort database images according Euclidean distance
    [~, index] = sort(score, 'ascend');  % single query
    
    [ap(k), CMC(k, :)] = compute_AP(good_index, junk_index, index);% compute AP for single query

end

CMC = mean(CMC);
%% print result
fprintf('single query:                                   mAP = %f, r1 precision = %f\r\n', mean(ap), CMC(1));