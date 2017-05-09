%% person re-identification on VIPeR dataset by metric learning

clc; clear all; close all;
run('../KISSME/toolbox/init.m');
addpath('../');

load viper_dirs.mat;
%% Set up parameters
params.numCoeffs = 100; %dimensionality reduction by PCA to 200 dimension


%% trial
load ../../resources/randselect10.mat;
load feature_rows.mat;

nSample = 632/2;
lookrank = nSample;
nloop = 10;
accuracy = zeros(lookrank,nloop);
for loop = 1:nloop
    loop
    clear HistA HistB;
    load(['hist_cam_loop' num2str(loop-1) '_500_20.0.mat']);
    % load('hist_500_20.0.mat');
    % HistA = HistA(hog_rows,:);
    % HistB = HistB(hog_rows,:);

    testIndex = selectsample(:,loop);
    testA = HistA(:,testIndex);
    testB = HistB(:,testIndex);

    bestM = eye(size(HistA, 1));
    score = sqdist(testA, testB, bestM);

    testscoreA = score';
    testscoreB = score;

    [qAmatch, qArank] = sort(testscoreA,'ascend');
    [qBmatch, qBrank] = sort(testscoreB,'ascend');
    qArankrnn = qArank(1:lookrank,:);
    qBrankrnn = qBrank(1:lookrank,:);

    countqA = zeros(1,lookrank);
    countqB = zeros(1,lookrank);
    for i = 1:nSample
        index = find(qArankrnn(:,i)==i);
        if(index>0)
            countqA(index:end) = countqA(index:end)+1;
        end
        index = find(qBrankrnn(:,i)==i);
        if(index>0)
            countqB(index:end) = countqB(index:end)+1;
        end
    end

    accuracyqA = countqA/nSample;
    accuracyqB = countqB/nSample;
    accuracy(:,loop) = (accuracyqA+accuracyqB)/2;
end
MR = mean(accuracy,2);
MR = MR(1:50)';
