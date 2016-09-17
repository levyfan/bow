%% person re-identification on VIPeR dataset by metric learning

clc; clear all; close all;
run('../KISSME/toolbox/init.m');
addpath('../');

load prid450s_dirs.mat;
%% Set up parameters
params.numCoeffs = 100; %dimensionality reduction by PCA to 200 dimension


%% trial
load ../../resources/randselect10_prid450s.mat;
load prid450s_hist_500_20.0.mat;

nSample = size(HistA,2)/2;
lookrank = nSample;
nloop = 10;
accuracy = zeros(lookrank,nloop);
for loop = 1:10
    loop
    clear HistA HistB;
    load(['hist_loop' num2str(loop-1) '_500_20.0.mat']);
    % load('prid450s_hist_500_20.0.mat');
    
    testIndex = selectsample(:,loop);
    trainIndex = 1:numberA;
    trainIndex(testIndex) = [];
    
    %% Extract training features and train PCA
    [ux,eigvalue,u,m] = mypca([HistA(:,trainIndex),HistB(:,trainIndex)]);
    Hist_train = (ux(1:params.numCoeffs,:));
    trainCam = [ones(1,nSample),2*ones(1,nSample)];
        
    testA = HistA(:,testIndex);
    testB = HistB(:,testIndex);
    testA = u'*(testA-repmat(m,1,size(testA,2)));
    testB = u'*(testB-repmat(m,1,size(testB,2)));
    testA = (testA(1:params.numCoeffs,:));
    testB = (testB(1:params.numCoeffs,:));

    %% generate ground truth pairs for training
    label = [trainIndex,trainIndex]; % identity label of all training bboxes
    cam = trainCam; % camera label of all training bboxes
    uni_label = unique(label);
    idxa = []; % index of the first image in a pair
    idxb = []; % index of the second image in a pair
    flag = []; % indicate whether two images are of the same identity
    for n = 1:length(uni_label)
        curr_label = uni_label(n);
        pos = find(label == uni_label(n));
        comb = nchoosek(pos,2);
        idxa = [idxa; comb(:, 1)];
        idxb = [idxb; comb(:, 2)];
    end
    % remove pairs from the same camera

    cam1 = cam(idxa);
    cam2 = cam(idxb);
    Eq_pos = find(cam1 == cam2);
    diff_pos = setdiff(1:length(idxa), Eq_pos);
    idxa = idxa(diff_pos);
    idxb = idxb(diff_pos);

    nPos = length(idxa);
    flag = [flag; ones(nPos, 1)];

    % generate negative training pairs
    nNeg = 0;
    for n = 1:length(uni_label)
        curr_label = uni_label(n);
        neg = find(label ~= curr_label);
        neg1 = find(cam(neg) ~= cam(curr_label));
        idxa = [idxa; curr_label*ones(length(neg1),1)];
        idxb = [idxb; label(neg(neg1))'];
        nNeg = nNeg + length(neg1);
    end

    %%%% training image pairs and their ground truth labels %%%%%%%%
    flag = [flag; zeros(nNeg, 1)];
    cama = cam(idxa);
    camb = cam(idxb);

    % Metric learning
    pair_metric_learn_algs = {...
    LearnAlgoKISSME(params), ...
    %     LearnAlgoMahal(), ...
    %     LearnAlgoMLEuclidean(), ...
    %     LearnAlgoITML(), ... 
    %     LearnAlgoLMNN() ...  
    };
    bestM = getKissMeMatrix(struct(), pair_metric_learn_algs,Hist_train,idxa,idxb,flag);

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
disp(MR([1,20]));