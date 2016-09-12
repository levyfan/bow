%% person re-identification on Market-1501 dataset by metric learning

clc; clear all; close all;
run('../KISSME/toolbox/init.m');
addpath('../');

%% load data for feature extraction
codebooksize = 350;

%% Set up parameters
params.numCoeffs = 200; %dimensionality reduction by PCA to 200 dimension

%% Extract training features and train PCA
Hist_train = importdata('market1501_new_hist_train_500_20.0.mat');
[ux,eigvalue,u,m] = mypca(Hist_train);

%% Extract testing and query features and apply PCA
Hist_test = importdata('market1501_new_hist_test_500_20.0.mat');
Hist_query = importdata('market1501_new_hist_query_500_20.0.mat');
ux_test=u'*(Hist_test-repmat(m,1,size(Hist_test,2)));
ux_query=u'*(Hist_query-repmat(m,1,size(Hist_query,2)));

%% generate ground truth pairs for training
label = importdata('train_label.mat'); % identity label of all training bboxes
cam = importdata('train_cam.mat'); % camera label of all training bboxes

uni_label = unique(label);
idxa = []; % index of the first image in a pair
idxb = []; % index of the second image in a pair
flag = []; % indicate whether two images are of the same identity
for n = 1:length(uni_label)
    curr_label = uni_label(n);
    pos = find(label == uni_label(n));
    comb = combntns(pos,2);
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
nTrainImg = length(label);
rand_pos = ceil(rand(150000, 2).*nTrainImg);
ID1 = label(rand_pos(:, 1));
ID2 = label(rand_pos(:, 2));
Eq_pos = find(ID1 == ID2);
diff_pos = setdiff(1:150000, Eq_pos); % remove pairs of the same identity
rand_pos = rand_pos(diff_pos, :);
cam1 = cam(rand_pos(:, 1));
cam2 = cam(rand_pos(:, 2));
Eq_pos = find(cam1 == cam2);
diff_pos = setdiff(1:length(rand_pos), Eq_pos);% remove pairs of the same camera

%%%% training image pairs and their ground truth labels %%%%%%%%
idxa = [idxa; rand_pos(diff_pos(1:nPos), 1)];
idxb = [idxb; rand_pos(diff_pos(1:nPos), 2)];
flag = [flag; zeros(nPos, 1)];

%% Metric learning

% ITML, LMNN are slow to train. If you want to test the two methods, please
% set params.numCoeffs to 100 or lower.
pair_metric_learn_algs = {...
    LearnAlgoKISSME(params), ...
%     LearnAlgoMahal(), ...
%     LearnAlgoMLEuclidean(), ...
%     LearnAlgoITML(), ... 
%     LearnAlgoLMNN() ...  
    };

[ ds ] = TrainValidateMarket(struct(), pair_metric_learn_algs,ux(1:params.numCoeffs,:),ux_test(1:params.numCoeffs,:),ux_query(1:params.numCoeffs,:),idxa,idxb,flag);

%% plot CMC curves
% please check the mAP and cmc (r1) scores in ds
figure;
s = 50;
CMC_curve = [ds.kissme.cmc; ]; %ds.mahal.cmc; ds.identity.cmc ];
plot(1:s, CMC_curve(:, 1:s));

