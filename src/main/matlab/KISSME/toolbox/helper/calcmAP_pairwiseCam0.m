function [ mAP, r1_precision ] = calcmAP_pairwiseCam0( M, data_test, data_query, indexMap, testCam, queryCam, nModel )
query=dir('dataset/query/*.jpg');
dataset=dir('dataset/bounding_box_test/*.jpg');
nTest = length(testCam);
nQuery = length(queryCam);
dist = zeros(nTest,nQuery);

ap = zeros(size(dist, 2), 1);
r1 = 0;
for k = 1:size(dist, 2)
    k
    for i = 1:nTest
        dist(i,k) = (data_query(:,k)-data_test(:,i))'*M{indexMap(testCam(i),queryCam(k))}*(data_query(:,k)-data_test(:,i));
    end
    
    % find groudtruth index (good and junk)
    file_name1 = ['dataset/gt_query/' query(k).name(1:19) '_good.mat'];
    good_index = importdata(file_name1);
    file_name2 = ['dataset/gt_query/' query(k).name(1:19) '_junk.mat'];
    junk_index = importdata(file_name2); 
    score = dist(:, k);
    [~, index] = sort(score, 'ascend');% the higher, the better
    ap(k) = compute_AP(good_index, junk_index, index);% see compute_AP
    count = 1;
    rank_id = str2double(dataset(index(count)).name(1:4)); %the same id,the same person
    rank_cam = dataset(index(count)).name(7);  % camera number
    query_id = str2double(query(k).name(1:4));
    query_cam = query(k).name(7);
    while( rank_id == query_id )
        if( rank_cam == query_cam) % ignore the same person from the same camera
            count = count+1;
            rank_id = str2double(dataset(index(count)).name(1:4));
            rank_cam = dataset(index(count)).name(7);
        elseif( rank_cam ~= query_cam) % hit!
            r1 = r1 + 1;
            break;
        end
    end
end
%% r1 precision
r1_precision = r1/size(dist, 2);
mAP = mean(ap);

end

