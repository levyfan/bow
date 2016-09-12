function [ mAP, r1_precision ] = calcmAP( M, data_test, data_query )
query=dir('/data/reid/market1501/dataset/query/*.jpg');
dataset=dir('/data/reid/market1501/dataset/bounding_box_test/*.jpg');
dist = sqdist(data_test, data_query, M);
ap = zeros(size(dist, 2), 1);
r1 = 0;
for k = 1:size(dist, 2)
    k
    % find groudtruth index (good and junk)
    file_name1 = ['/data/reid/market1501/dataset/gt_query/' query(k).name(1:19) '_good.mat'];
    good_index = importdata(file_name1);
    file_name2 = ['/data/reid/market1501/dataset/gt_query/' query(k).name(1:19) '_junk.mat'];
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

