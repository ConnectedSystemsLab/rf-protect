data_path = '../traj_save/trajs_collected_Home/traj_collect/raw/';
gt_data_path = 'D:\UIUC\Tag\Jay-data\SIGCOMM_data\trajs_gt';
data_info = dir(data_path);
num_trajs = (length(data_info) - 2);
%% Prefiltered

error_dist_all = [];
error_angle_all = [];

for i1=1:num_trajs
    name_data = data_info(i1+2).name;
    name_split = strsplit(name_data,'.');
    traj_index = str2num(name_split{1});
    data_gt = load([gt_data_path, '/ad_raw/', num2str(traj_index), '.mat']).traj_angle_dist; % 1 d, 2 angle
    data_gt(:,2) = data_gt(:,2)/pi *180;

    data_ours = load([data_path, '/', num2str(traj_index), '.mat']).traj1; % 2 d, 1 angle
%     error_dist = abs(data_gt(:, 1) - data_ours(:, 2));
%     error_angle = abs(data_gt(:, 2) - data_ours(:, 1));
    error_dist = abs(data_gt(:, 1) - data_ours(:, 2));
    error_angle = abs(data_gt(:, 2) - data_ours(:, 1));

    error_dist_all = [error_dist_all; error_dist];
    error_angle_all = [error_angle_all; error_angle];
end
% save('../SIGCOMM_plots/Results_data/error_dist_all_2.mat', "error_dist_all");
% save('../SIGCOMM_plots/Results_data/error_angle_all_2.mat', "error_angle_all");
figure;
cdfplot(error_dist_all);
title('Distance\_Prefiltered')
figure;
cdfplot(error_angle_all);
title('Angle\_Prefiltered')

%% ERROR in Distance

error_all = [];
error_all_mean = [];

traj_index_all = [];
for i1=1:num_trajs
    name_data = data_info(i1+2).name;
    name_split = strsplit(name_data,'.');
    traj_index = str2num(name_split{1});
    data_gt = load([gt_data_path, '/ad_raw/', num2str(traj_index), '.mat']).traj_angle_dist; % 1 d, 2 angle
    data_gt(:,2) = data_gt(:,2)/pi *180;

    angle_array = data_gt(:,2);
    distance_array = data_gt(:,1);
    xy_gt = [distance_array.*cosd(angle_array), distance_array.*sind(angle_array)];

    data_ours = load([data_path, '/', num2str(traj_index), '.mat']).traj1; % 2 d, 1 angle
    angle_array = data_ours(:,1);
    distance_array = data_ours(:,2);
    xy_ours = [distance_array.*cosd(angle_array), distance_array.*sind(angle_array)];

    error_tmp = xy_ours - xy_gt;
    error_tmp = error_tmp.*error_tmp;
    error_tmp = sqrt(error_tmp(:, 1)+error_tmp(:, 2)); 
    error_tmp_mean = mean(error_tmp);
    error_all = [error_dist_all; error_tmp];

    error_all_mean = [error_all_mean; error_tmp_mean];
    traj_index_all = [traj_index_all; traj_index]
end
figure;
cdfplot(error_all);
title('Trajectory Error');