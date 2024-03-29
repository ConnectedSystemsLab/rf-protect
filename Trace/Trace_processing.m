% Core Code of [RF-Protect, SIGCOMM'22] 
% This script takes the fmcw radar data, generates heatmaps and
% then conducts peak detection algorithm to get user trajectories based on these heatmaps. 
% 
% The generated trajectories are phantoms created by RF-Protect. The
% script will then compare these trajectories with the ground truth in
% terms of angle and distance accuracy.


%%
clear all
addpath './functions'
%% Parameters Setting

env = 1; % Choose the environments. 0: Home; 1: Office

% Put the Radar data path here
if env == 0
    usrp_data_path = '../Radar_data/trajs_from_usrp_Home';
    gt_data_path = '../Radar_data/trajs_gt';
else
    usrp_data_path = '../Radar_data/trajs_from_usrp_Office';
    gt_data_path = '../Radar_data/trajs_gt';
end

% whether displaying the heatmaps or not
SHOW_HMAP = 0;

% Trajectory Setting
step_size = 500*1e-5*5; % Time interval between two heatmaps
num_points = 500; % Number of steps for each trajectory

% Processing Parameters
search_range = 100; % The range to search for the start point of the trajectory

% Heatmap Plot Setting
htp_thre_top = 70;
htp_thre_bot = 25;

usrp_data_info = dir(usrp_data_path);
num_trajs = (length(usrp_data_info) - 2)/2;
%% Make directories to save data

mkdir('../traj_save/trajs_collected_Home/traj_collect/raw/');
mkdir('../traj_save/trajs_collected_Office/traj_collect/raw/');
%% Data Processing Loop

for i1=1:num_trajs

    name_channel0 = usrp_data_info(i1*2+1).name;
    name_channel1 = usrp_data_info(i1*2+2).name;
    name_split = strsplit(name_channel0,'_');
    traj_index = str2num(name_split{1});
    if traj_index>=1000
        continue
    end
    disp([i1, i1/num_trajs]);
    disp(name_channel0);

    [data0, s0] = read_complex_binary2([usrp_data_path, '/',name_channel0]);
    [data1, s1] = read_complex_binary2([usrp_data_path, '/',name_channel1]);
    [first_frame_index, points_frame] = fmcw_framz(data0, data1);
    [all_frames, ~] = fmcw_process_fast(first_frame_index, data0, data1);

    % Get the timestamp for each heatmap
    n_heatmaps = floor(length(points_frame)/(8*3));
    heatmap_times = zeros(n_heatmaps, 1);
    for i_h=1: n_heatmaps
        heatmap_times(i_h) = points_frame(i_h*8*3);
    end
    traj_collect = [];
    cord_pre = [-20, 0];
    
    % Iterate over every heatmap and do peak detection
    if SHOW_HMAP == 1
        figure;
    end
    tensor_counter = size(all_frames, 1);
    for i=1:1:tensor_counter
        mat = squeeze(all_frames(i,end-htp_thre_top:end-htp_thre_bot,:));
        mat = (mat - min(min(mat)))/(max(max(mat)) - min(min(mat)));
        p = PeakFind(mat, 1);
        if any(p)
        else
            continue
        end
        y = p(1);
        x = p(2);
        pos_d_max = find(y==max(y));
        if length(pos_d_max)>1
                len = length(x);
                M_value = [];
                for j=1:length(pos_d_max)
                    ind = pos_d_max(j);
                    M_value = [M_value, mat(y(ind), x(ind))];
                end
                max_M = max(M_value);
                poss = find(M_value==max_M);
        else
            poss = pos_d_max;
        end
        if SHOW_HMAP == 1
            imagesc(mat);
            hold on
            plot(x, y,'r+')
            title([i,max(max(squeeze(all_frames(i,end-htp_thre_top:end-htp_thre_bot,:))))])
            pause(0.01)
        end
        x_real = x(poss);
        y_real = y(poss); % x is angle, y is distance
        y_real = (htp_thre_top-htp_thre_bot-y_real(:,1)+20)*0.15-0.9;
        traj_collect = [traj_collect;[x_real, y_real]];
%         waitforbuttonpress
    end

    % Calibration and Saving
    traj_collect(:,1) = 180-traj_collect(:,1)-101;
    traj_collect(:,2) = traj_collect(:,2) - 2.35;
    if env == 0
        save(['../traj_save/trajs_collected_Home/traj_collect/',num2str(traj_index),'.mat'], 'traj_collect');
        save(['../traj_save/trajs_collected_Home/traj_collect/',num2str(traj_index),'_time.mat'], 'heatmap_times');
    else
        save(['../traj_save/trajs_collected_Office/traj_collect/',num2str(traj_index),'.mat'], 'traj_collect');
        save(['../traj_save/trajs_collected_Office/traj_collect/',num2str(traj_index),'_time.mat'], 'heatmap_times');
    end
    %% Process and Plot Trajecotory Angle and Distance
    % Import Groundtruth Trajectory
    d_gt = load([gt_data_path, '/ad_raw/', num2str(traj_index), '.mat']).traj_angle_dist;
    d_gt(:,2) = d_gt(:,2)/pi *180;

    % find the start time using brute force
    error_s = 100000;
    n_f_heatmaps = ceil(12.5/((heatmap_times(end) - heatmap_times(1))/length(heatmap_times)));
    for delta=-search_range:search_range
        start_index = 130+delta; 
        end_index = start_index+n_f_heatmaps; % we need 1023 heatmaps
        if end_index > size(heatmap_times, 1)
            break
        end
        traj1 = traj_collect(start_index:end_index,:);
        time_axis = heatmap_times(start_index:end_index)-heatmap_times(start_index);

        % interpolation
        xq = linspace(1, num_points, num_points)*step_size;
        traj1 = interp1(time_axis,traj1,xq);
        time_axis = xq;

        error = sum(abs(traj1(:,2) - d_gt(:,1)));
        if error < error_s
            final_index = start_index;
            error_s = error;
        end
    end
    error_angle = sum(abs(traj1(:,1) - d_gt(:,2)));
    disp([error_s/500, error_angle/500])
    start_index = final_index; 
    end_index = start_index+n_f_heatmaps; % we need 1023 heatmaps
    traj1 = traj_collect(start_index:end_index,:);
    time_axis = heatmap_times(start_index:end_index)-heatmap_times(start_index);

    % interpolation
    traj1 = interp1(time_axis,traj1,xq);
    time_axis = xq;


    % Plot raw angle and distance
    figure;
    subplot(2,1,1);
    plot(time_axis, traj1(:,2));
    hold on
    plot(time_axis, d_gt(:,1));
    xlabel('Time/seconds')
    title([num2str(traj_index),'\_Distance']);
    subplot(2,1,2);
    plot(time_axis, traj1(:,1));
    hold on
    plot(time_axis, d_gt(:,2));
    xlabel('Time/seconds')
    title([num2str(traj_index),'\_Angle']);
    if env == 0
        save(['../traj_save/trajs_collected_Home/raw/',num2str(traj_index),'.mat'], 'traj1');
    else
        save(['../traj_save/trajs_collected_Office/raw/',num2str(traj_index),'.mat'], 'traj1');
    end

end
function d=distance(cord_pre, cord_now)
    d = sqrt((cord_now(1)-cord_pre(1))^2 + (cord_now(2)-cord_pre(2))^2);
end