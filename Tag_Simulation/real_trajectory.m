function [data]=real_trajectory(index)
    traj_path = ['../../DATA/_DATA_4000_40/train/',num2str(index),'.mat'];
    data = load(traj_path);
    data = double(squeeze(data.data_save));
    data = [data,0.5*ones(length(data),1)];
end