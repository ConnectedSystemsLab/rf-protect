function [data]=gan_trajectory(index)
    traj_path = ['./data/',num2str(index),'.mat'];
    data = load(traj_path);
    data = double(squeeze(data.data));
    data = [data,0.5*ones(length(data),1)];
end