% this one is good for traj(:,2)
function traj = traj_smooth(data, thresh)
    figure(1);
    plot(data);
    last_val = data(1);
    index = 1;
    for i=2:length(data)
        if abs(data(i)-last_val) < thresh
            traj(index) = data(i);
            last_val = data(i);
            index = index+1;
        else
            traj(index) = last_val;
            index = index + 1;
        end
    end
    figure(2);
    plot(traj);
    traj = kalman_filter(traj);
    traj = traj(2:end);
    figure(3);
    plot(traj);
end