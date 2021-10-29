% this one is good for traj(:,1)
function traj = traj_smooth2(data, thresh)
%     figure(1);
%     plot(data);
    
%     figure(2);
%     data = medfilt1(data, 5);
    data = smoothdata(data, 'movmedian', 10);
%     plot(data)
%     
%     figure(3);
%     w = gausswin(5);
%     data = filter(w,1,data);
    data = smoothdata(data, 'gaussian');
%     plot(data)
    
    last_val = median(data(1:10));
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
%     figure(4);
%     plot(traj)
    
    traj = kalman_filter(traj);
    traj = traj(2:end);
%     figure(5);
%     plot(traj)
end

% this one is good for traj(:,1)
% function traj = traj_smooth2(data, thresh)
% %     figure(1);
%     plot(data);
%     
% %     figure(2);
% %     data = medfilt1(data, 5);
%     data = smoothdata(data, 'movmedian', 10);
%     plot(data)
% %     
% %     figure(3);
% %     w = gausswin(5);
% %     data = filter(w,1,data);
%     data = smoothdata(data, 'gaussian');
%     plot(data)
%     
%     last_val = median(data(1:10));
%     index = 1;
%     for i=2:length(data)
%         if abs(data(i)-last_val) < thresh
%             traj(index) = data(i);
%             last_val = data(i);
%             index = index+1;
%         else
%             traj(index) = last_val;
%             index = index + 1;
%         end
%     end
% %     figure(4);
% %     plot(traj)
%     
%     traj = kalman_filter(traj);
%     traj = traj(2:end);
% %     figure(5);
% %     plot(traj)
% end