clear,clc,close all;
time_interval = 4000;
start_interval = 500;
sample = 40;

data_path = '../TrajectoryData/';
mkdir(['../DATA/_DATA_',num2str(time_interval),'_',num2str(sample),'/train/']);
mkdir(['../DATA/_DATA_',num2str(time_interval),'_',num2str(sample),'/test/']);
delete(['../DATA/_DATA_',num2str(time_interval),'_',num2str(sample),'/train/*.mat']);
delete(['../DATA/_DATA_',num2str(time_interval),'_',num2str(sample),'/test/*.mat']);

save_path1 = ['../DATA/_DATA_',num2str(time_interval),'_',num2str(sample),'/train/'];
save_path2 = ['../DATA/_DATA_',num2str(time_interval),'_',num2str(sample),'/test/'];

Files = dir(fullfile('../TrajectoryData/*.csv'));
LengthFiles = length(Files);

save_id1 = 1;
save_id2 = 1;

person_id = 0;
rangex = [];
rangey = [];
for id = 1:16
    
    M = csvread([data_path, Files(id).name]);
    max_person = max(M(:,5));
    
%     max_x = max(M(:,1));
%     min_x = min(M(:,1));
%     max_y = max(M(:,2));
%     min_y = min(M(:,2));
    mean_x = mean(M(:,1));
    mean_y = mean(M(:,2));
    for person_id = 0:max_person
        data_person = M(M(:,5)==person_id,1:2);

        num_trace = floor(length(data_person)/time_interval);
        if num_trace == 0
            continue
        end
        
%         for trace_id = 0:num_trace-1
        trace_id = 0;
        while true
%             data_save = data_person(trace_id*time_interval+1:(trace_id+1)*time_interval,1:2);
            if trace_id*start_interval+time_interval > length(data_person)
                break
            end
            data_save = data_person(trace_id*start_interval+1:trace_id*start_interval+time_interval,1:2);
            trace_id = trace_id + 1;
            max_x = max(data_save(:,1));
            min_x = min(data_save(:,1));
            max_y = max(data_save(:,2));
            min_y = min(data_save(:,2));
%             mean_x = mean(data_save(:,1));
%             mean_y = mean(data_save(:,2));
            if min([max_x-min_x,max_y-min_y])<0.001
                continue
            end
            data_save = data_save(1:sample:end, 1:2);
            data_save(:,1) = data_save(:,1)- mean_x;
            data_save(:,2) = data_save(:,2)- mean_y;
%             data_save(:,1) = 2*((data_save(:,1)-min_x)/(max_x-min_x))-1;
%             data_save(:,2) = 2*((data_save(:,2)-min_y)/(max_y-min_y))-1;
%             scatter(data_save(:,1),data_save(:,2),'k');

            p = rand;
            if p<0.000
                save([save_path2,num2str(save_id2),'.mat'],'data_save');
                save_id2 = save_id2 +1;
            else
                save([save_path1,num2str(save_id1),'.mat'],'data_save');
                save_id1 = save_id1 + 1;
            end
            
            rangex = [rangex,max_x-min_x];
            rangey = [rangey,max_y-min_y];
        end
    end
end

range = max(rangex,rangey);
h1 = cdfplot(range);
%     length_M = length(M);
%     num_pics = floor(length_M/time_interval);
%     max_M_x = max(M(:,1));
%     min_M_x = min(M(:,1));
%     max_M_y = max(M(:,2));
%     min_M_y = min(M(:,2));
%     
%     
%     for pic_id = 0:num_pics-1
%         h_fig = figure('Visible', 'off');
%         plot(M(pic_id*time_interval+1:(pic_id+1)*time_interval,1), M(pic_id*time_interval+1:(pic_id+1)*time_interval,2),'k');
%         axis([min_M_x max_M_x min_M_y max_M_y]); 
%         set(gca,'xtick',[],'xticklabel',[])
%         set(gca,'ytick',[],'yticklabel',[])
%         axis off
% %         set(gca,'visible','off')        
%         set(gca,'LooseInset',get(gca,'TightInset'));
%         pos = get(gcf, 'Position');
%         set(gcf, 'Position', [pos(1) pos(2) 528, 528]);
%         saveas(h_fig,[save_path,num2str(save_id),'.png']);
%         save_id = save_id + 1;
%     end
%     for i = 1:length(M)
%         plot(M(i,1),M(i,2),'r.');
%         hold on;
%         pause(0.01);
%     end


