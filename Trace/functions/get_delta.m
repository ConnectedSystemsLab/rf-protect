% function max_distance = get_delta(mat)
%     data = squeeze(mat);
%     
%     big_val = -10000;
%     
%     for i=2:length(mat)
%         delta = abs(data(i)-data(i-1));
%         if delta > big_val
%             big_val = delta;
%         end
%     end
%     max_distance = big_val;
% end
function max_distance = get_delta(mat)
    data = squeeze(mat);
    
    big_val = 0;
    
    for i=2:length(mat)
        delta = abs(data(i)-data(i-1));
        big_val = big_val + delta;
    end
    max_distance = big_val;
end