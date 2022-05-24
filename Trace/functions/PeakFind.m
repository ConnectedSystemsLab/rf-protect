function  result=PeakFind(Mat, num)
    sz = size(Mat);
    Mat = medfilt2(Mat,[3,3]);
    m = sz(1);
    n = sz(2);
    kernel_size = 17;
    pad = (kernel_size-1)/2;
    M_pad = padarray(Mat, [pad, pad], nan, 'both');
%     M_pad = padarray(Mat, ((pad, pad), (pad, pad)), 'constant')
    pos = [];
    for i = 1:m
        for j = 1:n
            i_index = i + pad;
            j_index = j + pad;
            tmp = M_pad(i_index-pad:i_index+pad, j_index-pad:j_index+pad);
            if max(max(tmp)) == Mat(i, j)
                pos = [pos; [i, j, Mat(i, j)]];
            end
        end
    end
    if any(pos)
        pos_n = sortrows(pos, 3);
    
        sz2 = size(pos_n);
        if num <= sz2(1)
            result = pos_n(end-num+1:end,:,:); %for Circle micro on 8.31
%               result = sort(pos_n(end-9:end,:,:));
%               result = result(end, :, :);
        else
            result = [];
        end
    else
        result = [];
    end
    
end