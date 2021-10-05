function [first_frame_index] = fmcw_framz(channel0,trigger)
    n_frames = round(length(channel0)/1000)+100;
    n_frames;
    fft_len=951;
    fmcw_frames=zeros(5000,fft_len,181);
    c0_f = zeros(n_frames,fft_len);
    counter=1;
    secondcounter=1;
    length(trigger);
    %{
    i=100;
    while i<length(trigger)
        if real(trigger(i))<0
            idx_start=i;      
            %i
        if counter<=50000
            c0_f(counter,:)=channel0(idx_start:idx_start+1000);
            counter=counter+1;
            i=idx_start+5000;
        elseif idx_start>0 && counter>50000
            [v,t]=max(xcorr(c0_f(counter-1,:),channel0(idx_start:idx_start+1000)));
            t=t-1001;
            c0_f(counter,:)=channel0(idx_start+t:idx_start+1000+t);
            i=idx_start+5000;
            counter=counter+1;

        end
        end
        i=i+1;
    end
    %}
    %%Obtain FMCW Frames
    i=2;
    while i<length(trigger)-3000
        if trigger(i)<.5 && trigger(i-1)>.5
            c0_f(counter,:)=channel0(i+25:i+25+fft_len-1);
            i=i+1003;
            counter=counter+1;
            %plot(real(c0_f(counter-1,:)));
            %plot(db(fftshift(fft(real(c0_f(counter-1,:)-real(c0_f(10,:)))))));
            %xlim([400,600])
            %title(counter)
            %pause(.1);
        end
        i=i+1;
        %if secondcounter>1001
         %     counter=counter+1;
         %     secondcounter=1;
        %end
    end
    
    d_array = zeros(8,1);
    for j=1:1:8
        % plot(real(c0_f(j,:)))
         title(j)
        d_array(j) = get_delta(real(c0_f(j,:)));
        % pause(1)
%         pause(1)
    end
    first_frame_index = find(d_array==min(d_array));
    %fmcw_frames=c0_f;
end