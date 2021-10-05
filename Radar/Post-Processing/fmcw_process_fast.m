function [fmcw_frames, tensor_counter] = fmcw_process_fast(first_frame, channel0,trigger)
n_frames = round(length(channel0)/1000)+100;
fft_len=951;
fmcw_frames=zeros(12000,fft_len,181);
num_antennas=8;
c0_f = zeros(n_frames,fft_len);
counter=1;
length(trigger);
sampling_rate = 2e6;

%%Obtain FMCW Frames
i=2;
while i<length(trigger)-3000
    %i;
    if trigger(i)<.5 && trigger(i-1)>.5
        c0_f(counter,:)=channel0(i+25:i+25+fft_len-1);
        i=i+1003;
        counter=counter+1;
    end
    i=i+1;
end


first_frame=first_frame+1;
antenna_counter=0;
tensor_counter=1;

avg=zeros(num_antennas,fft_len);
avg_fact=3;
avg_count=1;
sec_ct=0;
% 
c0_f = c0_f(first_frame:end,:);
d1 = floor(length(c0_f)/8);
c0_f = c0_f(1:d1*8,:);
avg_frames = reshape(c0_f,  8, [],951);
avg_frames = permute(avg_frames, [2,1,3]);
d_tmp = size(avg_frames);
d2 = d_tmp(1);
d2 = floor(d2/avg_fact);
avg_frames = avg_frames(1:d2*avg_fact,:,:);
avg_frames = reshape(avg_frames, avg_fact,d2, 8, 951);
avg_frames = squeeze(mean(avg_frames,1));


% for i=first_frame:8:length(c0_f)-8 %% Iterate over frames
%     
%     antenna_samples=c0_f(i:i+num_antennas-1,:); 
%     avg=avg+antenna_samples;
%     sec_ct=sec_ct+1;
%     if sec_ct==avg_fact
%         avg=avg/avg_fact;
%         avg_frames(avg_count,:,:)=avg;
%         avg=zeros(num_antennas,fft_len);
%         avg_count=avg_count+1;
%         sec_ct=0;
%     end
% end
 num_samp = fft_len;

prev_antenna_samples=zeros(num_antennas,fft_len);
%for i=first_frame:8:length(c0_f)-8 %% Iterate over frames
c=3e8;
phi = (0:1:180) * pi / 180;
lambda=(c/6e9);
lambda_step=((c/6e9)-(c/7e9))/1000;
lambda_vec=lambda:-lambda_step:lambda-lambda_step*num_samp;

cosp=cos(phi).';
aoa_vec_array = zeros(951, 181, 8);
for i1 = 1:951
    aoa_vec_array(i1,:,:) = exp(cosp* (-1i * (0:num_antennas-1) * (pi*2*0.023/lambda)));
    lambda=lambda-lambda_step;
end
for i=1:1:size(avg_frames,1)
    %antenna_samples=c0_f(i:i+num_antennas-1,:); 
    antenna_samples=squeeze(avg_frames(i,:,:));
    %back_sub=antenna_samples-squeeze(avg_frames(100,:,:));
    back_sub=antenna_samples-prev_antenna_samples;
    prev_antenna_samples=antenna_samples;
    antenna_samples=back_sub;
    %asamp=c0_f(i:i,:);
    %for j=1:1:num_antennas
    %    antenna_samples(j,:)=c0_f(i:i,:);
    %end
    phi_res = 1;
    phi = (0:1:180) * pi / 180;
    phi_size = length(phi);
    AoA_signal = zeros(num_samp, phi_size);
    

 %{   
for jj = 1:num_samp
    for i_phi = 1:phi_size
        this_phi = phi(i_phi);
        aoa_vec = exp(-1i * (1:7) * pi *2*.023* cos(this_phi)/lambda);
        AoA_signal(jj, i_phi) = sum(antenna_samples(:, jj).' .* aoa_vec);
    end
    lambda=lambda+lambda_step;
end
        %}
%antenna_samples = repmat(antenna_samples(5,:),7,1);
for jj = 1:num_samp
        aoa_vec = squeeze(aoa_vec_array(jj,:,:));
        temp_samp=antenna_samples(:, jj);
        tsum=aoa_vec * temp_samp;
        AoA_signal(jj,:)=tsum;
end
    

% compute range FFT 
AoA_FFT = abs(fft(AoA_signal, [], 1)).^2;

%imagesc(log(AoA_FFT));
%pause(1);
%c=3e8;
%chirp_BW=1e6;
%distance_step = c / (2 * chirp_BW);
%distance = 0:distance_step:(distance_step*(1001-1));
%rho=distance;
%phi=(0:1:180) * pi / 180;
%plot_2Dheatmap_sph(AoA_FFT,rho,phi);
%title(i);
fmcw_frames(tensor_counter,:,:)=AoA_FFT;
%if tensor_counter>2
%    fmcw_frames(tensor_counter,:,:)=fmcw_frames(tensor_counter,:,:)-fmcw_frames(tensor_counter-1,:,:);
%end
tensor_counter=tensor_counter+1;


%imagesc(log(AoA_FFT))
%title(i);
%pause(.1);
end

%%%%END FMCW HERE





%for i=2:1:tensor_counter
%    fmcw_frames(i,:,:)=fmcw_frames(i,:,:)-fmcw_frames(1,:,:);
%end

%{
c0_fft = fft(c0_f,1001,2);
c0_fft(:,502:end)=0;%
c0_ifft = ifft(c0_fft,1001,2);
complex_frame=c0_ifft;
avg = zeros(1, 1001);
ref_frame=zeros(1,1001);
    for i=2:1:length(complex_frame)
     if mod(i,3)==0
         fmcw_frames(i/3,:)=avg/3-ref_frame;
         ref_frame=avg/3;
         avg=complex_frame(i,:);
     else
         avg=avg+complex_frame(i,:);
     end
 end
%}
%fmcw_frames=c0_f;
end

