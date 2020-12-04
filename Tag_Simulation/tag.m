clear all;
c =  physconst('LightSpeed');
fc = 77e9;                          % Operating freqency
range_max = 30;                    % maximum range estimation
tm = 5.5*range2time(range_max,c);   % sweep time
range_res = 0.1;                    % range resolution
bw = range2bw(range_res,c);         % bandwidth
num_antenna = 16;
sweep_slope = bw/tm;
fr_max = range2beat(range_max,sweep_slope,c);% the beat frequency corresponding to the maximum range

index_gan = 40;    % gan trajectory index
index_real = 1;   % real trajetory index

v_max = 2;                 % maximum speed of human movement
lambda = c/fc;       
fd_max = speed2dop(2*v_max,lambda);
fb_max = fr_max+fd_max;%maximum beat frequency.
fs = max(2*fb_max,bw);

waveform = phased.FMCWWaveform('SweepTime',tm,'SweepBandwidth',bw,'SampleRate',fs);
% subplot(211); plot(0:1/fs:tm-1/fs,real(sig));
% xlabel('Time (s)'); ylabel('Amplitude (v)');
% title('FMCW signal'); axis tight;
% subplot(212); spectrogram(sig,32,16,32,fs,'yaxis');
% title('FMCW signal spectrogram');
%% Target
car_dist = 43;
car_speed = 96*1000/3600;
% human_rcs = db2pow(-10);
human_rcs = db2pow(min(10*log10(car_dist)+5,20));

gan_traj = gan_trajectory(index_gan);
real_traj =  real_trajectory(index_real);

gan_traj = [linspace(0,198,100)', gan_traj];
real_traj = [linspace(0,198,100)', real_traj];

% real_traj(:,3) = real_traj(:,3)+5;

target = phased.RadarTarget('MeanRCS',human_rcs,'PropagationSpeed',c,...
    'OperatingFrequency',fc);
real_motion = phased.Platform('MotionModel','Custom','CustomTrajectory',real_traj);
gan_motion = phased.Platform('MotionModel','Custom', 'CustomTrajectory',gan_traj);

%% Radar System with mutliple antenna
ant_aperture = 6.06e-4;                         % in square meter
ant_gain = aperture2gain(ant_aperture,lambda);  % in dB
tx_ppower = db2pow(5)*1e-3;                     % in watts
tx_gain = 15+ant_gain;                           % in dB
rx_gain = 15+ant_gain;                          % in dB
rx_nf = 0;                                    % in dB

transmitter = phased.Transmitter('PeakPower',tx_ppower,'Gain',tx_gain);
antenna_tx = phased.IsotropicAntennaElement('FrequencyRange',[fc-bw fc+bw]);
radiator = phased.Radiator('PropagationSpeed',c,'OperatingFrequency',fc,...
    'Sensor',antenna_tx);

channel = phased.FreeSpace('PropagationSpeed',c,...
    'OperatingFrequency',fc,'SampleRate',fs, 'TwoWayPropagation',false);

antenna_rx = phased.ULA('NumElements',num_antenna, 'ElementSpacing',lambda/2);
collector = phased.Collector(...
    'PropagationSpeed',c,...
    'OperatingFrequency',fc,'Sensor',antenna_rx);
receiver = phased.ReceiverPreamp('Gain',rx_gain,'NoiseFigure',rx_nf,...
        'SampleRate',fs);
for i = 1:num_antenna
    receiver_array{i} = phased.ReceiverPreamp('Gain',rx_gain,'NoiseFigure',rx_nf,...
        'SampleRate',fs);
    radarmotion_array{i} = phased.Platform('InitialPosition',[(i-1)*lambda/2;0;0.5]);
end
%% Radar signal simulation
Nsweep = 64;
xr = complex(zeros(waveform.SampleRate*waveform.SweepTime,num_antenna));
transmitter_position = [(num_antenna-1)*lambda/4;0;0.5]; % transmitter is at the middle of the antenna array. 

for index =1:100
    [tgt_pos,tgt_vel] = real_motion(2);
%     tgt_pos = [real_traj(5,2:3),0.5]';
    [tgtrng,tgtang] = rangeangle(tgt_pos,transmitter_position)
    % Transmit one chirp
    sig = waveform();
    txsig = transmitter(sig);
    txsig = radiator(txsig,tgtang);
    txsig = channel(txsig,transmitter_position,tgt_pos,[0;0;0],[0;0;0]);
    txsig = target(txsig);
    txsig = channel(txsig,tgt_pos,transmitter_position,[0;0;0],[0;0;0]);
    txsig = collector(txsig,-tgtang);
    txsig = receiver(txsig);
    xr = dechirp(txsig,sig);
    % pdist([transmitter_position';tgt_pos'], 'euclidean')
    % angle_x(transmitter_position, tgt_pos) 
    % receive the signal from every antenna
    % for i = 1:num_antenna
    %     radarmotion = radarmotion_array{i};
    %     % Antenna doesn't change the position
    %     [radar_pos,radar_vel] = radarmotion(waveform.SweepTime);
    %     receiver = receiver_array{i};
    % 
    %     % Propagate the signal and reflect off the target
    %     txsig = channel(txsig,transmitter_position,tgt_pos,[0;0;0],[0;0;0]);
    %     txsig = target(txsig);
    %     txsig = channel(txsig,tgt_pos,radar_pos,[0;0;0],[0;0;0]);
    %     % Dechirp the received radar return
    %     txsig = receiver(txsig);
    %     dechirpsig = dechirp(txsig,sig);
    % 
    %     % Visualize the spectrum
    % %         specanalyzer([txsig dechirpsig]);
    % 
    %     xr(:,i) = dechirpsig;
    %     plot(0:1/fs:tm-1/fs,real(dechirpsig));
    % end

    %% Range and Doppler Estimation
    % antennaarray = phased.ULA('NumElements',4,'ElementSpacing',lambda/2);
    rngangresp = phased.RangeAngleResponse('SweepSlope',sweep_slope,...
        'SensorArray',antenna_rx,'OperatingFrequency',fc,...
        'SampleRate',fs,'RangeMethod','FFT','PropagationSpeed',c);
    [resp,rng_grid,ang_grid] = rngangresp(xr);
%     plotResponse(rngangresp,xr,'Unit','db');
%     axis([-90 90 0 range_max])
    resp = abs(resp);
    [a, b] = find(resp==max(max(resp)));
    range_estimate = rng_grid(a)
    angle_estimate = ang_grid(b)

    targetx = transmitter_position(1)+range_estimate*cosd(angle_estimate);
    targety = transmitter_position(2)+range_estimate*sind(angle_estimate);
    
    targetx1 = transmitter_position(1)+tgtrng*cosd(tgtang(1));
    targety1 = transmitter_position(2)+tgtrng*sind(tgtang(1));
    get_trajectory_real(index,:) = [targetx1, targety1];
    get_trajectory(index,:) = [targetx,targety];
end

% plot(real_traj(:,2),real_traj(:,3));
plot(get_trajectory_real(:,1),get_trajectory_real(:,2));
hold on
% get_trajectory(:,1) = optimal_filter(get_trajectory(:,1));
% get_trajectory(:,2) = optimal_filter(get_trajectory(:,2));
plot(get_trajectory(:,1),get_trajectory(:,2));
legend('real','real-detected');
% 
% rngangresp = phased.RangeAngleResponse('SweepSlope',sweep_slope,...
%     'SensorArray',antenna_rx,'OperatingFrequency',fc,...
%     'SampleRate',fs,'PropagationSpeed',c);
% [resp,rng_grid,ang_grid] = rngangresp(xr,[1;1]);
% plotResponse(rngangresp,xr,[1;2],'Unit','db');

% axis([-90 90 0 range_max])
% % beat_signal = abs(fft2(xr));
% % heatmap(beat_signal);
% % beat_signal = xr;
% % range_fft=fft(beat_signal(:,:));
% % rho = ((1:1:825)*fs/825)*3e8/(2*sweep_slope);
% % phi=(0:1:180)*pi/180;
% % N=4;
% % sph_pwr=[];
% % for theta=0:1:180
% %   power=0;
% %   for k=1:1:N
% %     phik=-pi*k*cosd(theta);
% %     power=power+beat_signal(:,k)*exp(1j*phik);
% %   end
% %   power=abs(fft(power));
% %   sph_pwr=[sph_pwr, [power(:)]];
% % end
% % %sph_pwr=transpose(sph_pwr)
% % % save the 2D radar heatmap as a 2D maxtrix 'sph_pwr' with the 1st
% % % dimension for rho and the 2nd dimension for phi
% % plot_2Dheatmap_sph(sph_pwr,rho,phi);
% rngdopresp = phased.RangeAngleResponse('PropagationSpeed',c,...
%     'OperatingFrequency',fc,'SampleRate',fs,...
%     'RangeMethod','FFT','SweepSlope',sweep_slope,...
%     'RangeFFTLengthSource','Property','RangeFFTLength',2048);
% mfcoeffs = [1;1];
% [resp,rng_grid,ang_grid] = rngdopresp(xr);
% clf;
% plotResponse(rngdopresp,xr);                     % Plot range Doppler map
% % axis([-v_max v_max 0 range_max])
% clim = caxis;