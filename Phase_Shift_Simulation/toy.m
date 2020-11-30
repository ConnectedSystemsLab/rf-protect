c = 3e8;
fc = 5e9;
lambda = c/fc;


% Tx position
tx_pos = [[0;0],[0;lambda/2]];
rx_pos = [[.5;.5],[.5;.5+lambda/2]];
% Rx position, randomly selected with a fixed seperation of lambda/2
rx_pos1 = 50*rand(2,1)+1;
rotate = 360*rand();
rx_pos2 = [rx_pos1(1,1)+lambda/2*cosd(rotate);rx_pos1(2,1)+lambda/2*sind(rotate)];
%rx_pos = [rx_pos1,rx_pos2];

res = 0.1;
phi_array = (-180:res:180);
ang_array = zeros(1,length(phi_array));
ang_array2 = zeros(1,length(phi_array));
for i = 1:length(phi_array)
    phi = phi_array(1,i);
    H = zeros(2,2);
    H(1,1) = exp(-1i*2*pi/lambda*pdist([tx_pos(:,1)';rx_pos(:,1)'], 'euclidean'));
    H(1,2) = exp(-1i*2*pi/lambda*pdist([tx_pos(:,1)';rx_pos(:,2)'], 'euclidean'));
    H(2,1) = exp(-1i*2*pi/lambda*pdist([tx_pos(:,2)';rx_pos(:,1)'], 'euclidean')+1i*phi*pi/180);
    H(2,2) = exp(-1i*2*pi/lambda*pdist([tx_pos(:,2)';rx_pos(:,2)'], 'euclidean')+1i*phi*pi/180);

    h_r1 = H(1,1)+H(2,1);
    h_r2 = H(1,2)+H(2,2);
    ang_array(1,i) = acosd((angle(h_r2)-angle(h_r1))/pi);
    ang_array2(1,i)= unwrap(angle(h_r2/h_r1));
end
%plot(phi_array, ang_array);
%xlim([-180, 180]);
%xlabel('Phase shift');
%ylabel('Angel');
%title('Angel-Phase shift')
plot(phi_array,abs(h_r1));
%plot(phi_array,abs(h_r2));
%xlim([50, 55]);
xlabel('Phase shift');
ylabel('Angel');
title('Angel-Phase shift')

