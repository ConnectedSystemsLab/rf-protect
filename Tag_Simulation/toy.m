c = 3e8;
fc = 5e9;
lambda = c/fc;

% Tx position
tx_pos = [[0;0],[0;lambda/2],[0;lambda]];
% rx_pos = [[0.5;2],[0.5;2+ lambda/2]];
% Rx position, randomly selected with a fixed seperation of lambda/2
rx_pos1 = 5*rand(2,1)+1;
rotate = 360*rand();
rx_pos2 = [rx_pos1(1,1)+lambda/2*cosd(rotate);rx_pos1(2,1)+lambda/2*sind(rotate)];
rx_pos = [rx_pos1,rx_pos2];

res = 1;
phi_array1 = (-180:res:180);
phi_array2 = (-180:res:180);
ang_array = zeros(length(phi_array1),length(phi_array1));
for i = 1:length(phi_array1)
    for j = 1:length(phi_array1)
        phi1 = phi_array1(1,i);
        phi2 = phi_array1(1,j);
        H = zeros(2,2);
        H(1,1) = exp(-1i*2*pi/lambda*pdist([tx_pos(:,1)';rx_pos(:,1)'], 'euclidean'));
        H(1,2) = exp(-1i*2*pi/lambda*pdist([tx_pos(:,1)';rx_pos(:,2)'], 'euclidean'));
        H(2,1) = exp(-1i*2*pi/lambda*pdist([tx_pos(:,2)';rx_pos(:,1)'], 'euclidean')+1i*phi1*pi/180);
        H(2,2) = exp(-1i*2*pi/lambda*pdist([tx_pos(:,2)';rx_pos(:,2)'], 'euclidean')+1i*phi1*pi/180);
        H(3,1) = exp(-1i*2*pi/lambda*pdist([tx_pos(:,3)';rx_pos(:,1)'], 'euclidean')+1i*phi2*pi/180);
        H(3,2) = exp(-1i*2*pi/lambda*pdist([tx_pos(:,3)';rx_pos(:,2)'], 'euclidean')+1i*phi2*pi/180);
        h_r1 = H(1,1)+H(2,1)+H(3,1);
        h_r2 = H(1,2)+H(2,2)+H(3,2);
    %     ang_array(1,i) = acos((angle(h_r2)-angle(h_r1))/pi);
    %     ang_array(1,i) = angle(h_r2/h_r1);
        ang_array(i,j) = angle(h_r2/h_r1);
    end
end
plot(phi_array1, phi_array1, ang_array);
% hold on;
% plot(phi_array, ang_array2);
% legend('h1','h2');
% xlabel('Phase shift');
% ylabel('Angel');