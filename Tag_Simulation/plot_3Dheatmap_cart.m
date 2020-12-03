function plot_3Dheatmap_cart(cart_pwr,X,Y,Z) 
    
    clim = [0 3e6];
    pc_threshold = 2e5;

    figure;
    cart_pwr_front = squeeze(max(cart_pwr,[],2));
    cart_pwr_top = squeeze(max(cart_pwr,[],3));
    
    % Front view
    subplot(2,2,1);
    imagesc(X,Z,cart_pwr_front.'); 
    colormap jet; caxis(clim);
    xlabel('X'); ylabel('Z');
    set(gca,'XDir', 'normal','YDir', 'normal');

    % Side view
    cart_pwr_side = squeeze(max(cart_pwr,[],1));
    subplot(2,2,3);
    imagesc(Y,Z,cart_pwr_side.');
    colormap jet; caxis(clim);
    xlabel('Y'); ylabel('Z');
    set(gca, 'YDir', 'normal', 'XDir', 'reverse');

    % Top view
    subplot(2,2,[2,4]);
    imagesc(X,Y,cart_pwr_top.'); set(gca, 'YDir', 'normal', 'XDir', 'normal');
    colormap jet; caxis(clim);
    xlabel('X'); ylabel('Y');
    
    % Point cloud
    pc_idx = find(cart_pwr>pc_threshold);
    [x_idx, y_idx, z_idx] = ind2sub([length(X),length(Y),length(Z)],pc_idx);

    figure;
    pcshow([X(x_idx),Y(y_idx),Z(z_idx)],'MarkerSize',80);
    colormap jet;
    xlabel('x'); ylabel('y'); zlabel('z');
    xlim([min(X),max(X)]); ylim([min(Y),max(Y)]); zlim([min(Z),max(Z)]);
    set(gca,'FontSize',30)
end
