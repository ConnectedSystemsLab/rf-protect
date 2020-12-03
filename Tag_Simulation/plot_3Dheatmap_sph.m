function plot_3Dheatmap_sph(sph_pwr,rho,phi,theta) 
    figure;
    sph_pwr_front = squeeze(max(sph_pwr,[],1));
    sph_pwr_top = squeeze(max(sph_pwr,[],3));
    sph_pwr_side = squeeze(max(sph_pwr,[],2));

    clim = [0 1e6];
    pc_threshold = 1e5;

    subplot(2,2,1);
    imagesc(sph_pwr_front.'); 
    colormap jet; caxis(clim);
    %     pbaspect([kernel(1) kernel(2) 6])
    xlabel('phi'); ylabel('theta');
    set(gca,'XDir', 'reverse');

    subplot(2,2,2);
    %     imagesc(sph_pwr_side.');
    polarplot3d(sph_pwr_side,'radialrange',[min(rho) max(rho)],'angularrange',[min(theta) max(theta)],'polargrid',{rho theta},'GridColor','none');
    view([-90 90]);
    colormap jet; caxis(clim);
    %     pbaspect([kernel(1) kernel(2) 6])
    xlabel('theta'); ylabel('rho');
    %     set(gca, 'YDir', 'reverse', 'XDir', 'reverse');

    subplot(2,2,[3,4]);
    %     imagesc(sph_pwr_top); set(gca, 'YDir', 'normal', 'XDir', 'reverse');
    polarplot3d(sph_pwr_top,'radialrange',[min(rho) max(rho)],'angularrange',[min(phi) max(phi)],'polargrid',{rho phi},'GridColor','none');
    view(2);
    colormap jet; caxis(clim);
    %     pbaspect([kernel(1) kernel(3) 6]);
    xlabel('phi'); ylabel('rho');

    pc_idx = find(sph_pwr>pc_threshold);
    [rho_idx,phi_idx, theta_idx] = ind2sub([length(rho),length(phi),length(theta)],pc_idx);
    [x,y,z] = sph2cart(phi(phi_idx),pi/2-theta(theta_idx),rho(rho_idx));

    figure;
    pcshow([x.',y.',z.'],'MarkerSize',80);
    colormap jet;
    xlabel('x'); ylabel('y'); zlabel('z');
    xlim([-4,4]); ylim([3 15]); zlim([-1.5 1.5]);
    set(gca,'FontSize',30);
end