function plot_2Dheatmap_sph(sph_pwr,rho,phi) 
    figure;
    clim = [0 0.5];

    polarplot3d(sph_pwr,'radialrange',[min(rho) max(rho)],'angularrange',[min(phi) max(phi)],'polargrid',{rho phi},'GridColor','none');
    view(2);
    colormap jet; caxis(clim);
    xlabel('phi'); ylabel('rho');
    title('3: 2D Heatmap from 1D Array','FontSize',20)
end