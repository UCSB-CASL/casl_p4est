function plot_signals(time, F, xaxis_dim)
    if(nargin < 3)
        xaxis_dim = [0.0 400.0];
    end
    FONTSIZE = 32;

    figure('Units','normalized','Position',[0.01 0.35 0.45 0.6]);
    hold on
    plot(time, F(:, 1), 'b-', 'linewidth',3, 'Markersize', 10)
    xlabel('$t$', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
    % ylabel('$F_{x} = \frac{2}{\rho \pi r^{2} u_{0}^{2}} (\int_{\Gamma} (-p \mathbf{I} + 2 \mu \mathbf{D}) \cdot \mathbf{n} \, \mathrm{d}\Gamma) \cdot \mathbf{e}_{x} $', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
    ylabel('$F_{x}$', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
    grid on
    xticks = 0:50:400;
    set(gca,'fontsize',FONTSIZE, 'xtick', xticks);
    axis([xaxis_dim 0.52 0.75])

%     figure('Units','normalized','Position',[0.5 0.15 0.45 0.8]);
%     subplot(2, 1, 1)
%     hold on
%     plot(time, F(:, 2), 'b-', 'linewidth',3, 'Markersize', 10)
%     xlabel('$t$', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
%     % ylabel('$F_{y} = \frac{2}{\rho \pi r^{2} u_{0}^{2}} (\int_{\Gamma} (-p \mathbf{I} + 2 \mu \mathbf{D}) \cdot \mathbf{n} \, \mathrm{d}\Gamma) \cdot \mathbf{e}_{y} $', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
%     ylabel('$F_{y}$', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
%     grid on
%     xticks = 0:50:400;
%     set(gca,'fontsize',FONTSIZE, 'xtick', xticks);
%     axis([xaxis_dim -0.15 0.15])
%     
%     subplot(2, 1, 2)
%     hold on
%     plot(time, F(:, 3), 'b-', 'linewidth',3, 'Markersize', 10)
%     xlabel('$t$', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
%     % ylabel('$F_{z} = \frac{2}{\rho \pi r^{2} u_{0}^{2}} (\int_{\Gamma} (-p \mathbf{I} + 2 \mu \mathbf{D}) \cdot \mathbf{n} \, \mathrm{d}\Gamma) \cdot \mathbf{e}_{z} $', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
%     ylabel('$F_{z}$', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
%     grid on
%     xticks = 0:50:400;
%     set(gca,'fontsize',FONTSIZE, 'xtick', xticks);
%     axis([xaxis_dim -0.15 0.15])
    

    figure('Units','normalized','Position',[0.5 0.15 0.45 0.8]);
    hold on
    plot(time, sqrt(F(:, 2).^2 + F(:, 3).^2), 'b-', 'linewidth',3, 'Markersize', 10)
    xlabel('$t$', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
    ylabel('$F_{\mathrm{trans.}}$', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
    grid on
    xticks = 0:50:400;
    set(gca,'fontsize',FONTSIZE, 'xtick', xticks);
    axis([xaxis_dim 0.0 0.15])
end