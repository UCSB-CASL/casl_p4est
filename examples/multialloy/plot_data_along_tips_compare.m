clear;

x_start = 0.1;
x_max = 2;

dir = {'/home/dbochkov/dendrites1/d1_0.00001_g_00500/67/', ...
    '/home/dbochkov/dendrites1/d1_0.00005_g_00500/82/', ...
    '/home/dbochkov/dendrites1/d1_0.00010_g_00500/82/', ...
    '/home/dbochkov/dendrites1/d1_0.00050_g_00500/79/'};

% dir = {'/home/dbochkov/Dendrites/2/d1_0.00001_g_00500/72/', ...
%     '/home/dbochkov/Dendrites/2/d1_0.00002_g_00500/72/', ...
%     '/home/dbochkov/Dendrites/2/d1_0.00004_g_00500/72/', ...
%     '/home/dbochkov/Dendrites/2/d1_0.00008_g_00500/72/', ...
%     '/home/dbochkov/Dendrites/2/d1_0.00016_g_00500/72/', ...
%     '/home/dbochkov/Dendrites/2/d1_0.00032_g_00500/72/'};

% dir = {'/home/dbochkov/Dendrites/3/data_53/D1_0.00001_G_00500.1964881/dendrites/00053/', ...
%     '/home/dbochkov/Dendrites/3/data_53/D1_0.00002_G_00500.1964882/dendrites/00053/', ...
%     '/home/dbochkov/Dendrites/3/data_53/D1_0.00004_G_00500.1964883/dendrites/00053/', ...
%     '/home/dbochkov/Dendrites/3/data_53/D1_0.00008_G_00500.1964884/dendrites/00053/', ...
%     '/home/dbochkov/Dendrites/3/data_53/D1_0.00016_G_00500.1964885/dendrites/00053/', ...
%     '/home/dbochkov/Dendrites/3/data_53/D1_0.00032_G_00500.1964886/dendrites/00053/'};

% dir = {'/home/dbochkov/Dendrites/3/data_last/D1_0.00001_G_00500.1964881/dendrites/00053/', ...
%     '/home/dbochkov/Dendrites/3/data_last/D1_0.00002_G_00500.1964882/dendrites/00060/', ...
%     '/home/dbochkov/Dendrites/3/data_last/D1_0.00004_G_00500.1964883/dendrites/00066/', ...
%     '/home/dbochkov/Dendrites/3/data_last/D1_0.00008_G_00500.1964884/dendrites/00070/', ...
%     '/home/dbochkov/Dendrites/3/data_last/D1_0.00016_G_00500.1964885/dendrites/00067/', ...
%     '/home/dbochkov/Dendrites/3/data_last/D1_0.00032_G_00500.1964886/dendrites/00066/'};

% dir = {'/home/dbochkov/Dendrites/4/data_115/D1_0.00001_G_00500.1972666/dendrites/00115/', ...
%     '/home/dbochkov/Dendrites/4/data_115/D1_0.00002_G_00500.1972671/dendrites/00115/', ...
%     '/home/dbochkov/Dendrites/4/data_115/D1_0.00004_G_00500.1972676/dendrites/00115/', ...
%     '/home/dbochkov/Dendrites/4/data_115/D1_0.00008_G_00500.1972679/dendrites/00115/', ...
%     '/home/dbochkov/Dendrites/4/data_115/D1_0.00016_G_00500.1972680/dendrites/00115/', ...
%     '/home/dbochkov/Dendrites/4/data_115/D1_0.00032_G_00500.1972681/dendrites/00115/'};

% dir = {'/home/dbochkov/Dendrites/4/data_last/D1_0.00001_G_00500.1972666/dendrites/00115/', ...
%     '/home/dbochkov/Dendrites/4/data_last/D1_0.00002_G_00500.1972671/dendrites/00129/', ...
%     '/home/dbochkov/Dendrites/4/data_last/D1_0.00004_G_00500.1972676/dendrites/00141/', ...
%     '/home/dbochkov/Dendrites/4/data_last/D1_0.00008_G_00500.1972679/dendrites/00149/', ...
%     '/home/dbochkov/Dendrites/4/data_last/D1_0.00016_G_00500.1972680/dendrites/00149/', ...
%     '/home/dbochkov/Dendrites/4/data_last/D1_0.00032_G_00500.1972681/dendrites/00146/'};

case_name = {'D1/D0 = 1', ...
    'D1/D0 = 2', ...
    'D1/D0 = 4', ...
    'D1/D0 = 8', ...
    'D1/D0 = 16', ...
    'D1/D0 = 32'};

Tm = 1996;
m0 =-874;
m1 =-1378;
k0 = 0.848;
k1 = 0.848;

liq_rez = 100;

colors = lines(16);
markers = {'+','o','*','.','x','s','d','^','v','>','<','p','h'};

fig_c = figure;
fig_t = figure;
fig_l = figure;
fig_p = figure;

for I = 1:length(dir)
    
    n = 1;
    
    phi   = importdata(strcat(dir{I},'/phi.txt'));
    c0    = importdata(strcat(dir{I},'/c0.txt'));
    c1    = importdata(strcat(dir{I},'/c1.txt'));
    t     = importdata(strcat(dir{I},'/t.txt'));
    vn    = importdata(strcat(dir{I},'/vn.txt'));
    c0s   = importdata(strcat(dir{I},'/c0s.txt'));
    c1s   = importdata(strcat(dir{I},'/c1s.txt'));
    tf    = importdata(strcat(dir{I},'/tf.txt'));
    kappa = importdata(strcat(dir{I},'/kappa.txt'));
    velo  = importdata(strcat(dir{I},'/velo.txt'));
    
    nx = length(phi(1,:));
    
    n = min(n,length(phi(:,1)));
    
    x = linspace(0, x_max, nx);
    
    solid  = phi > 0;
    liquid = phi < 0;
    
    sorting = sortrows([sum(solid'); 1:length(solid(:,1))]', 'descend');
    d = sorting(1,2);
    
    c0 = c0./liquid;
    c1 = c1./liquid;
    
    c0s = c0s./solid;
    c1s = c1s./solid;
    
    tf  = tf./solid;
    
    figure(fig_c);
    
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor', 'Color', 'LineStyle'};
    PropValue = {markers{I}, 1, 2, 'auto', colors(I,:), 'none'};
    
    hold on
    L = plot(x, c0(d,:), 'DisplayName', case_name{I}); set(L, PropName, PropValue);
    L = plot(x, c1(d,:),'HandleVisibility','off'); set(L, PropName, PropValue);
    L = plot(x, c0s(d,:)./k0,'HandleVisibility','off'); set(L, PropName, PropValue);
    L = plot(x, c1s(d,:)./k0,'HandleVisibility','off'); set(L, PropName, PropValue);
    hold off
    
    legend;
    xlabel('Distance');
    ylabel('Concentration');
    grid on
    
    figure(fig_t);
    
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor', 'Color', 'LineStyle'};
    PropValue = {markers{I}, 1, 2, 'auto', colors(I,:), 'none'};
    
    hold on
    L = plot(x, tf(d,:), 'DisplayName', case_name{I}); set(L, PropName, PropValue);
    hold off
    
    xlabel('Distance');
    ylabel('Temperature');
    legend;
    grid on
    
    % plot crystallization paths on phase diagram
    figure(fig_l);
    
    if I == 1
        D = [];
        for i = 1:n
            D = [D, sorting(i,2)];
        end
        liq_c0 = linspace(min(min(c0s(D,:).*solid(D,:)))/k0, max(max(c0s(D,:).*solid(D,:)))/k0, liq_rez);
        liq_c1 = linspace(min(min(c1s(D,:).*solid(D,:)))/k1, max(max(c1s(D,:).*solid(D,:)))/k1, liq_rez+1);
        
        for i = 1:liq_rez
            for j = 1:liq_rez+1
                liq_tf(j,i) = Tm + m0*liq_c0(i) + m1*liq_c1(j);
            end
        end
        
        surf(liq_c0, liq_c1, liq_tf, 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Liquidus');
    end
    
    hold on
    plot3(c0s(d,:)/k0./(x>=x_start), c1s(d,:)/k1./(x>=x_start), tf(d,:)./(x>=x_start), 'LineStyle', '-', 'MarkerSize', 2, 'LineWidth', 1, 'Marker', markers{I}, 'DisplayName', case_name{I}, 'Color', colors(I,:));
    hold off
    
    grid on
    legend;
    xlabel('C_0');
    ylabel('C_1');
    zlabel('Temperature');
    
    figure(fig_p);
    
    
    hold on
    plot(c0s(d,:)/k0./(x>=x_start), c1s(d,:)/k1./(x>=x_start), 'LineStyle', '-', 'MarkerSize', 2, 'LineWidth', 1, 'Marker', markers{I}, 'DisplayName', case_name{I}, 'Color', colors(I,:));
    hold off
    
    grid on
    legend;
    xlabel('C_0');
    ylabel('C_1');
end