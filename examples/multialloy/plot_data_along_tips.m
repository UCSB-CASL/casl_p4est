clear;

x_start = 0.1;
x_max = 4;
x_plot = 2;

dir = '/home/dbochkov/Output/multialloy/dendrites';
% dir = '/home/dbochkov/dendrites1/d1_0.00001_g_00500/67/';
% dir = '/home/dbochkov/dendrites1/d1_0.00005_g_00500/67/';
% dir = '/home/dbochkov/dendrites1/d1_0.00010_g_00500/67/';
% dir = '/home/dbochkov/dendrites1/d1_0.00050_g_00500/67/';

% dir = '/home/dbochkov/dendrites1/d1_0.00005_g_00500/82/';
% dir = '/home/dbochkov/dendrites1/d1_0.00010_g_00500/82/';
% dir = '/home/dbochkov/dendrites1/d1_0.00050_g_00500/79/';

% dir = '/home/dbochkov/Dendrites/2/d1_0.00001_g_00500/72/';
% dir = '/home/dbochkov/Dendrites/2/d1_0.00002_g_00500/72/';
% dir = '/home/dbochkov/Dendrites/2/d1_0.00004_g_00500/72/';
% dir = '/home/dbochkov/Dendrites/2/d1_0.00008_g_00500/72/';
% dir = '/home/dbochkov/Dendrites/2/d1_0.00016_g_00500/72/';
% dir = '/home/dbochkov/Dendrites/2/d1_0.00032_g_00500/72/';

% dir = '/home/dbochkov/Outputs/multialloy_optimized_shifted/dendrites/00083/';

% dir = '/home/dbochkov/Dendrites/3/data_last/D1_0.00001_G_00500.1964881/dendrites/00053/';
% dir = '/home/dbochkov/Dendrites/3/data_last/D1_0.00002_G_00500.1964882/dendrites/00060/';
% 
% 
% dir = '/home/dbochkov/Outputs/multialloy_optimized/dendrites/00118';
% dir = '/home/dbochkov/Outputs/multialloy_optimized/dendrites/00139';

% dir = '/home/dbochkov/Dendrites/4/data_last/D1_0.00001_G_00500.1972666/dendrites/00115/';
% dir = '/home/dbochkov/Dendrites/4/data_last/D1_0.00002_G_00500.1972671/dendrites/00129/';
% dir = '/home/dbochkov/Dendrites/4/data_last/D1_0.00004_G_00500.1972676/dendrites/00141/';
% dir = '/home/dbochkov/Dendrites/4/data_last/D1_0.00008_G_00500.1972679/dendrites/00149/';
% dir = '/home/dbochkov/Dendrites/4/data_last/D1_0.00016_G_00500.1972680/dendrites/00149/';
% dir = '/home/dbochkov/Dendrites/4/data_last/D1_0.00032_G_00500.1972681/dendrites/00146/';

% dir = '/home/dbochkov/Dendrites/4/data_115/D1_0.00001_G_00500.1972666/dendrites/00115/';
% dir = '/home/dbochkov/Dendrites/4/data_115/D1_0.00002_G_00500.1972671/dendrites/00115/';
% dir = '/home/dbochkov/Dendrites/4/data_115/D1_0.00004_G_00500.1972676/dendrites/00115/';
% dir = '/home/dbochkov/Dendrites/4/data_115/D1_0.00008_G_00500.1972679/dendrites/00115/';
% dir = '/home/dbochkov/Dendrites/4/data_115/D1_0.00016_G_00500.1972680/dendrites/00115/';
% dir = '/home/dbochkov/Dendrites/4/data_115/D1_0.00032_G_00500.1972681/dendrites/00115/';

% dir = '/home/dbochkov/Dendrites/5/data_380/D1_0.00001_G_00500.2173295/dendrites/00380/';
% dir = '/home/dbochkov/Dendrites/5/data_380/D1_0.00002_G_00500.2173296/dendrites/00380/';
% dir = '/home/dbochkov/Dendrites/5/data_380/D1_0.00004_G_00500.2173297/dendrites/00380/';
% dir = '/home/dbochkov/Dendrites/5/data_380/D1_0.00008_G_00500.2173298/dendrites/00380/';
% dir = '/home/dbochkov/Dendrites/5/data_380/D1_0.00016_G_00500.2173299/dendrites/00380/';
dir = '/home/dbochkov/Dendrites/5/data_380/D1_0.00032_G_00500.2173301/dendrites/00380/';


Tm = 1996;

m0 =-874;
m1 =-1378;
k0 = 0.848;
k1 = 0.848;

liq_rez = 100;

phi   = importdata(strcat(dir,'/phi.txt'));
c0    = importdata(strcat(dir,'/c0.txt'));
c1    = importdata(strcat(dir,'/c1.txt'));
t     = importdata(strcat(dir,'/t.txt'));
vn    = importdata(strcat(dir,'/vn.txt'));
c0s   = importdata(strcat(dir,'/c0s.txt'));
c1s   = importdata(strcat(dir,'/c1s.txt'));
tf    = importdata(strcat(dir,'/tf.txt'));
kappa = importdata(strcat(dir,'/kappa.txt'));
velo  = importdata(strcat(dir,'/velo.txt'));

nx = length(phi(1,:));

n = 3;
n = min(n,length(phi(:,1)));

x = linspace(0, x_max, nx);

solid  = phi > 0;
liquid = phi < 0;

sorting = sortrows([sum(solid'); 1:length(solid(:,1))]', 'descend');

c0 = c0./liquid;
c1 = c1./liquid;

c0s = c0s./solid;
c1s = c1s./solid;

tf  = tf./solid;


colors = lines(16);
markers = {'+','o','*','.','x','s','d','^','v','>','<','p','h'};

figure;
hold on
for i = 1:n
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor', 'Color', 'LineStyle'};
    PropValue = {markers{i}, 1, 3, 'auto', colors(i,:), 'none'};
    L = plot(x, c0(sorting(i,2),:), 'DisplayName', ['Dendrite ', num2str(i)]); set(L, PropName, PropValue);
%     if i == 1
%         hold on
%     end
    L = plot(x, c1(sorting(i,2),:),'HandleVisibility','off'); set(L, PropName, PropValue);
    L = plot(x, c0s(sorting(i,2),:)./k0,'HandleVisibility','off'); set(L, PropName, PropValue);
    L = plot(x, c1s(sorting(i,2),:)./k0,'HandleVisibility','off'); set(L, PropName, PropValue);
end
hold off
legend
xlabel('Distance');
ylabel('Concentration');
xlim([0, x_plot]);
grid on

figure;
for i = 1:n
    PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor', 'Color', 'LineStyle'};
    PropValue = {markers{i}, 1, 4, 'auto', colors(i,:), '-'};
    L = plot(x, tf(sorting(i,2),:), 'DisplayName', ['Dendrite ', num2str(i)]); set(L, PropName, PropValue);
    if i == 1
        hold on
    end
end
hold off
xlabel('Distance');
ylabel('Temperature');
xlim([0, x_plot]);
legend
grid on

% plot crystallization paths on phase diagram

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

figure
surf(liq_c0, liq_c1, liq_tf, 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Liquidus');

hold on

for i = 1:n
    
    d = sorting(i,2);
    
    path_c0 = [];
    path_c1 = [];
    path_tf = [];
    
    for j = 1:nx
        if x(j) > x_start && solid(d,j)
            path_c0 = [path_c0, c0s(d,j)];
            path_c1 = [path_c1, c1s(d,j)];
            path_tf = [path_tf, tf(d,j)];
        end
    end
    
    path_c0 = path_c0/k0;
    path_c1 = path_c1/k1;
    
%     plot3(path_c0, path_c1, path_tf, '-', 'MarkerSize', 2, 'LineWidth', 1)
    plot3(c0s(d,:)/k0, c1s(d,:)/k1, tf(d,:), '-', 'MarkerSize', 3, 'LineWidth', 1, 'Marker', markers{i}, 'DisplayName', ['Dendrite ', num2str(i)])
    
end

hold off
grid on
legend
xlabel('C_0');
ylabel('C_1');
zlabel('Temperature');


figure

hold on

for i = 1:n
    
    d = sorting(i,2);
    plot(c0s(d,:)/k0, c1s(d,:)/k1, '-', 'MarkerSize', 2, 'LineWidth', 1, 'Marker', markers{i}, 'DisplayName', ['Dendrite ', num2str(i)])
    
end

hold off

grid on
legend
xlabel('C_0');
ylabel('C_1');