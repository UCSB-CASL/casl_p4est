clear;

x_start = 0.1;

dir = '/home/dbochkov/Output/multialloy/dendrites';

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

n = 6;
% n = length(phi(:,1));

x = linspace(0, 1, nx);

solid  = phi > 0;
liquid = phi < 0;

sorting = sortrows([sum(solid'); 1:length(solid(:,1))]', 'descend');

c0 = c0./liquid;
c1 = c1./liquid;

c0s = c0s./solid;
c1s = c1s./solid;

tf  = tf./solid;

plot(x, c0(sorting(1,2),:));
hold on
plot(x, c1(sorting(1,2),:));
plot(x, c0s(sorting(1,2),:));
plot(x, c1s(sorting(1,2),:));
for i = 2:n
    plot(x, c0(sorting(i,2),:));
    plot(x, c1(sorting(i,2),:));
    plot(x, c0s(sorting(i,2),:));
    plot(x, c1s(sorting(i,2),:));
end
hold off

figure;
plot(x, tf(sorting(1,2),:));
hold on
for i = 2:n
    plot(x, tf(sorting(i,2),:));
end
hold off

% plot crystallization paths on phase diagram

Tm = 1996;
m0 =-874;
m1 =-1378;
k0 = 0.848;
k1 = 0.848;

liq_rez = 100;

liq_c0 = linspace(min(min(c0s.*solid))/k0, max(max(c0s.*solid))/k0, liq_rez);
liq_c1 = linspace(min(min(c1s.*solid))/k1, max(max(c1s.*solid))/k1, liq_rez+1);


for i = 1:liq_rez
    for j = 1:liq_rez+1
        liq_tf(j,i) = Tm + m0*liq_c0(i) + m1*liq_c1(j);
    end
end

figure
surf(liq_c0, liq_c1, liq_tf, 'EdgeColor', 'none', 'FaceAlpha', 0.5);

hold on

for i = 1:n
    
    d = sorting(i,2);
    
    path_c0 = [];
    path_c1 = [];
    path_tf = [];
    
    for i = 1:nx
        if x(i) > x_start && solid(d,i)
            path_c0 = [path_c0, c0s(d,i)];
            path_c1 = [path_c1, c1s(d,i)];
            path_tf = [path_tf, tf(d,i)];
        end
    end
    
    path_c0 = path_c0/k0;
    path_c1 = path_c1/k1;
    
%     plot3(path_c0, path_c1, path_tf, '-', 'MarkerSize', 2, 'LineWidth', 1)
    plot3(c0s(d,:)/k0, c1s(d,:)/k1, tf(d,:), '-', 'MarkerSize', 2, 'LineWidth', 1)
    
end

hold off
grid on


figure

hold on

for i = 1:n
    
    d = sorting(i,2);
    plot(c0s(d,:)/k0, c1s(d,:)/k1, '-', 'MarkerSize', 2, 'LineWidth', 1)
    
end

hold off