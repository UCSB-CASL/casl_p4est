clear;

out_dir_all = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/sphere/without_all/convergence';
out_dir_one = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/sphere/without_one/convergence';

h = importdata(strcat(out_dir_all,'/h.txt'));

order_all = 2;
order_one = 2;

num_xticks = 5;
num_yticks = 5;

q_max_isb_all = importdata(strcat(out_dir_all,'/q_max_isb.txt'));
q_max_isb_one = importdata(strcat(out_dir_one,'/q_max_isb.txt'));

figure;

a_all = max(q_max_isb_all./h.^order_all)/3;
a_one = min(q_max_isb_one./h.^order_one);

PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
PropValue = {'o', 1, 3, 'auto'};
    
L = loglog(h, q_max_isb_all); set(L, PropName, PropValue);
hold on 
L = loglog(h, q_max_isb_one); set(L, PropName, PropValue);
loglog(h, a_all*h.^order_all, '-k', 'LineWidth', 2);
% loglog(h, a_one*h.^order_one, '-k', 'LineWidth', 2);
hold off

grid on 

% axes
xlabel('Grid resolution');
ylabel('Integration error');
   
xmin = min(h);
xmax = max(h);

xrel = xmax/xmin;

xmin = xmin/xrel^0.1;
xmax = xmax*xrel^0.1;

ymin = min(min([q_max_isb_all; q_max_isb_one]));
ymax = max(max([q_max_isb_all; q_max_isb_one]));

yrel = ymax/ymin;

ymin = ymin/yrel^0.1;
ymax = ymax*yrel^0.1;

xlim([xmin, xmax]);
ylim([ymin, ymax]);

xticks(round(10.^linspace(log10(xmin), log10(xmax), num_xticks), 1, 'significant'));
yticks(10.^[floor(log10(ymin)):round((log10(ymax)-log10(ymin))/num_yticks):ceil(log10(ymax))]);

% legend
L = legend('Full area', 'One cell', '2nd order');

% figure size
fig = gcf;
fig.Units = 'inches';
fig.Position = [10 10 3 2.5];
    

out_dir_all = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/sphere/with_all/convergence';
out_dir_one = '/home/dbochkov/Dropbox/Docs/Papers/05_mls_sc_poisson_solver/data/integration/sphere/with_one/convergence';

order_all = 4;
order_one = 4;

q_max_isb_all = importdata(strcat(out_dir_all,'/q_max_isb.txt'));
q_max_isb_one = importdata(strcat(out_dir_one,'/q_max_isb.txt'));

figure;

a_all = max(q_max_isb_all./h.^order_all);
a_one = min (q_max_isb_one./h.^order_one);

PropName  = {'Marker', 'LineWidth', 'MarkerSize', 'MarkerFaceColor'};
PropValue = {'o', 1, 3, 'auto'};

L = loglog(h, q_max_isb_all); set(L, PropName, PropValue);
hold on 
L = loglog(h, q_max_isb_one); set(L, PropName, PropValue);
loglog(h, a_all*h.^order_all, '-k', 'LineWidth', 2);
hold off

grid on 

% axes
xlabel('Grid resolution');
ylabel('Integration error');

xmin = min(h);
xmax = max(h);

xrel = xmax/xmin;

xmin = xmin/xrel^0.1;
xmax = xmax*xrel^0.1;

ymin = min(min([q_max_isb_all; q_max_isb_one]));
ymax = max(max([q_max_isb_all; q_max_isb_one]));

yrel = ymax/ymin;

ymin = ymin/yrel^0.1;
ymax = ymax*yrel^0.1;

xlim([xmin, xmax]);
ylim([ymin, ymax]);

xticks(round(10.^linspace(log10(xmin), log10(xmax), num_xticks), 1, 'significant'));
yticks(10.^[floor(log10(ymin)):round((log10(ymax)-log10(ymin))/num_yticks):ceil(log10(ymax))]);

% legend
L = legend('Full area', 'One cell', '4th order');

% figure size
fig = gcf;
fig.Units = 'inches';
fig.Position = [10 10 3 2.5];