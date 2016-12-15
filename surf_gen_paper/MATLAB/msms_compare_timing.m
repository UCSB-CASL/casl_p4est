clear all;


lmin=[4,4,5,5,5];
lmax=[7,8,9,10,11];
mols=['6rxn'; '1d65'; '2err'; '1aa2'; '2x6a'; '3M3D'; '1mah'; '1fss'; '1OED'; '3ecd'; '1JB0'; '3J6D'];
atoms=[667, 760, 1638, 1755, 4294, 8346, 9133, 9252, 10245, 23468, 34720, 131664];
  

times_fast = zeros(length(atoms), length(lmin));


%%Data from msms_compare.7921000.txt

times_fast(:,1)=[0.342, 0.518, 0.295, 0.473, 0.291, 0.406, 0.323, 0.419, 0.413, 0.358, 0.414, 0.327];
nodes_fast(:,1)=[17510, 28855, 14701, 28491, 14174, 23451, 18263, 25324, 22540, 20233, 23486, 18438];
times_fast(:,2)=[0.808, 1.619, 0.702, 1.432, 0.658, 1.119, 0.827, 1.229, 1.098, 0.937, 1.110, 0.815];
nodes_fast(:,2)=[51880, 100808, 39530, 100898, 36538, 81040, 57253, 89915, 78741, 67744, 85724, 59042];
times_fast(:,3)=[3.272	6.25	2.611	6.286	2.458	4.835	3.403	5.287	4.681	3.828	4.773	3.288];
nodes_fast(:,3)=[232228	447596	174219	452635	161512	369077	260287	408499	395127	320408	445281	294663];
times_fast(:,4)=[10.982	20.836	9.072	22.4	8.297	21.024	14.081	23.685	21.029	16.163	21.687	13.764];
nodes_fast(:,4)=[879936	1735988	644392	2001660	580330	1582491	1060807	1771092	1813669	1361327	2117694	1465829];
times_fast(:,5)=[39.998	77.044	32.771	81.446	32.793	79.384	59.217	89.297	88.526	78.001	115.716	73.956];
nodes_fast(:,5)=[3493670 6806782	2771095	7648041	2575580	7998187	5610226	8985196	9257182	6514929	10240970	7766836];

area_fast(:,1) = [3186.766	2194.445	4254.071	4293.911	10212.342	13562.505	14238.882	15040.804	21841.541	34223.003	61530.231	216033.342];
area_fast(:,2) = [3476.496	2278.592	4643.455	4594.678	11519.834	15124.784	16066.594	16805.523	26104.266	38775.263	70818.872	261243.008];
area_fast(:,3) = [3588.238	2334.928	4880.658	4744.736	12355.79	16244.647	17541.792	17924.725	29090.801	42993.299	79017.092	305499.822];
area_fast(:,4) = [3663.6	2456.839	5029.152	5009.767	12907.532	16924.212	18573.954	18740.888	31009.942	46725.768	85462.103	344314.672];
area_fast(:,5) = [3756.771	2464.306	5441.541	5203.71	13314.989	18916.315	19540.439	21040.447	33422.91	49031.379	89356.528	373388.238];
area_msms = [3405.72 4065.94 5189.28 5389.36 13605.47 17493.79 18509.91 18581.67 32413.19 49791.05 89219.35 403729.06];
area_exact = [3573.8 3827.86 4917.44 5714.00 12969.68 16626.72 19431.54 19495.40 31013.07 47360.27 93297.92 384569.69];


times_msms = [0.07, 0.07 0.10, 0.10, 0.28, 0.38, 0.83, 0.41, 1.95, 2.89, 5.70, 29.74];  

rel_error_fast = area_fast;


for i=1:length(area_fast(1,:))
    rel_error_fast(:,i) = (abs(area_fast(:,i) - area_exact')./area_exact');
    average_error_fast(i) = mean(rel_error_fast(:,i));
end


rel_error_msms = abs(area_msms - area_exact)./area_exact; 
average_error_msms = mean(rel_error_msms);
%% plot!

figure(1);
semilogy(lmax,rel_error_fast(7,:),'b-x'); hold on;
semilogy(lmax, ones(length(lmax))*rel_error_msms(7),'-k'); hold off;
title('error in area for 1MAH protien')
xlabel('lmax');
ylabel('Error');
set(gca, 'XTick', [7 8 9 10 11]);
legend('fast','msms');
fixfig;

figure(2)
loglog(atoms, rel_error_fast(:,1),'r-'); hold on;
loglog(atoms, rel_error_fast(:,2),'g-.');
loglog(atoms, rel_error_fast(:,3),'b--');
loglog(atoms, rel_error_fast(:,4),'m--');
loglog(atoms, rel_error_fast(:,5),'k--');hold off;
title('Error in surface area')
xlabel('atoms');
ylabel('error in area for 1MAH protien');
legend('fast 4,7', 'fast 4,8', 'fast 5,9', 'fast 5,10', 'fast 5,11', 'msms');
fixfig;

%}
figure(4)
loglog(atoms, times_fast(:,1),'r-'); hold on;
loglog(atoms, times_fast(:,3),'g-.');
loglog(atoms, times_fast(:,5),'b--');
loglog(atoms, times_msms,'k-');hold off;

%loglog(atoms, times_msms,'k-'); 



legend('Fast 4,7', 'Fast 5,9', 'Fast 5,11', 'MSMS');
xlabel('number of atoms');
ylabel('time (s)');
fixfig();





