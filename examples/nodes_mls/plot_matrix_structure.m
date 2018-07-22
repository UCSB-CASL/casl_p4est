clear;

dir = '/home/dbochkov/Outputs/poisson_nodes_mls/matrix';

run(strcat(dir, "/mat_0.m"));

for i = 1:length(zzz)
    zzz(i,1) = vec(zzz(i,1));
    zzz(i,2) = vec(zzz(i,2));
end

mat = spconvert(zzz);
spy(mat,5);

xlabel('');

fig = gcf;
fig.Units = 'inches';
fig.Position = [10 10 3 2.5];