clear;

dir = '/home/dbochkov/Dropbox/Docs/Papers/09_jump_solver/v0.1/data/matrix/matrix_cont';

run(strcat(dir, "/mat_0.m"));

for i = 1:length(zzz)
    zzz(i,1) = vec(zzz(i,1));
    zzz(i,2) = vec(zzz(i,2));
    zzz(i,3) = 1;
end

mat_c = spconvert(zzz);

dir = '/home/dbochkov/Dropbox/Docs/Papers/09_jump_solver/v0.1/data/matrix/matrix_disc';

run(strcat(dir, "/mat_0.m"));

for i = 1:length(zzz)
    zzz(i,1) = vec(zzz(i,1));
    zzz(i,2) = vec(zzz(i,2));
    zzz(i,3) = 1;
end

mat_d = spconvert(zzz)-mat_c;

spy(mat_d, '.r', 4);
hold on
spy(mat_c, '.k', 4);
hold off

xlabel('');

fig = gcf;
fig.Units = 'inches';
fig.Position = [10 10 3 2.5];