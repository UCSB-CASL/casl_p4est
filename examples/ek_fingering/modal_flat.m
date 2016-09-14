
dt = 1e-5;
it = 5;
modes = 0:20;
A = -1;
B = 0.;
M = 0.01;
R = 100;
S = 100;
abs(A*B) < 1
abs(A*B*S^2/R/M) < 1
F = @(A,B,M,R,S) (M*((1-M)*(1+R)+2*A*(S-1))+abs(A*B)*(M^2-S^2))/(M*(1+M)*(1+R)-abs(A*B)*(M+S)^2);
G = @(A,B,M,R,S) (M*(1+R)-abs(A*B)*(M+S^2))/(M*(1+M)*(1+R)-abs(A*B)*(M+S)^2);
% prefix = '/Users/mohammad/repos/parcasl/examples/ek_fingering/Release';
prefix = '/mnt/server/code/parcasl/examples/ek_fingering/release';
path = sprintf('%s/coupled/flat/_Dirichlet_F_-0.960788_G_0.990099_A_-1_B_0_M_0.01_S_100_R_100',prefix);
% prefix = '/Users/mohammad/repos/casl/examples/viscous_fingering/Release';
% path = prefix;

styles = {'bo', 'rs', 'm<', 'k>'};
mcolor  ={'b','r','m','k'};
s = 1;
figure(1); hold on;
sigma = zeros(3,length(modes));
for lmax=[10]    
    for m=0:20
        file_base = sprintf('%s/err_%d_%d', path, lmax, m);
        err_nm2 = load(strcat(file_base,sprintf('_%d.txt',it-2)));
        
        fft_nm2 = abs(fft(err_nm2(:,2)));

        err_nm1 = load(strcat(file_base,sprintf('_%d.txt',it-1)));
        fft_nm1 = abs(fft(err_nm1(:,2)));

        err_n   = load(strcat(file_base,sprintf('_%d.txt',it)));
        fft_n   = abs(fft(err_n(:,2)));

        derr = (3*fft_n-4*fft_nm1+fft_nm2)/2/dt;

        sigma(s,m+1) = derr(m+1)/fft_n(m+1);
                
    end
    plot(0:20,sigma(s,:),styles{s}, 'markersize', 8,...
        'markerfacecolor',mcolor{s});
    s = s + 1;
end
%%

m = linspace(0,20);
Ca = 250;
plot(m,m*F(A,B,M,R,S)+m.*(-m.^2)*G(A,B,M,R,S)/Ca, 'k-', 'linewidth',2); shg

axis square;
set(gca, 'fontsize', 18);
xlabel('$m$', 'fontsize', 18, 'interpreter', 'latex');
ylabel('$\sigma_m$', 'fontsize', 18, 'interpreter', 'latex');