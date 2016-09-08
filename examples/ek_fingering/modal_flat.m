
dt = 1e-5;
it = 5;
modes = 0:20;
A = 0;
B = 0.0;
M = 0.1;
R = 9;
S = 10;
abs(A*B) < 1
abs(A*B*S^2/R/M) < 1
F = @(A,B,M,R,S) (M*((1-M)*(1+R)+2*A*(S-1))+abs(A*B)*(M^2-S^2))/(M*(1+M)*(1+R)-abs(A*B)*(M+S)^2);
G = @(A,B,M,R,S) (M*(1+R)-abs(A*B)*(M+S^2))/(M*(1+M)*(1+R)-abs(A*B)*(M+S)^2);
prefix = '/Users/mohammad/repos/parcasl/examples/ek_fingering/Release';
% prefix = '/mnt/server/code/parcasl/examples/ek_fingering/release';
path = sprintf('%s/two_fluid/flat/mue_0.1/2p',prefix);
% prefix = '/Users/mohammad/repos/casl/examples/viscous_fingering/Release';
% path = prefix;

styles = {'bo', 'rs', 'm<', 'k>'};
mcolor  ={'b','r','m','k'};
s = 1;
figure(1); hold on;
sigma = zeros(3,length(modes));
for lmax=[7:10]    
    for m=0:20
        file_base = sprintf('%s/err_%d_%d', path, lmax, m);
        
        err_nm2 = load(strcat(file_base,sprintf('_%d.txt',it-2)));
%         fft_nm2 = abs(fft(err_nm2(:,2) - mean(err_nm2(:,2))));
        fft_nm2 = abs(fft(err_nm2(:,2)));

        err_nm1 = load(strcat(file_base,sprintf('_%d.txt',it-1)));
%         fft_nm1 = abs(fft(err_nm1(:,2) - mean(err_nm1(:,2))));
        fft_nm1 = abs(fft(err_nm1(:,2)));

        err_n   = load(strcat(file_base,sprintf('_%d.txt',it)));
%         fft_n   = abs(fft(err_n(:,2) - mean(err_n(:,2))));
        fft_n   = abs(fft(err_n(:,2)));

        derr = (3*fft_n-4*fft_nm1+fft_nm2)/2/dt;
%         derr = (3*err_n(:,2)-4*err_nm1(:,2)+err_nm2(:,2))/2/dt;
%         derr = (fft_n - fft_nm1)/dt;

    %     s = mean(derr./fft_n);
        sigma(s,m+1) = derr(m+1)/fft_n(m+1);
%         sigma(s,m+1) = mean(derr./err_n(:,2));    
                
    % 
    %     figure(2+m);
    %     subplot(2,1,1);
    %     plot(err_n(:,1), err_n(:,2)); 
    % 
    %     subplot(2,1,2);
    %     modes = 0:25;
    %     stem(modes, abs(fft_n(modes+1)), 'linewidth', 1.5);
    end
    plot(0:20,sigma(s,:),styles{s}, 'markersize', 8,...
        'markerfacecolor',mcolor{s});
    s = s + 1;
end
%%
A = 0;
B = 0.;
M = 0.1;
R = 10;
S = 10;
% abs(A*B) < 1
% abs(A*B*S^2/R/M) < 1
% 
m = linspace(0,20);
Ca = 250;
plot(m,m*F(A,B,M,R,S)+m.*(-m.^2)*G(A,B,M,R,S)/Ca, 'k-', 'linewidth',2); shg

axis square;
set(gca, 'fontsize', 18);
xlabel('$m$', 'fontsize', 18, 'interpreter', 'latex');
ylabel('$\sigma_m$', 'fontsize', 18, 'interpreter', 'latex');

% legend(gca, {'$l_{max} = 10$','$l_{max} = 11$','$l_{max} = 12$', ...
%     'Theory'}, 'fontsize', 18, 'location', 'southwest', ...
%     'interpreter', 'latex');

% legend(gca, {'$M = 0.2$','$M = 1$','$M = 2$', ...
%     'Theory'}, 'fontsize', 18, 'location', 'southwest', ...
%     'interpreter', 'latex');

shg;
% ylim([-14 4]);
% xlim([0, 20]);
% print -depsc2 -f1 -r300 modal



