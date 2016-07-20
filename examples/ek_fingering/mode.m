
dt = 1e-4;
it = 5;
modes = 0:20;
% prefix = '/Users/mohammad/repos/parcasl/examples/ek_fingering/Release/';
prefix = '/Users/mohammad/repos/casl/examples/viscous_fingering/Release';
% path = strcat(prefix,'two_fluid/circle/mue_0.01');
% path = strcat(prefix,'one_fluid/circle');
path = prefix;

close all;
styles = {'bo', 'rs', 'm<', 'k>'};
s = 1;
figure(1); hold on;
for lmax=11:11
    sigma = zeros(1,length(modes));
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
%         derr = (fft_n - fft_nm1)/dt;

    %     s = mean(derr./fft_n);
        sigma(m+1) = derr(m+1)/fft_n(m+1);
                
    % 
    %     figure(2+m);
    %     subplot(2,1,1);
    %     plot(err_n(:,1), err_n(:,2)); 
    % 
    %     subplot(2,1,2);
    %     modes = 0:25;
    %     stem(modes, abs(fft_n(modes+1)), 'linewidth', 1.5);
    end
    plot(0:20,sigma,styles{s});
    s = s + 1;
end
%%
m = linspace(0,20);
M = 0.;
Ca = 250;
plot(m,-1+m*(1-M)/(1+M)+m.*(1-m.^2)/(1+M)/Ca, 'k-', 'linewidth',2); shg

axis square;
set(gca, 'fontsize', 14);
xlabel('$m$', 'fontsize', 18, 'interpreter', 'latex');
ylabel('$\sigma_m$', 'fontsize', 18, 'interpreter', 'latex');

legend(gca, {'$l_{max} = 9$','$l_{max} = 10$','$l_{max} = 11$','$l_{max} = 12$', ...
    'Theory'}, 'fontsize', 14, 'location', 'southwest', ...
    'interpreter', 'latex');
shg;
ylim([-14 6]);
xlim([0, 20]);
print -depsc2 -f1 -r300 modal



