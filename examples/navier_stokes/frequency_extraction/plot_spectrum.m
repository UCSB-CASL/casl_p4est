function plot_spectrum(fs, fft_spectrum)
    len = size(fft_spectrum, 1);
    one_sided_spectrum_magnitude = abs(fft_spectrum)/len;
    if(mod(len, 2) == 0)
        one_sided_spectrum_magnitude = one_sided_spectrum_magnitude(1:0.5*len+1, :);
        one_sided_spectrum_magnitude(2:end-1, :) = 2*one_sided_spectrum_magnitude(2:end-1, :);
        freq = fs*(0:(0.5*len))/len;
    else
        one_sided_spectrum_magnitude = one_sided_spectrum_magnitude(1:0.5*(len+1), :);
        one_sided_spectrum_magnitude(2:end, :) = 2*one_sided_spectrum_magnitude(2:end, :);
        freq = fs*(0:(0.5*(len-1)))/len;
    end
    FONTSIZE = 32;

    figure('Units','normalized','Position',[0.01 0.35 0.45 0.6]);
    hold on
    plot(freq, one_sided_spectrum_magnitude(:, 1), 'b-', 'linewidth',3, 'Markersize', 10)
    xlabel('freq. [Hz]', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
    ylabel('Amplitude of Fourier($F_{x}$)', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
    grid on
    xticks = 0:0.05:0.5;
    set(gca,'fontsize',FONTSIZE, 'xtick', xticks);
    axis([0 0.5 0.0 0.02])

    figure('Units','normalized','Position',[0.5 0.15 0.45 0.8]);
    hold on
    plot(freq, one_sided_spectrum_magnitude(:, 2), 'b-', 'linewidth',3, 'Markersize', 10)
    xlabel('freq. [Hz]', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
    ylabel('Amplitude of Fourier($F_{\mathrm{trans}}$)', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
    grid on
    xticks = 0:0.05:0.5;
    set(gca,'fontsize',FONTSIZE, 'xtick', xticks);
    axis([0 0.5 0.0 0.04])
    
%     subplot(2, 1, 2)
%     hold on
%     plot(freq, one_sided_spectrum_magnitude(:, 3), 'b-', 'linewidth',3, 'Markersize', 10)
%     xlabel('freq. [Hz]', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
%     ylabel('Amplitude of Fourier($F_{z}$)', 'fontsize',  FONTSIZE, 'Interpreter', 'Latex')
%     grid on
%     xticks = 0:0.05:0.5;
%     set(gca,'fontsize',FONTSIZE, 'xtick', xticks);
%     axis([0 0.5 0.0 0.04])
end