function [filtered_signal, avg] = filter_signals(time, F, cutoff_frequency, starting_avg)
    if(nargin < 4)
        starting_avg = 0;
    end
    avg             = zeros(1, 2);
    time_avg        = 0.0;
    integral        = zeros(1, size(F, 2));
    filtered_signal = exp(-cutoff_frequency*(time - time(1)))*F(1, :);
    for k = 2 : length(time)
        dt = time(k) - time(k-1);
        integral = exp(-cutoff_frequency*dt)*integral + ....
            + 0.5*(F(k, :) + F(k-1, :))*(1.0-exp(-cutoff_frequency*dt));
        filtered_signal(k, :) = filtered_signal(k, :) + integral;
        if((time(k) >= starting_avg) && (time(k-1) >= starting_avg))
            avg(1)         = avg(1) + 0.5*(time(k)-time(k-1))*F(k-1, 1);
            avg(1)         = avg(1) + 0.5*(time(k)-time(k-1))*F(k, 1);
            avg(2)         = avg(2) + 0.5*(time(k)-time(k-1))*sqrt(F(k-1, 2)^2 + F(k-1, 3)^2);
            avg(2)         = avg(2) + 0.5*(time(k)-time(k-1))*sqrt(F(k, 2)^2 + F(k, 3)^2);
            time_avg    = time_avg + (time(k)-time(k-1));
        end
    end
    avg = avg/time_avg;
end