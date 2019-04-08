function filtered_signal = filter_signals(time, F, cutoff_frequency, starting_time)
    if(nargin < 4)
        starting_time = 0;
    end
    filtered_signal = F;
    start = 1;
    while time(start) < starting_time
        start = start +1;
    end
    integral = zeros(1, size(F, 2));
    filtered_signal(start:end, :) = exp(-cutoff_frequency*(time(start:end) - time(start)))*F(start, :);
    for k = start+1 : length(time)
        dt = time(k) - time(k-1);
        integral = exp(-cutoff_frequency*dt)*integral + ....
            + 0.5*(F(k, :) + F(k-1, :))*(1.0-exp(-cutoff_frequency*dt));
        filtered_signal(k, :) = filtered_signal(k, :) + integral;
    end
end