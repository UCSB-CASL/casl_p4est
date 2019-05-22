function [clipped_time, clipped_signals] = clip_and_shift_signals(time, signals, t_start, t_end)
    tstart = 1;
    while time(tstart) < t_start
        tstart = tstart+1;
    end
    tend = tstart;
    while time(tend) < t_end
        tend = tend+1;
    end
    clipped_time    = time(tstart:tend, :) -time(tstart);
    clipped_signals = signals(tstart:tend, :);
end