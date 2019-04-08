function [duplicate_time, duplicate_signals] = duplicate_signals(time, signals)
    duplicate_time = [time; time+time(end) + 0.5*(time(2)-time(1) + time(end)-time(end-1))];
    duplicate_signals = [signals; signals];
end