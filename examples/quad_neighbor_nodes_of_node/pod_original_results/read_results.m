function [exec_time, err_grad, err_second_derivatives] = read_results(filename)
    fid = fopen(filename, 'r');
    line = fgetl(fid);
    while(length(line) < 55 || ~strcmp(line(1:55), 'calculating gradient and second derivatives ... done in'))
        line = fgets(fid);
    end
    line = fgets(fid);
    exec_time = sscanf(line, ' %f secs. on process 0 [Note: only showing root''s timings]');
    
    while(length(line) < 28 || ~strcmp(line(1:28), 'The errors in gradient are: '))
        line = fgets(fid);
    end
    err_grad = [];
    while (length(line) > 1)
        tmp = sscanf(line, '%f')';
        err_grad = [err_grad; tmp];
        line = fgets(fid);
    end
    
    while(length(line) < 38 || ~strcmp(line(1:38), 'The errors in second derivatives are: '))
        line = fgets(fid);
    end
    err_second_derivatives = [];
    while (length(line) > 1)
        tmp = sscanf(line, '%f')';
        err_second_derivatives = [err_second_derivatives; tmp];
        line = fgets(fid);
    end
    fclose(fid);
    return;
end