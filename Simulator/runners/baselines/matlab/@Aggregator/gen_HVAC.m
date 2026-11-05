function HVAC = gen_HVAC(obj,N,number_of_day)
    HVAC = [];
    global T;
    matlab_root = fileparts(fileparts(mfilename('fullpath')));
    source = load(fullfile(matlab_root, 'data', 'NYC_TMP.mat'), 'T_out');
    T_out = source.T_out;
    T_out = T_out(mod(number_of_day,27)+1,:)';
    T_out(T_out<=23) = 23;
    for i = 1:N
        HVAC(i).alpha = 0.15+0.02*rand;
        HVAC(i).beta = 0.9+0.2*rand;
        HVAC(i).P_r = 4+2*rand;
        HVAC(i).T0 = 20;
        HVAC(i).Tmax = 22;
        HVAC(i).Tmin = 18;
        HVAC(i).T_out = T_out;
    end
end
