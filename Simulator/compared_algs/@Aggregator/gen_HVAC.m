function HVAC = gen_HVAC(obj,N,number_of_day)
    HVAC = [];
    global T;
    load NYC_TMP;%温度数据之后再改
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