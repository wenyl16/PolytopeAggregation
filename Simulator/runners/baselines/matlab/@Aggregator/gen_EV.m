%% 生成电动汽车充电需求
function EV = gen_EV(obj,N)
    EV = [];
    k = 0;
    i = 0;
    P_chg = 7;
    B = 60;
    eta_chg=0.95;%充电效率
    while i <N
        p1=0.8;
        p = unifrnd(0,1);
        if p<=p1
            start_time=min(normrnd(19,2),22);
            end_time=max(24+7,normrnd(24+8,1.5));
            end_SOC=0.9+0.09*binornd(1,0.5);
            start_SOC=min(0.9,max(normrnd(0.5,0.2),0.3));
            if (end_SOC-start_SOC)*B/(end_time-start_time)>P_chg
                1
            end
                
            k = k + 1;
            EV(k).ta = start_time;
            EV(k).td = 24;
            EV(k).SOCa = start_SOC;
            EV(k).SOCd = (end_SOC-start_SOC)/(end_time-start_time)*(23.75-start_time)+start_SOC;
            EV(k).SOCmax = EV(k).SOCd;
            EV(k).P_chg = P_chg;
            EV(k).B = B;
            EV(k).eta_chg = eta_chg;
            
            k = k + 1;
            EV(k).ta = 0;
            EV(k).td = end_time-24;
            EV(k).SOCa = (end_SOC-start_SOC)/(end_time-start_time)*(23.75-start_time)+start_SOC;
            EV(k).SOCd = end_SOC;
            EV(k).SOCmax = 1;
            EV(k).P_chg = P_chg;
            EV(k).B = B;
            EV(k).eta_chg = eta_chg;
            
            i=i+1;
        else
            start_time=min(10,max(7,normrnd(8.5,1.5)));
            end_time=min(18,start_time+unifrnd(7,9));
            end_SOC=0.9+0.09*binornd(1,0.5);
            start_SOC=min(0.9,max(normrnd(0.6,0.2),0.2));
            if (end_SOC-start_SOC)*B/(end_time-start_time)>P_chg
                1
            end
            k = k + 1;
            EV(k).ta = start_time;
            EV(k).td = end_time;
            EV(k).SOCa = start_SOC;
            EV(k).SOCd = end_SOC;
            EV(k).SOCmax = 1;
            EV(k).P_chg = P_chg;
            EV(k).B = B;
            EV(k).eta_chg = eta_chg;
            
            i=i+1;
        end
    end
end