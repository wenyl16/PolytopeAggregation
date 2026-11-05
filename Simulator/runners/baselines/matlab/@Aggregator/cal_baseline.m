%% 计算基线
function BL = cal_baseline(obj)
    global deltaT T;
    N_samples = length(obj.Eqps);
    BL = zeros(T,N_samples);
    for k = 1:N_samples
    Eqps = obj.Eqps(k);
    for i = 1:length(Eqps.EV)
        charging_demand = Eqps.EV(i);
        ta = min(round(charging_demand.ta/deltaT)+1,T);
        td = min(round(charging_demand.td/deltaT)+1,T);
        SOCa = charging_demand.SOCa;
        SOCd = charging_demand.SOCd;
        SOCmax = charging_demand.SOCmax;
        P_chg = charging_demand.P_chg;
        eta_chg = charging_demand.eta_chg;
        B = charging_demand.B;
        
        P0 = zeros(T,1);
        t_chg= ceil((SOCd-SOCa)*B/(P_chg*eta_chg*deltaT));
        P0(min(T,max(ta,1)):min(T,max(ta+t_chg-2,1)))=P_chg;
        P0(min(T,max(ta+t_chg-1,1)))=(SOCd-SOCa)*B/(eta_chg*deltaT) - (t_chg-1)*P_chg;
        BL(:,k) = BL(:,k) + P0;
    end
    
    for i = 1:length(Eqps.PV)
        PV = Eqps.PV(i);
        T0 = length(PV.Pmin);
        %P0 = -PV.Pmax(1:T0/T:T0);%以光伏最大出力为基线
        P0 = 0;%以光伏出力为0为基线
        BL(:,k) = BL(:,k)+P0;
    end
    
    for i = 1:length(Eqps.HVAC)
        
        HVAC = Eqps.HVAC(i);
        T0 = length(HVAC.T_out);
        P0 = HVAC.alpha*(HVAC.T_out(1:T0/T:T0)-HVAC.T0)/HVAC.beta;%算出来基准功率刚好没有越界
        P0 = max(P0,0);
        BL(:,k) = BL(:,k)+P0;
    end
    end
    
end
        