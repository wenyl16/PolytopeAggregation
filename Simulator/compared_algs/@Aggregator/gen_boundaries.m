function [Pmin,Pmax,Emin,Emax] = gen_boundaries(obj,Eqps)
    global deltaT T;
    Pmin = [];
    Pmax = [];
    Emin = [];
    Emax = [];
    for i = 1:length(Eqps.EV)
        charging_demand = Eqps.EV(i);
        ta = floor(charging_demand.ta/deltaT)+1;
        td = min(T,floor(charging_demand.td/deltaT)+1);
        SOCa = charging_demand.SOCa;
        SOCd = charging_demand.SOCd;
        P_chg = charging_demand.P_chg;
        B = charging_demand.B;
        eta_chg = charging_demand.eta_chg;
        %UP = P_chg*deltaT*[zeros(1,ta-1),ones(1,min(T-ta,td-ta)),zeros(1,T-td)]';
        UP = zeros(T,1);
        UP(ta:td) = P_chg*deltaT;
        LP = zeros(T,1);
        LE = min([(SOCd-SOCa)*B/eta_chg*ones(1,T);max([zeros(1,T);P_chg*deltaT*(1-td:T-td)+(SOCd-SOCa)*B/eta_chg])])';
        UE = min([(1-SOCa)*B/eta_chg*ones(1,T);max([zeros(1,T);P_chg*deltaT*(2-ta:1+T-ta)])])';
        Pmin = [Pmin,LP];
        Pmax = [Pmax,UP];
        Emin = [Emin,LE];
        Emax = [Emax,max([UE';LE'])'];
    end
    
    for i = 1:length(Eqps.ESS)
        ESS = Eqps.ESS(i);
        E0 = ESS.SOC0*ESS.B;
        P_chg= ESS.P_chg;
        eta_chg = ESS.eta_chg;
        B= ESS.B;
        
        UP = P_chg*deltaT*ones(T,1);
        LP = -P_chg*deltaT*ones(T,1);
        LE = max([-P_chg*deltaT*[T-1:-1:0];
            -E0/eta_chg*ones(1,T);
            -P_chg*deltaT*[1:T]])'; 
        UE = min([P_chg*deltaT*[T-1:-1:0];
            (B-E0)/eta_chg*ones(1,T);
            P_chg*deltaT*[1:T]])'; 
        
        Pmin = [Pmin,LP];
        Pmax = [Pmax,UP];
        Emin = [Emin,LE];
        Emax = [Emax,UE];
    end
    
    for i = 1:length(Eqps.PV)
        PV = Eqps.PV(i);
        T0 = length(PV.Pmin);
        UP = -PV.Pmin(1:T0/T:T0)*deltaT;
        LP = -PV.Pmax(1:T0/T:T0)*deltaT;
        LE = -cumsum(PV.Pmax(1:T0/T:T0))*deltaT;
        UE = -cumsum(PV.Pmin(1:T0/T:T0))*deltaT;
        Pmin = [Pmin,LP];
        Pmax = [Pmax,UP];
        Emin = [Emin,LE];
        Emax = [Emax,UE];
    end
    
    for i = 1:length(Eqps.HVAC)
        HVAC = Eqps.HVAC(i);
        
        P_slowest = zeros(T,1);
        T_in = zeros(T+1,1);
        T_in(1) = HVAC.T0;
        T0 = length(HVAC.T_out);
        T_out = HVAC.T_out(1:T0/T:T0);
        for t = 1:T
            T_in(t+1) = T_in(t)+HVAC.alpha*deltaT*(T_out(t)-T_in(t));
            if T_in(t+1)>HVAC.Tmax
                P_slowest(t) = (T_in(t+1) - HVAC.Tmax)/(HVAC.beta*deltaT);
                T_in(t+1) = HVAC.Tmax;
            end
        end
        P_fastest = zeros(T,1);
        for t = 1:T
            T_in(t+1) = T_in(t)+HVAC.alpha*deltaT*(T_out(t)-T_in(t))-HVAC.beta*deltaT*HVAC.P_r;
            if T_in(t+1)<HVAC.Tmin
                P_fastest(t) = HVAC.P_r - (HVAC.Tmin - T_in(t+1))/(HVAC.beta*deltaT);
                T_in(t+1) = HVAC.Tmin;
            else
                P_fastest(t) = HVAC.P_r;
            end
        end
        
        
        UP = HVAC.P_r*deltaT*ones(T,1);
        LP = zeros(T,1);
        LE = cumsum(P_slowest)*deltaT; 
        UE = cumsum(max([P_fastest';P_slowest'])')*deltaT; 
        Pmin = [Pmin,LP];
        Pmax = [Pmax,UP];
        Emin = [Emin,LE];
        Emax = [Emax,UE];
    end
end