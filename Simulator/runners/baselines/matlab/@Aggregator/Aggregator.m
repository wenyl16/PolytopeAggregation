classdef Aggregator
    properties
        bus
        Eqps
        baseline
        model
        bounds
        templete_id;%1：我的fr1,2：我的fr2，
    end
    
    methods
        function obj = Aggregator(bus,nEV,nHVAC,PV_base,nESS,N_samples)
            obj.bus = bus;
            for i = 1:N_samples
                obj.Eqps(i).EV=obj.gen_EV(nEV);
                obj.Eqps(i).ESS=obj.gen_ESS(nESS);
                obj.Eqps(i).PV=obj.gen_PV(PV_base,i);
                obj.Eqps(i).HVAC=obj.gen_HVAC(nHVAC,i);
            end
        end
        function obj = model_agg_with_boundaries(obj,templete_id, boundaries)
            obj.templete_id = templete_id;
            global deltaT T;
            obj.bounds.pmin = boundaries.pmin;
            obj.bounds.pmax = boundaries.pmax;
            obj.bounds.emin = boundaries.emin;
            obj.bounds.emax = boundaries.emax;
            if templete_id<=2
                [A,b,fs,solvertime]=obj.innerapprox_mine(templete_id,obj.bounds);
                obj.model.errors = fs(2:end);
                obj.model.solvertime = solvertime;
            elseif templete_id==3
                [A,b,solvertime]=obj.cal_AFRwang(obj.bounds);
                obj.model.solvertime = solvertime;
            elseif templete_id == 4 
                [A,b]=obj.cal_AFRbox(obj.bounds);
            end
            obj.model.A = A;
            obj.model.b = b;
        end
    end
end

