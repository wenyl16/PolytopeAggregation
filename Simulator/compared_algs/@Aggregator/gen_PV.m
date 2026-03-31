function PV = gen_PV(obj,PV_base,number_of_day)
    PV = [];
    if PV_base ~=0
        load PV_gen;
        PV.Pmax = PV_base*PV_samples(:,mod(number_of_day,10)+1);
        PV.Pmin = zeros(length(PV.Pmax),1);
    end
end