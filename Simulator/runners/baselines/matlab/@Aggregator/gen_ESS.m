function ESS = gen_ESS(obj,N)
    ESS = [];
    for i = 1:N
        ESS(i).P_chg = 10;
        ESS(i).SOC0 = 0.3+0.4*rand;
        ESS(i).B = 50+30*binornd(1,0.5);
        ESS(i).eta_chg = 0.92+0.05*rand;
    end
end