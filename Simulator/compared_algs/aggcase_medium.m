function aggcase = aggcase_medium()
    % bus nEV nHVAC PV_base/kW nESS
    aggcase.name = 'medium';
    buses = [2,4,7,8,11,12,14,16,18,21,24,25,26,29,30,32];
    % buses = [2,4,7,8];
    % eqps = [40,0,20,1];
    eqps = [20,0,0,0];
    aggcase.data=[buses',ones(size(buses')).*eqps];
    
end