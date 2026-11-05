function PV = gen_PV(obj,PV_base,number_of_day)
    PV = [];
    if PV_base ~=0
        matlab_root = fileparts(fileparts(mfilename('fullpath')));
        code_root = fileparts(fileparts(fileparts(fileparts(matlab_root))));
        source = load(fullfile(code_root, 'data', 'profiles_data', 'PV_samples.mat'), 'PV_samples');
        PV_samples = source.PV_samples;
        PV.Pmax = PV_base*PV_samples(:,mod(number_of_day,10)+1);
        PV.Pmin = zeros(length(PV.Pmax),1);
    end
end
