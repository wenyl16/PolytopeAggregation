function [polytope, diagnostics] = run_baseline(template_id, boundary_file, output_file)
%RUN_BASELINE Compute the MATLAB inner approximation used for comparison.
%   run_baseline() uses template 1 and the shared T24 boundary data.
%   Templates 1--4 retain the original Aggregator implementations and solver.
%   Run after adding this directory to the MATLAB path.

    matlab_root = fileparts(mfilename('fullpath'));
    code_root = fileparts(fileparts(fileparts(fileparts(matlab_root))));
    addpath(matlab_root);
    if nargin < 1 || isempty(template_id)
        template_id = 1;
    end
    validateattributes(template_id, {'numeric'}, {'scalar', 'integer', '>=', 1, '<=', 4});
    if nargin < 2 || isempty(boundary_file)
        boundary_file = fullfile(matlab_root, 'data', 'boundaries.mat');
    end
    if nargin < 3 || isempty(output_file)
        result_dir = fullfile(code_root, 'results', 'aggregation', 'T24', 'baselines', 'matlab');
        if template_id == 1
            output_file = fullfile(result_dir, 'polytope_params.mat');
        else
            output_file = fullfile(result_dir, sprintf('polytope_template%d.mat', template_id));
        end
    end
    source = load(fullfile(matlab_root, 'data', 'Aggs_data.mat'), 'Aggs');
    Aggs = source.Aggs;
    diagnostics.template_id = template_id;
    diagnostics.boundary_file = boundary_file;
    diagnostics.algorithm = 'Original Aggregator inner approximation; no algorithm changes';
    diagnostics.aggregators = cell(size(Aggs));
    boundary_data = load(boundary_file, 'pmax', 'emax', 'pmin', 'emin');
    boundaries.pmax = boundary_data.pmax;
    boundaries.emax = boundary_data.emax;
    boundaries.pmin = boundary_data.pmin;
    boundaries.emin = boundary_data.emin;
    global deltaT T;
    deltaT = 1;
    T = 24 / deltaT;
    assert(size(boundaries.pmax, 1) == T, 'This MATLAB comparison uses the T24 case.');
    for i = 1:numel(Aggs)
        Aggs{i}.model = [];
        Aggs{i} = Aggs{i}.model_agg_with_boundaries(template_id, boundaries);
        if isfield(Aggs{i}.model, 'errors')
            diagnostics.aggregators{i}.errors = Aggs{i}.model.errors;
            diagnostics.aggregators{i}.iterations = numel(Aggs{i}.model.errors);
            diagnostics.aggregators{i}.converged = ~isempty(Aggs{i}.model.errors) && ...
                isfinite(Aggs{i}.model.errors(end)) && Aggs{i}.model.errors(end) <= 1e-4;
        end
        if isfield(Aggs{i}.model, 'solvertime')
            diagnostics.aggregators{i}.solvertime = Aggs{i}.model.solvertime;
        end
    end
    polytope.A = Aggs{1}.model.A * cumsum(eye(T)) * ...
                 diag(sum(boundaries.pmax, 2) - sum(boundaries.pmin, 2));
    polytope.b = Aggs{1}.model.b - Aggs{1}.model.A * ...
                 cumsum(eye(T)) * sum(boundaries.pmin, 2);
    output_dir = fileparts(output_file);
    if ~isempty(output_dir) && ~isfolder(output_dir)
        mkdir(output_dir);
    end
    diagnostics.finite_result = all(isfinite(polytope.A(:))) && all(isfinite(polytope.b(:)));
    save(output_file, 'polytope', 'diagnostics');
end
