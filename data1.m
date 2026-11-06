S = load('Training_DataSet/TrainingData_Engine.mat');
runs = S.Train_EngineRun;               % 260x1 cell, each a struct with .Data and .Regime
vars = cellstr(string(S.Variable_List)); % 24 column names

T = cell(numel(runs), 1);
for k = 1:numel(runs)
    d = runs{k}.Data;                    % N x 24
    n = size(d, 1);

    t = array2table(d, 'VariableNames', vars);
    t = addvars(t, repmat(k, n, 1), (1:n)', runs{k}.Regime(:), ...
        'Before', 1, 'NewVariableNames', {'EngineID', 'Cycle', 'Regime'});
    T{k} = t;
end

out = vertcat(T{:});
writetable(out, 'TrainingData_Engine.csv', 'Encoding', 'UTF-8');
fprintf('Wrote %d rows x %d columns\n', height(out), width(out));