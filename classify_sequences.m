% Classify Sequences
% Author: Gonçalo S. Martins
% This script takes a number of pre-clustered image sequences and builds
% the appropriate dictionaries, using the K-SVD toolbox <link goes here>.
%
% USAGE:
% Code sections are numbered for user convenience.

%% 1. Decompose Training Data
% Clear any previous training
clear dicts train_cluster_names test_cluster_names scale classification_results

% Load train clusters
% This script assumes the following folder structure:
%
% <base_dir>/<something>/1
%             ^         /2
%             |         /
%        One per seq.   ^
%                       | One per cluster (names should be 1, 2,
%                                                    etc.)
base_dir = 'sequences_cropped/train/' % base_dir must end in '/'
% Extract the folder names of the sequences
seqs = dir(base_dir);
seqs = seqs(3:end);

% Scale to resize the images we read
scale = 0.1;

% For each subfolder of the main data folder (sequence)
n = 1; % linear counter for clusters
for i = 1:length(seqs)
    % Determine the number of clusters
    n_clusters = dir(strcat(base_dir, seqs(i).name));
    n_clusters = length(n_clusters)-2;
    % For each cluster
    for j = 1:n_clusters
        % Determine the names of the images in the folder
        folder = strcat(base_dir, seqs(i).name, '/' ,int2str(j));
        files = dir(folder);
        files = files(3:end);
        % For each image in the cluster
        for k = 1:length(files)
            filename = strcat(folder, '/', files(k).name);
            train_clusters{n}{k} = imresize(rgb2gray(imread(filename)), scale);
        end
        % Write down the name of the training sequence
        train_cluster_names{n} = seqs(i).name;
        % Increment cluster counter
        n = n + 1;
    end 
end
fprintf('Loaded %d clusters from directory %s.', n-1, base_dir)

% From here on out, the cell array train_clusters will contain all of the
% training clusters. Each element of train_clusters is a cell array
% containing all of the images of that cluster, one per cell.

% Organize clusters into matrices
% Each column of the matrix will be a different image
train_mats = {};
for i = 1:length(train_clusters)
    train_mats{i} = [];
    for j = 1:length(train_clusters{i})
        train_mats{i} = [train_mats{i}, double(train_clusters{i}{j}(:))];
    end
end
%size(train_mats)

% Decompose training matrices
% Whoa, almost looks like a civilized dictionary:
disp('Obtaining dictionaries from training sequences')
params=struct('K', 4, ...
              'numIteration', 15, ... 
              'errorFlag', 0, ...
              'L', 10, ...
              'preserveDCAtom', 0, ...
              'InitializationMethod','DataElements', ...
              'displayProgress', 0);

dicts = {};
decomps = {};
for i = 1:length(train_mats)
    info = sprintf('Obtaining dictionary %d of %d.', i, length(train_mats));
    disp(info)
    [dicts{i}, temp] = KSVD(train_mats{i}, params);
    decomps{i} = temp.CoefMatrix;
end
disp('Training done!')

% Clean up unneded variables
clearvars -except dicts train_cluster_names scale

%% 2. Load Test Clusters
base_dir = 'sequences_cropped/test/' % base_dir must end in '/'
% Extract the folder names of the sequences
seqs = dir(base_dir);
seqs = seqs(3:end);

% For each subfolder of the main data folder (sequence)
n = 1; % linear counter for clusters
for i = 1:length(seqs)
    % Determine the number of clusters
    n_clusters = dir(strcat(base_dir, seqs(i).name));
    n_clusters = length(n_clusters)-2;
    % For each cluster
    for j = 1:n_clusters
        % Determine the names of the images in the folder
        folder = strcat(base_dir, seqs(i).name, '/' ,int2str(j));
        files = dir(folder);
        files = files(3:end);
        % For each image in the cluster
        for k = 1:length(files)
            filename = strcat(folder, '/', files(k).name);
            test_clusters{n}{k} = imresize(rgb2gray(imread(filename)), scale);
        end
        % Write down the name of the training sequence
        test_cluster_names{n} = seqs(i).name;
        disp(seqs(i).name)
        % Increment cluster counter
        n = n + 1;
    end 
end

% build test matrices
test_mats = {};
for i = 1:length(test_clusters)
    test_mats{i} = [];
    for j = 1:length(test_clusters{i})
        test_mats{i} = [test_mats{i}, double(test_clusters{i}{j}(:))];
    end
end

clearvars -except test_mats dicts train_cluster_names test_cluster_names scale

%% 3. Classify Test Clusters
% This section decomposes the test cluster into each of the dictionaries,
% and prints the residuals for each dictionary, as well as the dictionary
% that produced the least error.
% Decompose test matrices
disp('Decomposing test sequences into previous dictionaries')
classification_results = [];
for n = 1:length(test_mats)
    fprintf('Decomposing test sequence %d with dictionary ', n)
    decomp_results = {};
    for i = 1:length(dicts)
        fprintf('%d ', i)
        params_test=struct('K', 4, ...
                       'numIteration', 15, ... 
                       'errorFlag', 0, ...
                       'L', 10, ...
                       'preserveDCAtom', 0, ...
                       'InitializationMethod','GivenMatrix', ...
                       'initialDictionary', dicts(i),...
                       'displayProgress', 0);

        [temp1, temp2] = KSVD(test_mats{n}, params_test);
        decomp_results{i} = temp2.CoefMatrix;
    end               
    fprintf('\n')
    
    % Calculate residual
    % (it should hold that Data equals approximately
    % Dictionary*output.CoefMatrix) <--- key to calculating residual!!
    error = [];
    for i = 1:length(dicts)
        error(i) = abs(sum(sum(dicts{i}*decomp_results{i} - test_mats{n})));
    end
    %disp('Calculated the following residuals:')
    %disp(error)
    %disp('The training sequence that better represents the test sequence is')
    [temp, idx] = min(error);
    %disp(idx)
    
    
    %classification_results = [classification_results; n, idx, train_cluster_names(n)];
    classification_results{n}.matched_cluster = idx;
    classification_results{n}.test_cluster_name = test_cluster_names(n);
    classification_results{n}.matched_cluster_name = train_cluster_names(idx);
end

% Announce classification
disp('Results:')
for n = 1:length(test_mats)
    disp(n)
    disp(classification_results{n})
end

% Clear unnecessary variables
clearvars -except dicts train_cluster_names test_cluster_names classification_results scale

%% Clear everything except training
clearvars -except dicts train_cluster_names test_cluster_names
