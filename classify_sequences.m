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

%% 2. Load Test Cluster Filenames
% This section defines the filenames of the clusters to be used for
% testing. This is the place to change if you want to test with different
% clusters.
% Cluster 1
test_files{1} = ['sequences_cropped/test/filipa_test/1/1.jpg';
                 'sequences_cropped/test/filipa_test/1/3.jpg';
                 'sequences_cropped/test/filipa_test/1/4.jpg';
                 'sequences_cropped/test/filipa_test/1/6.jpg'];
test_cluster_names{1} = ('filipa');

% Cluster 2
test_files{2} = ['sequences_cropped/test/filipa_test/2/64.jpg';
                 'sequences_cropped/test/filipa_test/2/65.jpg';
                 'sequences_cropped/test/filipa_test/2/66.jpg';
                 'sequences_cropped/test/filipa_test/2/67.jpg';
                 'sequences_cropped/test/filipa_test/2/68.jpg'];
test_cluster_names{2} = ('filipa');

% Cluster 3
test_files{3} = ['sequences_cropped/test/filipa_test/3/171.jpg';
                 'sequences_cropped/test/filipa_test/3/172.jpg';
                 'sequences_cropped/test/filipa_test/3/173.jpg';
                 'sequences_cropped/test/filipa_test/3/174.jpg';
                 'sequences_cropped/test/filipa_test/3/175.jpg';
                 'sequences_cropped/test/filipa_test/3/176.jpg';
                 'sequences_cropped/test/filipa_test/3/177.jpg'];
test_cluster_names{3} = ('filipa');
             
% Cluster 4
test_files{4} = ['sequences_cropped/test/goncalo_test/1/22.jpg';
                 'sequences_cropped/test/goncalo_test/1/23.jpg';
                 'sequences_cropped/test/goncalo_test/1/24.jpg';
                 'sequences_cropped/test/goncalo_test/1/25.jpg';
                 'sequences_cropped/test/goncalo_test/1/26.jpg'];
test_cluster_names{4} = ('goncalo');

% Cluster 5
test_files{5} = ['sequences_cropped/test/goncalo_test/2/03.jpg';
                 'sequences_cropped/test/goncalo_test/2/13.jpg';
                 'sequences_cropped/test/goncalo_test/2/14.jpg';
                 'sequences_cropped/test/goncalo_test/2/15.jpg'];
test_cluster_names{5} = ('goncalo');

% Cluster 6
test_files{6} = ['sequences_cropped/test/goncalo_test/3/102.jpg';
                 'sequences_cropped/test/goncalo_test/3/103.jpg';
                 'sequences_cropped/test/goncalo_test/3/104.jpg';
                 'sequences_cropped/test/goncalo_test/3/132.jpg';
                 'sequences_cropped/test/goncalo_test/3/133.jpg';
                 'sequences_cropped/test/goncalo_test/3/134.jpg'];
test_cluster_names{6} = ('goncalo');

% Cluster 7
test_files{7} = ['sequences_cropped/test/awais_test/1/361.jpg';
                 'sequences_cropped/test/awais_test/1/362.jpg';
                 'sequences_cropped/test/awais_test/1/363.jpg';
                 'sequences_cropped/test/awais_test/1/364.jpg'];
test_cluster_names{7} = ('awais');

% Cluster 8
test_files{8} = ['sequences_cropped/test/awais_test/2/16.jpg';
                 'sequences_cropped/test/awais_test/2/17.jpg';
                 'sequences_cropped/test/awais_test/2/18.jpg';
                 'sequences_cropped/test/awais_test/2/19.jpg'];
test_cluster_names{8} = ('awais');

% Cluster 9
test_files{9} = ['sequences_cropped/test/awais_test/3/286.jpg';
                 'sequences_cropped/test/awais_test/3/287.jpg';
                 'sequences_cropped/test/awais_test/3/288.jpg';
                 'sequences_cropped/test/awais_test/3/289.jpg'];
test_cluster_names{9} = ('awais');
             
% Cluster 10
test_files{10} = ['sequences_cropped/test/miguel_test/1/431.jpg';
                  'sequences_cropped/test/miguel_test/1/432.jpg';
                  'sequences_cropped/test/miguel_test/1/433.jpg';
                  'sequences_cropped/test/miguel_test/1/434.jpg'];
test_cluster_names{10} = ('miguel');

% Cluster 11
test_files{11} = ['sequences_cropped/test/miguel_test/2/316.jpg';
                  'sequences_cropped/test/miguel_test/2/317.jpg';
                  'sequences_cropped/test/miguel_test/2/318.jpg';
                  'sequences_cropped/test/miguel_test/2/319.jpg'];
test_cluster_names{11} = ('miguel');

% Cluster 12
test_files{12} = ['sequences_cropped/test/miguel_test/3/152.jpg';
                  'sequences_cropped/test/miguel_test/3/153.jpg';
                  'sequences_cropped/test/miguel_test/3/154.jpg';
                  'sequences_cropped/test/miguel_test/3/155.jpg'];
test_cluster_names{12} = ('miguel');



%% 3. Load Test Clusters
% This section loads the test cluster from the pre-defined image names.
% the cell array train_clusters will contain all of the test clusters.              
for i = 1:length(test_files)
    % Load all cluster images, cluster by cluster
    n_imgs = size(test_files{i});
    n_imgs = n_imgs(1);
    for j = 1:n_imgs
        %fprintf('Loading file %s into cluster %d.\n', test_files{i}(j,:), i) 
        test_clusters{i}{j} = imresize(rgb2gray(imread(test_files{i}(j,:))), scale);
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
%size(test_mats)

% Clear unnecessary variables
clearvars -except test_mats dicts train_cluster_names test_cluster_names scale

%% 4. Classify Test Clusters
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
    disp(classification_results{n})
end

% Clear unnecessary variables
clearvars -except dicts train_cluster_names test_cluster_names classification_results scale

%% Clear everything except training
clearvars -except dicts train_cluster_names test_cluster_names
