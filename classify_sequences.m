%%
% Classify Sequences
% Author: Gonçalo S. Martins
% This script takes a number of pre-clustered image sequences and builds
% the appropriate dictionaries, using the K-SVD toolbox <link goes here>.
% This script assumes the following folder structure:
%
% <sequences_root>/seq_<something>/1
%                 ^               /2
%                 |               /
%                 One per seq.    ^
%                                 | One per cluster (names should be 1, 2,
%                                                    etc.)

% For now, this needs to stay; we're not cleaning up properly at the end
clear all

% Load image clusters
% For now, filenames are hard-coded (and the above structure is not
% actually used or needed
% train_files{1} = ['sequences_cropped/filipa_1/1/33.jpg';
%                   'sequences_cropped/filipa_1/1/34.jpg';
%                   'sequences_cropped/filipa_1/1/35.jpg';
%                   'sequences_cropped/filipa_1/1/36.jpg';];
%              
% train_files{2} = ['sequences_cropped/filipa_1/2/41.jpg';
%                   'sequences_cropped/filipa_1/2/42.jpg';
%                   'sequences_cropped/filipa_1/2/43.jpg';
%                   'sequences_cropped/filipa_1/2/44.jpg';
%                   'sequences_cropped/filipa_1/2/45.jpg'];
%               
% for i = 1:length(train_files)
%     % Load all cluster images, cluster by cluster
%     n_imgs = size(train_files{i});
%     n_imgs = n_imgs(1);
%     for j = 1:n_imgs
%         train_clusters{i}{j} = rgb2gray(imread(train_files{i}(i,:)));
%     end
% end

% Load train clusters
base_dir = 'sequences_cropped/train/'
seqs = dir(base_dir);
seqs = seqs(3:end);
seqs(1).name;
length(seqs);

% For each subfolder of the main data folder (sequence)
n = 1 % linear counter for clusters
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
            train_clusters{n}{k} = rgb2gray(imread(filename));
        end
        n = n + 1;
    end 
end

% Load test clusters
test_files{1} = ['sequences_cropped/test/filipa_test/1/1.jpg';
                 'sequences_cropped/test/filipa_test/1/3.jpg';
                 'sequences_cropped/test/filipa_test/1/4.jpg';
                 'sequences_cropped/test/filipa_test/1/6.jpg'];
              
for i = 1:length(test_files)
    % Load all cluster images, cluster by cluster
    n_imgs = size(test_files{i});
    n_imgs = n_imgs(1);
    for j = 1:n_imgs
        test_clusters{i}{j} = rgb2gray(imread(test_files{i}(i,:)));
    end
end

% From here on out, the cell array train_clusters will contain all of the
% training clusters. Each element of train_clusters is a cell array
% containing all of the images of that cluster, one per cell.
% Similarly, the cell array train_clusters will contain all of the test
% clusters.

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

test_mats = {};
for i = 1:length(test_clusters)
    test_mats{i} = [];
    for j = 1:length(test_clusters{i})
        test_mats{i} = [test_mats{i}, double(test_clusters{i}{j}(:))];
    end
end
%size(test_mats)

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
    info = sprintf('Obtaining dictionary %d of %d.', i, length(train_mats))
    [dicts{i}, temp] = KSVD(train_mats{i}, params);
    decomps{i} = temp.CoefMatrix;
end

% Decompose test matrices
disp('Decomposing test sequences into previous dictionaries')
decomp_results = {};
for i = 1:length(dicts)
    info = ['Decomposing test sequence with dictionary ', num2str(i)];
    disp(info)
    params_test=struct('K', 4, ...
                   'numIteration', 15, ... 
                   'errorFlag', 0, ...
                   'L', 10, ...
                   'preserveDCAtom', 0, ...
                   'InitializationMethod','GivenMatrix', ...
                   'initialDictionary', dicts(i),...
                   'displayProgress', 0);
               
    [temp1, temp2] = KSVD(test_mats{1}, params_test);
    decomp_results{i} = temp2.CoefMatrix;
end               

% Calculate residual
% (it should hold that Data equals approximately
% Dictionary*output.CoefMatrix) <--- key to calculating residual!!
error = [];
for i = 1:length(dicts)
    error(i) = abs(sum(sum(dicts{i}*decomp_results{i} - test_mats{1})));
end
disp('Calculated the following residuals:')
error
disp('The training sequence that better represents the test sequence is')
[temp, idx] = min(error);
idx


% Announce classification
disp('Done!')