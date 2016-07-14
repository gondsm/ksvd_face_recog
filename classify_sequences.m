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
train_1_names = ['sequences/filipa_1/1/33.jpg';
                 'sequences/filipa_1/1/34.jpg';
                 'sequences/filipa_1/1/35.jpg';
                 'sequences/filipa_1/1/36.jpg']

train_2_names = ['sequences/filipa_1/2/41.jpg';
                 'sequences/filipa_1/2/42.jpg';
                 'sequences/filipa_1/2/43.jpg';
                 'sequences/filipa_1/2/44.jpg';
                 'sequences/filipa_1/2/45.jpg']
             
n_train_1 = size(train_1_names);
n_train_1 = n_train_1(1);
train_1 = {};
for i = 1:n_train_1
    train_1{i} = rgb2gray(imread(train_1_names(i,:)));
end

n_train_2 = size(train_2_names);
n_train_2 = n_train_2(1);
train_2 = {};
for i = 1:n_train_2
    train_2{i} = rgb2gray(imread(train_2_names(i,:)));
end

% We'll test on training images, to check how stuff works
test_1 = train_1;

%size(train_1{1})
%imshow(train_1{4})
%size(train_1{1}(:)')
%length(train_1)

% From here on out, the cell array train_clusters will contain all of the
% training clusters. Each element of train_clusters is a cell array
% containing all of the images of that cluster, one per cell.
train_clusters{1} = train_1;
train_clusters{2} = train_2;
%length(train_clusters)

% Similarly, the cell array train_clusters will contain all of the test
% clusters.
test_clusters{1} = test_1;

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

% train_1_mat = []
% for i = 1:length(train_1)
%     train_1_mat = [train_1_mat, double(train_1{i}(:))];
% end
% size(train_1_mat)

% train_2_mat = []
% for i = 1:length(train_2)
%     train_2_mat = [train_2_mat, double(train_2{i}(:))];
% end
% size(train_2_mat)

% Decompose training matrices
% Whoa, almost looks like a civilized dictionary:
params=struct('K', 4, ...
              'numIteration', 15, ... 
              'errorFlag', 0, ...
              'L', 10, ...
              'preserveDCAtom', 0, ...
              'InitializationMethod','DataElements', ...
              'displayProgress', 1)

%[dict, out] = KSVD(train_1_mat, params);
%[dict, out] = KSVD(train_2_mat, params);
% size(dict)
% size(out)
dicts = {};
decomps = {};
for i = 1:length(train_mats)
    [dicts{i}, temp] = KSVD(train_mats{i}, params);
    decomps{i} = temp.CoefMatrix;
end

% Decompose test matrices

% Calculate residual
% (it should hold that Data equals approximately
% Dictionary*output.CoefMatrix) <--- key to calculating residual!!

% Announce classification
disp('Done!')