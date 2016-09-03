% Classify Sequences
% Author: Gonçalo S. Martins
% This script takes a number of pre-clustered image sequences and builds
% the appropriate dictionaries, using the K-SVD toolbox <link goes here>.
%
% USAGE:
% Code sections are numbered for user convenience.

%% 1. Load Training Data
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

%% 2. Decompose training matrices
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
clearvars -except dicts train_cluster_names scale test_mats test_cluster_names

%% 3. Load Test Clusters
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
    disp(n)
    disp(classification_results{n})
end

% Clear unnecessary variables
clearvars -except dicts train_cluster_names test_cluster_names classification_results scale

%% Alternative Step 1: Get Training Data from Clustering
% Load sequence images from a folder
base_dir = 'train_sequences_extended/'; % base_dir must end in '/'

% Number of clusters to use
K = 10;

% Scale of the images to load
scale = 0.1;

% Read folder names
seqs = dir(base_dir);
seqs = seqs(3:end);

% Linear counter for training matrices
n = 1;
train_mats = {};

% For each sequence in the folder
for f = 1:length(seqs)
    % Extract the folder names of the sequences
    folder = strcat(base_dir, seqs(f).name);
    folder = strcat(folder, '/');
    imgs = dir(folder);
    imgs = imgs(3:end);

    % For each subfolder of the main data folder (sequence)
    % For each image
    for i = 1:length(imgs)
        % Build filename
        filename = strcat(folder, imgs(i).name);
        % Load image
        temp_img = rgb2gray(imread(filename));
        % Scale image so largest dimension is 300
        temp_img = imresize(temp_img, 300/min(size(temp_img)));
        % Crop to 300x300
        temp_img = temp_img(1:300, 1:300);
        %disp(size(temp_img))
        % Scale and load into sequence
        sequence{i}= imresize(temp_img, scale);
    end
    fprintf('Loaded %d images from directory %s.\n', i, folder)

    % Determine the two "most different" images
    i_star = 1; % Indices of the images will go here
    j_star = 1;
    T = [];
    I = [];
    while length(T) < K
        if length(T) == 0 % This is the first iteration
            max_dist = 0;
            % Calculate the distances between all images
            % the two with the largest distance will be the first seeds in T
            for i = 1:length(sequence)
                for j = 1:length(sequence)
                    if j ~= i
                        % Calculate distance
                        dist = norm(double(sequence{i}(:) - sequence{j}(:)));
                        % Update maximum, if necessary
                        if dist > max_dist
                            max_dist = dist;
                            i_star = i;
                            j_star = j;
                        end
                    end
                end
            end
            T = [i_star, j_star]; % These images are the "centers" of the clusters
            I = [i_star, j_star]; % I will keep the indices of the images already "used"
        else
            max_dist = 0;
            i_star = 0;
            % Calculate the distances between each image and the cluster seeds
            % the most distant will become a new seed
            for i = 1:length(sequence)
                repeat = 0;
                if length(I(I==i)) > 0 || length(T(T==i)) > 0
                    repeat = 1;
                end
                if repeat == 0
                    for j = 1:length(T)
                        % Calculate distance
                        dist = norm(double(sequence{T(j)}(:) - sequence{i}(:)));
                        % Update maximum, if necessary
                        if dist > max_dist
                            max_dist = dist;
                            i_star = i;
                        end
                    end
                end
            end
            T = [T, i_star]; % These images are the "centers" of the clusters
            I = [I, i_star]; % I will keep the indices of the images already "used"
        end
    end
    fprintf('Found me some optima:')
    disp(T)

    % Partition the sequence into the clusters already seeded
    for i = 1:K
        clusters{i} = T(i);
    end

    % For every image remaining in the sequence
    for i = 1:length(sequence)
        % if i not in I, i.e. if image i is not in a cluster yet
        % (if it is, we skip this iteration)
        i_in_I = false;
        for j = 1:length(I)
            if i == I(j)
                i_in_I = true;
                break
            end
        end
        if i_in_I == true
            continue
        end
        %disp(i)

        % Determine which cluster the image should go to
        k_star = 0;
        min_dist = flintmax('double');
        % For every cluster center
        for k = 1:length(T)
            % Calculate distance to cluster center
            dist = norm(double(sequence{T(k)}(:) - sequence{i}(:)));
            % Update minimum, if required
            if dist < min_dist
                min_dist = dist;
                k_star = k;
            end
        end

        % Add the image to the cluster
        clusters{k_star} = [clusters{k_star}, i];
        I = [I, i];
    end

    % Add clusters to final data structure
    for ii = 1:length(clusters)
        % Only add clusters with enough images to decompose
        if length(clusters{ii}) > 3
            train_mats{n} = [];
            train_cluster_names{n} = seqs(f).name;
            for jj = 1:length(clusters{ii})
                temp_img = sequence(clusters{ii}(jj));
                temp_img = temp_img{1};
                train_mats{n} = [train_mats{n}, double(temp_img(:))];
            end
            n = n+1;
        end
    end
end
clearvars -except clustered_seqs folder_names scale train_cluster_names train_mats

%% Alternative Steps 1 and 3: Get Training and Test Data from Clustering
% Random images will be read for both
% Load sequence images from a folder
clear all
base_dir = 'train_sequences_maximum/'; % base_dir must end in '/'

% Number of clusters to use
K = 8;

% Scale of the images to load
scale = 0.1;

% Read folder names
seqs = dir(base_dir);
seqs = seqs(3:end);

% First, we'll do training data
% Linear counter for training matrices
n = 1;
train_mats = {};

% For each sequence in the folder
for f = 1:length(seqs)
    % Extract the folder names of the sequences
    folder = strcat(base_dir, seqs(f).name);
    folder = strcat(folder, '/');
    imgs = dir(folder);
    imgs = imgs(3:end);

    % For each subfolder of the main data folder (sequence)
    % For each image
    for i = 1:100
        idx = randi(length(imgs));
        % Build filename
        filename = strcat(folder, imgs(idx).name);
        % Load image
        temp_img = rgb2gray(imread(filename));
        % Scale image so largest dimension is 300
        temp_img = imresize(temp_img, 300/min(size(temp_img)));
        % Crop to 300x300
        temp_img = temp_img(1:300, 1:300);
        %disp(size(temp_img))
        % Scale and load into sequence
        sequence{i}= imresize(temp_img, scale);
    end
    fprintf('Loaded %d images from directory %s.\n', i, folder)

    % Determine the two "most different" images
    i_star = 1; % Indices of the images will go here
    j_star = 1;
    T = [];
    I = [];
    while length(T) < K
        if length(T) == 0 % This is the first iteration
            max_dist = 0;
            % Calculate the distances between all images
            % the two with the largest distance will be the first seeds in T
            for i = 1:length(sequence)
                for j = 1:length(sequence)
                    if j ~= i
                        % Calculate distance
                        dist = norm(double(sequence{i}(:) - sequence{j}(:)));
                        % Update maximum, if necessary
                        if dist > max_dist
                            max_dist = dist;
                            i_star = i;
                            j_star = j;
                        end
                    end
                end
            end
            T = [i_star, j_star]; % These images are the "centers" of the clusters
            I = [i_star, j_star]; % I will keep the indices of the images already "used"
        else
            max_dist = 0;
            i_star = 0;
            % Calculate the distances between each image and the cluster seeds
            % the most distant will become a new seed
            for i = 1:length(sequence)
                repeat = 0;
                if length(I(I==i)) > 0 || length(T(T==i)) > 0
                    repeat = 1;
                end
                if repeat == 0
                    for j = 1:length(T)
                        % Calculate distance
                        dist = norm(double(sequence{T(j)}(:) - sequence{i}(:)));
                        % Update maximum, if necessary
                        if dist > max_dist
                            max_dist = dist;
                            i_star = i;
                        end
                    end
                end
            end
            T = [T, i_star]; % These images are the "centers" of the clusters
            I = [I, i_star]; % I will keep the indices of the images already "used"
        end
    end
    fprintf('Found me some optima:')
    disp(T)

    % Partition the sequence into the clusters already seeded
    clear clusters
    for i = 1:K
        clusters{i} = T(i);
    end

    % For every image remaining in the sequence
    for i = 1:length(sequence)
        % if i not in I, i.e. if image i is not in a cluster yet
        % (if it is, we skip this iteration)
        i_in_I = false;
        for j = 1:length(I)
            if i == I(j)
                i_in_I = true;
                break
            end
        end
        if i_in_I == true
            continue
        end
        %disp(i)

        % Determine which cluster the image should go to
        k_star = 0;
        min_dist = flintmax('double');
        % For every cluster center
        for k = 1:length(T)
            % Calculate distance to cluster center
            dist = norm(double(sequence{T(k)}(:) - sequence{i}(:)));
            % Update minimum, if required
            if dist < min_dist
                min_dist = dist;
                k_star = k;
            end
        end

        % Add the image to the cluster
        clusters{k_star} = [clusters{k_star}, i];
        I = [I, i];
    end

    % Add clusters to final data structure
    for ii = 1:length(clusters)
        % Only add clusters with enough images to decompose
        if length(clusters{ii}) > 3
            train_mats{n} = [];
            train_cluster_names{n} = seqs(f).name;
            for jj = 1:length(clusters{ii})
                temp_img = sequence(clusters{ii}(jj));
                temp_img = temp_img{1};
                train_mats{n} = [train_mats{n}, double(temp_img(:))];
            end
            n = n+1;
        end
    end
end

% Second, we'll do test data
% Linear counter for training matrices
clearvars -except train_mats train_cluster_names scale K seqs base_dir
K = K/2;
n = 1;
test_mats = {};

% For each sequence in the folder
for f = 1:length(seqs)
    % Extract the folder names of the sequences
    folder = strcat(base_dir, seqs(f).name);
    folder = strcat(folder, '/');
    imgs = dir(folder);
    imgs = imgs(3:end);

    % For each subfolder of the main data folder (sequence)
    % For each image
    for i = 1:50
        idx = randi(length(imgs));
        % Build filename
        filename = strcat(folder, imgs(idx).name);
        % Load image
        temp_img = rgb2gray(imread(filename));
        % Scale image so largest dimension is 300
        temp_img = imresize(temp_img, 300/min(size(temp_img)));
        % Crop to 300x300
        temp_img = temp_img(1:300, 1:300);
        %disp(size(temp_img))
        % Scale and load into sequence
        sequence{i}= imresize(temp_img, scale);
    end
    fprintf('Loaded %d images from directory %s.\n', i, folder)

    % Determine the two "most different" images
    i_star = 1; % Indices of the images will go here
    j_star = 1;
    T = [];
    I = [];
    while length(T) < K
        if length(T) == 0 % This is the first iteration
            max_dist = 0;
            % Calculate the distances between all images
            % the two with the largest distance will be the first seeds in T
            for i = 1:length(sequence)
                for j = 1:length(sequence)
                    if j ~= i
                        % Calculate distance
                        dist = norm(double(sequence{i}(:) - sequence{j}(:)));
                        % Update maximum, if necessary
                        if dist > max_dist
                            max_dist = dist;
                            i_star = i;
                            j_star = j;
                        end
                    end
                end
            end
            T = [i_star, j_star]; % These images are the "centers" of the clusters
            I = [i_star, j_star]; % I will keep the indices of the images already "used"
        else
            max_dist = 0;
            i_star = 0;
            % Calculate the distances between each image and the cluster seeds
            % the most distant will become a new seed
            for i = 1:length(sequence)
                repeat = 0;
                if length(I(I==i)) > 0 || length(T(T==i)) > 0
                    repeat = 1;
                end
                if repeat == 0
                    for j = 1:length(T)
                        % Calculate distance
                        dist = norm(double(sequence{T(j)}(:) - sequence{i}(:)));
                        % Update maximum, if necessary
                        if dist > max_dist
                            max_dist = dist;
                            i_star = i;
                        end
                    end
                end
            end
            T = [T, i_star]; % These images are the "centers" of the clusters
            I = [I, i_star]; % I will keep the indices of the images already "used"
        end
    end
    fprintf('Found me some optima:')
    disp(T)

    % Partition the sequence into the clusters already seeded
    clear clusters
    for i = 1:K
        clusters{i} = T(i);
    end

    % For every image remaining in the sequence
    for i = 1:length(sequence)
        % if i not in I, i.e. if image i is not in a cluster yet
        % (if it is, we skip this iteration)
        i_in_I = false;
        for j = 1:length(I)
            if i == I(j)
                i_in_I = true;
                break
            end
        end
        if i_in_I == true
            continue
        end
        %disp(i)

        % Determine which cluster the image should go to
        k_star = 0;
        min_dist = flintmax('double');
        % For every cluster center
        for k = 1:length(T)
            % Calculate distance to cluster center
            dist = norm(double(sequence{T(k)}(:) - sequence{i}(:)));
            % Update minimum, if required
            if dist < min_dist
                min_dist = dist;
                k_star = k;
            end
        end

        % Add the image to the cluster
        clusters{k_star} = [clusters{k_star}, i];
        I = [I, i];
    end

    % Add clusters to final data structure
    for ii = 1:length(clusters)
        % Only add clusters with enough images to decompose
        if length(clusters{ii}) > 3
            test_mats{n} = [];
            test_cluster_names{n} = seqs(f).name;
            for jj = 1:length(clusters{ii})
                temp_img = sequence(clusters{ii}(jj));
                temp_img = temp_img{1};
                test_mats{n} = [test_mats{n}, double(temp_img(:))];
            end
            n = n+1;
        end
    end
end

clearvars -except clustered_seqs folder_names scale train_cluster_names train_mats test_mats test_cluster_names