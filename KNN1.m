% K-nearest neighbor rule %
clear all;, close all;

r=13;      % dimensionality of each subspace
imgnum=1;  % test sample number for displaying Uc
nclass=10; % number of classes

% loading data-file
load('./usps_resampled/usps.mat');
[d,ndata]=size(trai);

% normalization
for ii = 1 : ndata
  trai(:,ii)=trai(:,ii)./norm(trai(:,ii));
end

%K-NN法をすべてのテストサンプルに行う
labels = zeros(4649, 1);
for n = 1 : 4649
  %テストサンプルと各訓練データのユークリッド距離を計算する
  D = zeros(1, 4649);
  for i = 1 : 4649
    D(1, i) = norm(test(:, n) - trai(:, i));
  end
  %もっともユークリッド距離の小さいクラスを求める
  [D_sorted, index] = sort(D);
  class_num = trai_label(index(1, 1), 1);
  labels(n, 1) = class_num;
end

%混同行列を作成する
conf_matrix = zeros(10, 10);
for j = 1 : 4649
  %行番号と列番号を取得する
  line_num = test_label(j, 1) + 1;
  column_num = labels(j, 1) + 1;
  %混同行列内の値を１増やす
  conf_matrix(line_num, column_num) += 1;
end

%クラス平均認識率を求める
A = diag(conf_matrix);
B = sum(conf_matrix, 2);
accuracy = A ./ B;
accuracy_mean = mean(accuracy);
printf('accuracy = %f\n', accuracy_mean);
