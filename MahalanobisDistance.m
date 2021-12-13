% Mahalanobis Distance rule %
warning('off', 'all');
clear all;, close all;

% loading data-file
load('./usps_resampled/usps.mat');
[d,ndata]=size(trai);

% normalization
for ii = 1 : ndata
  trai(:,ii)=trai(:,ii)./norm(trai(:,ii));
  test(:,ii)=test(:,ii)./norm(test(:,ii));
end

%各クラスのデータをまとめた行列を求める
class_count = zeros(1, 10);
class_data = zeros(256, 800, 10);
for k = 0 : 9
  for m = 1 : 4649
    if trai_label(m, 1) == k
      class_count(1, k + 1) += 1;
      class_data(:, class_count(1, k + 1), k + 1) = trai(:, m);
    endif
  endfor
endfor

%マハラノビス法をすべてのテストサンプルに行う
labels = zeros(4649, 1);
for i = 1 : 4649
  D = zeros(1, 10);
  for n = 1 : 10
    %各クラスのデータで不要な列を削除する
    data = class_data(:, :, n);
    data(:, class_count(1, n) + 1 : 800) = [];
    %テストサンプルと各訓練クラスデータのマハラノビス距離を計算する
    inv_cov = inv(cov(data'));
    mean_class = mean(data, 2);
    D(1, n) = (test(:, i) - mean_class)' * inv_cov * (test(:, i) - mean_class);
  end
  %もっともマハラノビス距離の小さいクラスを求める
  [D_sorted, index] = sort(D);
  class_num = index(1, 1) - 1;
  labels(i, 1) = class_num;
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
