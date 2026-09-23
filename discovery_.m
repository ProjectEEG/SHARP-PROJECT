%% 1. 数据准备
close all; clear; clc;

% LOSO 是确定性的，不需要随机种子来控制划分，但 ReliefF 内部可能有随机性
rng(2026); 

load('57features.mat'); 

% 统一数据
y = double(y1');       % (165x1)
X = X_train;           % (165x38)

n_samples = length(y);

% 定义阈值列表 (用于最后计算指标)
thresholds = [0.05, 0.10, 0.15, 0.2, 0.25,0.26,0.27,0.28, 0.29,0.3,0.31,0.32,0.33,0.34, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]; 
num_thresh = length(thresholds);

% 用于存储每一个样本作为测试集时的预测概率
% LOSO 的核心：把这个向量填满
full_preds_prob = zeros(n_samples, 1);
full_true_labels = zeros(n_samples, 1);

fprintf('=== 开始留一法嵌套交叉验证 (LOSO Nested CV) ===\n');
fprintf('总样本数: %d (将循环 %d 次)\n', n_samples, n_samples);

%% 2. LOSO 外层循环 (遍历每个样本)
% 建议：如果有 Parallel Computing Toolbox，将 for 改为 parfor 可显著加速
for k = 1:n_samples
    
    fprintf('Processing Subject %d / %d ...\n', k, n_samples);
    
    % --- 1. 划分数据 (Leave-One-Out) ---
    te_idx = false(n_samples, 1);
    te_idx(k) = true;      % 第 k 个样本做测试
    tr_idx = ~te_idx;      % 其余 N-1 个做训练
    
    X_fold_train = X(tr_idx, :);
    y_fold_train = y(tr_idx);
    X_fold_val   = X(te_idx, :);
    y_fold_val   = y(te_idx);
    
    % --- 2. 调用堆叠系统 (内层训练) ---
    % 注意：内层依然使用 5-Fold CV 来做参数搜索和 OOF 特征生成
    % 这是标准的 Nested CV 做法，效率最高
    [pred_prob, true_val, ~, ~] = run_stacked_svm_system_loso(X_fold_train, y_fold_train, X_fold_val, y_fold_val);
    
    % --- 3. 收集结果 ---
    full_preds_prob(k) = pred_prob;
    full_true_labels(k) = true_val;
end

fprintf('\n循环结束，开始计算整体指标...\n');

%% 3. 计算整体指标并保存
% 在 LOSO 中，只有当所有样本都预测完，才能算 AUC
pos_class = 1; 

% 计算整体 AUC
[Xroc, Yroc, ~, Global_AUC] = perfcurve(full_true_labels, full_preds_prob, pos_class);

AUCPLOT_stack_val.x =  Xroc;
AUCPLOT_stack_val.y =  Yroc;
save('AUCPLOT_stack_val.mat', 'AUCPLOT_stack_val');

% 准备结果矩阵
results_mat = zeros(num_thresh, 8); % Thresh, AUC, BAC, ACC, SENSI, SPECI, PPV, NPV

for i = 1:num_thresh
    ts = thresholds(i);
    binary_preds = full_preds_prob > ts;
    
    % 计算混淆矩阵
    conf_mat = confusionmat(full_true_labels, double(binary_preds), 'Order', [0, 1]);
    TN = conf_mat(1,1); FP = conf_mat(1,2);
    FN = conf_mat(2,1); TP = conf_mat(2,2);
    
    ACC   = (TP + TN) / sum(conf_mat(:));
    SENSI = TP / (TP + FN + 1e-10);
    SPECI = TN / (TN + FP + 1e-10);
    BAC   = (SENSI + SPECI) / 2;
    PPV   = TP / (TP + FP + 1e-10);
    NPV   = TN / (TN + FN + 1e-10);
    
    results_mat(i, :) = [ts, Global_AUC, BAC, ACC, SENSI, SPECI, PPV, NPV];
end

% 保留 4 位小数
results_mat = round(results_mat, 4);

% 创建 Table
col_names = {'Threshold', 'Global_AUC', 'Global_BAC', 'Global_ACC', 'Global_SENSI', 'Global_SPECI', 'Global_PPV', 'Global_NPV'};
results_table = array2table(results_mat, 'VariableNames', col_names);

% 保存结果
excel_filename = 'NestedCV_LOSO_Results.xlsx';
try
    writetable(results_table, excel_filename);
    fprintf('\n✅ 成功！LOSO 结果已保存至: %s\n', excel_filename);
    disp(results_table);
catch ME
    fprintf('\n❌ 保存失败 (请关闭 Excel 文件): %s\n', ME.message);
end

%% ============================================================
%  函数：run_stacked_svm_system_loso
%  (内层保持 K-Fold 用于高效生成 Stacking 特征和选参)
% ============================================================
function [pred_scores, true_labels, meta_model, L1_scalers] = run_stacked_svm_system_loso(X_train, y_train, X_test, y_test)
    % 0. 预处理
    y_train = double(y_train(:)); 
    y_test  = double(y_test(:));
    true_labels = y_test;
    
    n_train = size(X_train, 1);
    n_test  = size(X_test, 1);
    
    % 特征分组
    cols_cog = 1:6; 
    cols_sym = 7:17;
    cols_eeg = 18:26; 
    cols_eeg1 = 27:57;
    
    % --- 特征选择与分组 (保持你的 ReliefF 逻辑) ---
    cols_eeg_rs1 = 27:57;

    y_train_cat = categorical(y_train);
    
    % 注意：这里是在 (N-1) 个训练样本上做特征选择，完全合规
    [idx, ~] = relieff(X_train(:, cols_eeg_rs1), y_train_cat, 15);
    cols_eeg_rs1 = idx(1:fix(length(cols_eeg_rs1)/2))+cols_eeg_rs1(1)-1;
    feature_groups = {cols_cog, cols_sym, cols_eeg, cols_eeg_rs1};
    group_names = {'Cognitive', 'Symptoms', 'EEG', 'EEG_RS'};
    num_groups = length(feature_groups);
    
    L1_train_feats = zeros(n_train, num_groups);
    L1_test_feats  = zeros(n_test, num_groups);
    L1_scalers = struct(); 
    
    % 1. 生成内部 CV 结构 (用于 Stacking 的 Level-0 训练)
    % 内层使用 5折交叉验证是最高效的，不需要在内层也搞 LOSO
    n_repeats_inner = 5; 
    n_folds_inner = 5;
    cv_partitions = cell(n_repeats_inner, 1);
    for r = 1:n_repeats_inner
        cv_partitions{r} = cvpartition(y_train, 'KFold', n_folds_inner);
    end
    
    % =========================================================
    % 2. Level-0 训练 (带 Gap Penalty 选参)
    % =========================================================
    box_constraints = [0.0001, 0.001, 0.01, 0.1, 1, 10]; 
    
    for g = 1:num_groups
        curr_group_name = group_names{g};
        X_sub_train = X_train(:, feature_groups{g});
        X_sub_test  = X_test(:, feature_groups{g});
        
        % A. 选参 (Train/Test Gap Penalty 逻辑)
        best_score = -inf; best_bc = 1;
        
        for bc = box_constraints
            bacs_val_temp = [];
            bacs_train_temp = [];
            
            for r = 1:n_repeats_inner
                cvp = cv_partitions{r};
                for k = 1:n_folds_inner
                    tr_idx = training(cvp, k); te_idx = test(cvp, k);
                    try
                        mdl = fitcsvm(X_sub_train(tr_idx,:), y_train(tr_idx), ...
                            'KernelFunction', 'linear', 'Prior', 'uniform', ...
                            'Standardize', true, 'BoxConstraint', bc);
                        
                        % Val BAC
                        y_pred_val = predict(mdl, X_sub_train(te_idx, :));
                        stats_val = confusionmat(y_train(te_idx), y_pred_val);
                        if size(stats_val,1)==2
                            bac_val = (stats_val(2,2)/sum(stats_val(2,:)) + stats_val(1,1)/sum(stats_val(1,:)))/2;
                        else; bac_val = 0.5; end
                        bacs_val_temp = [bacs_val_temp; bac_val];
                        
                        % Train BAC
                        y_pred_tr = predict(mdl, X_sub_train(tr_idx, :));
                        stats_tr = confusionmat(y_train(tr_idx), y_pred_tr);
                        if size(stats_tr,1)==2
                            bac_tr = (stats_tr(2,2)/sum(stats_tr(2,:)) + stats_tr(1,1)/sum(stats_tr(1,:)))/2;
                        else; bac_tr = 0.5; end
                        bacs_train_temp = [bacs_train_temp; bac_tr];
                    catch; continue; end
                end
            end
            
            mean_val = mean(bacs_val_temp);
            mean_tr  = mean(bacs_train_temp);
            gap = abs(mean_tr - mean_val);
            score = mean_val - 0.5 * gap; % 惩罚过拟合
%             score = mean_val;
            if score > best_score; best_score = score; best_bc = bc; end
        end
        
        % B. OOF Predictions (用内层 CV 生成 Stacking 特征)
        score_sum = zeros(n_train, 1); count = zeros(n_train, 1);
        for r = 1:n_repeats_inner
            cvp = cv_partitions{r};
            for k = 1:n_folds_inner
                tr_idx = training(cvp, k); te_idx = test(cvp, k);
                tmp_mdl = fitcsvm(X_sub_train(tr_idx,:), y_train(tr_idx), ...
                    'BoxConstraint', best_bc, 'KernelFunction', 'linear', ...
                    'Prior', 'uniform', 'Standardize', true);
                [~, sc] = predict(tmp_mdl, X_sub_train(te_idx,:));
                score_sum(te_idx) = score_sum(te_idx) + sc(:,2);
                count(te_idx) = count(te_idx) + 1;
            end
        end
        raw_oof_scores = score_sum ./ count;
        
        % C. Test Feats (在所有 N-1 个样本上训练，预测那 1 个测试样本)
        final_L0_model = fitcsvm(X_sub_train, y_train, ...
            'BoxConstraint', best_bc, 'KernelFunction', 'linear', ...
            'Prior', 'uniform', 'Standardize', true);
        [~, sc_test] = predict(final_L0_model, X_sub_test);
        raw_test_scores = sc_test(:, 2);
        
        % D. Scaling
        min_s = min(raw_oof_scores); max_s = max(raw_oof_scores);
        L1_scalers.(curr_group_name).min = min_s; 
        L1_scalers.(curr_group_name).max = max_s;
        if max_s > min_s
            L1_train_feats(:, g) = 2*((raw_oof_scores - min_s)/(max_s - min_s)) - 1;
            L1_test_feats(:, g)  = 2*((raw_test_scores - min_s)/(max_s - min_s)) - 1;
        else
            L1_train_feats(:, g) = 0; L1_test_feats(:, g) = 0;
        end
    end
    
    % =========================================================
    % 3. Level-1 训练 (Meta-Model)
    % =========================================================
    meta_box_constraints = [0.0001, 0.001, 0.01, 0.1, 1, 10];
    best_meta_score = -inf; best_meta_c = 1;
    
    for bc = meta_box_constraints
        bacs_meta_val = []; bacs_meta_tr = [];
        for r = 1:n_repeats_inner
            cvp = cv_partitions{r};
            for k = 1:n_folds_inner
                tr_idx = training(cvp, k); te_idx = test(cvp, k);
                try
                    mdl = fitcsvm(L1_train_feats(tr_idx,:), y_train(tr_idx), ...
                        'KernelFunction', 'linear', 'Prior', 'uniform', ...
                        'Standardize', true, 'BoxConstraint', bc);
                    
                    y_pred = predict(mdl, L1_train_feats(te_idx,:));
                    stats = confusionmat(y_train(te_idx), y_pred);
                    if size(stats,1)==2
                        bac_val = (stats(2,2)/sum(stats(2,:)) + stats(1,1)/sum(stats(1,:)))/2;
                    else; bac_val = 0.5; end
                    bacs_meta_val = [bacs_meta_val; bac_val];
                    
                    y_pred_tr = predict(mdl, L1_train_feats(tr_idx,:));
                    stats_tr = confusionmat(y_train(tr_idx), y_pred_tr);
                    if size(stats_tr,1)==2
                        bac_tr = (stats_tr(2,2)/sum(stats_tr(2,:)) + stats_tr(1,1)/sum(stats_tr(1,:)))/2;
                    else; bac_tr = 0.5; end
                    bacs_meta_tr = [bacs_meta_tr; bac_tr];
                catch; continue; end
            end
        end
        mean_val = mean(bacs_meta_val);
        mean_tr = mean(bacs_meta_tr);
        gap = abs(mean_tr - mean_val);
        score = mean_val - 0.5 * gap;
        if score > best_meta_score; best_meta_score = score; best_meta_c = bc; end
    end
    
    % 4. Final Meta Model Prediction
    % 训练最终元模型 (N-1 样本)
    meta_model = fitcsvm(L1_train_feats, y_train, ...
        'KernelFunction', 'linear', 'Prior', 'uniform', ...
        'Standardize', true, 'BoxConstraint', best_meta_c);
    
    [~, final_scores] = predict(meta_model, L1_test_feats);

    % 转为概率输出 (fitPosterior)
%     meta_model = fitPosterior(meta_model);
%     [~, final_scores] = predict(meta_model, L1_test_feats);
    pred_scores = final_scores(:, 2); % 返回那 1 个测试样本的概率
end