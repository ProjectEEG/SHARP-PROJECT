%% 1. 数据载入与预处理
close all;
clc;
clear;
% --- 设置随机种子保证结果可复现 ---
rng(2026);

load('57features.mat'); 

% --- 格式对齐 ---
y_train = double(y1');
y_test  = double(y2'); 
X_train_data = X_train;
X_test_data  = X_test;
feature1 = name;
% --- 定义特征模态 ---
% (保留你提供的定义，注意：如果有重复定义的变量，MATLAB会使用最后一次赋值)
cols_cog = 1:6; 
cols_sym = 7:17;
cols_eeg = 18:26; 
cols_eeg1 = 27:57;
cols_eeg_rs1 = 27:57;
y_train_cat = categorical(y_train);
[idx, weights] = relieff(X_train_data(:, cols_eeg_rs1), y_train_cat, 15);
cols_eeg_rs1 = idx(1:length(cols_eeg_rs1)/2)+cols_eeg_rs1(1)-1;

feature_groups = {cols_cog, cols_sym, cols_eeg, cols_eeg_rs1};

group_names = {'Cognitive', 'Symptoms', 'ERP', 'RS'};

num_groups = length(feature_groups);

n_train = size(X_train_data, 1);
n_test  = size(X_test_data, 1);

% --- 初始化 Level-1 特征矩阵 ---
L1_train_feats = zeros(n_train, num_groups);
L1_test_feats  = zeros(n_test, num_groups);
final_models = struct(); 

fprintf('=== 开始堆叠泛化 (Stacked Generalization) ===\n');

% 【关键步骤 1】预先生成固定的 CV 划分结构
n_repeats = 5; 
n_folds = 5;
cv_partitions = cell(n_repeats, 1);
for r = 1:n_repeats
    cv_partitions{r} = cvpartition(y_train, 'KFold', n_folds);
end
fprintf('已生成并锁定 %dx%d CV 结构。\n', n_repeats, n_folds);

%% 2. Level-0: 训练基模型 (生成 OOF 特征)
for g = 1:num_groups
    curr_group_name = group_names{g};
    fprintf('\n>> 处理 Level-0 模态: %s ...\n', curr_group_name);
    X_sub_train = X_train_data(:, feature_groups{g});
    X_sub_test  = X_test_data(:, feature_groups{g});
    
    % --- A. 网格搜索 (仅使用 Test Fold BAC) ---
    box_constraints = [ 0.0001, 0.001, 0.01, 0.1, 1, 10]; 
    best_bac = -inf; 
    best_bc = 1;
    
    for bc = box_constraints
        bacs_temp = [];
        for r = 1:n_repeats
            cvp = cv_partitions{r};
            for k = 1:n_folds
                tr_idx = training(cvp, k);
                te_idx = test(cvp, k);

                try
                    mdl = fitcsvm(X_sub_train(tr_idx,:), y_train(tr_idx), ...
                        'KernelFunction', 'linear', 'Prior', 'uniform', ...
                        'Standardize', true, 'BoxConstraint', bc);
                    
                    % 预测验证折
                    y_pred = predict(mdl, X_sub_train(te_idx, :));
                    
                    % 计算 BAC
                    stats = confusionmat(y_train(te_idx), y_pred);
                    if size(stats,1)==2
                        bac_iter = (stats(2,2)/sum(stats(2,:)) + stats(1,1)/sum(stats(1,:)))/2;
                    else
                        bac_iter = 0.5;
                    end
                    bacs_temp = [bacs_temp; bac_iter];
                catch; continue; end
            end
        end
        % 选择平均 BAC 最高的参数
        if mean(bacs_temp) > best_bac
            best_bac = mean(bacs_temp); 
            best_bc = bc; 
        end
    end
    fprintf('   最优 C: %.4f (CV Test BAC: %.4f)\n', best_bc, best_bac);
    
    % --- B. 生成 OOF 特征 (Level-1 的训练数据) ---
    score_sum = zeros(n_train, 1); 
    count = zeros(n_train, 1);
    
    for r = 1:n_repeats
        cvp = cv_partitions{r};
        for k = 1:n_folds
            tr_idx = training(cvp, k); 
            te_idx = test(cvp, k);
            
            tmp_mdl = fitcsvm(X_sub_train(tr_idx,:), y_train(tr_idx), ...
                'BoxConstraint', best_bc, 'KernelFunction', 'linear', ...
                'Prior', 'uniform', 'Standardize', true);
            
            [~, sc] = predict(tmp_mdl, X_sub_train(te_idx,:));
            score_sum(te_idx) = score_sum(te_idx) + sc(:,2);
            count(te_idx) = count(te_idx) + 1;
        end
    end
    raw_scores_oof = score_sum ./ count;
    
    % --- C. 生成 Test 特征 (Level-1 的测试数据) ---
    final_models.(curr_group_name).model = fitcsvm(X_sub_train, y_train, ...
        'BoxConstraint', best_bc, 'KernelFunction', 'linear', ...
        'Prior', 'uniform', 'Standardize', true);
        
    [~, sc_test] = predict(final_models.(curr_group_name).model, X_sub_test);
    raw_scores_test = sc_test(:, 2);
    
    % --- D. 归一化 (Scaling to [-1, 1]) ---
    min_s = min(raw_scores_oof); 
    max_s = max(raw_scores_oof);
    final_models.(curr_group_name).min_s = min_s; 
    final_models.(curr_group_name).max_s = max_s;
    
    if max_s > min_s
        L1_train_feats(:, g) = 2*((raw_scores_oof - min_s)/(max_s - min_s)) - 1;
        L1_test_feats(:, g)  = 2*((raw_scores_test - min_s)/(max_s - min_s)) - 1;
    else
        L1_train_feats(:, g) = 0; 
        L1_test_feats(:, g) = 0;
    end
end

%% 3. Level-1: 元模型优化与训练
fprintf('\n>> 优化并训练 Level-1 元模型 ...\n');
meta_box_constraints = [ 0.0001, 0.001, 0.01, 0.1, 1, 10];
best_meta_bac = -inf; 
best_meta_c = 1;

for bc = meta_box_constraints
    bacs_meta = [];
    for r = 1:n_repeats
        cvp = cv_partitions{r};
        for k = 1:n_folds
            tr_idx = training(cvp, k);
            te_idx = test(cvp, k);
            
            try
                % 训练元模型
                mdl = fitcsvm(L1_train_feats(tr_idx,:), y_train(tr_idx), ...
                    'KernelFunction', 'linear', 'Prior', 'uniform', ...
                    'Standardize', true, 'BoxConstraint', bc);
                
                y_pred = predict(mdl, L1_train_feats(te_idx,:));
                
                stats = confusionmat(y_train(te_idx), y_pred);
                if size(stats,1)==2
                    bac_iter = (stats(2,2)/sum(stats(2,:)) + stats(1,1)/sum(stats(1,:)))/2;
                else
                    bac_iter = 0.5;
                end
                bacs_meta = [bacs_meta; bac_iter];
            catch; continue; end
        end
    end
    if mean(bacs_meta) > best_meta_bac
        best_meta_bac = mean(bacs_meta); 
        best_meta_c = bc; 
    end
end
fprintf('   Level-1 最优 C: %.4f (CV Test BAC: %.4f)\n', best_meta_c, best_meta_bac);

% 使用最优 C 训练最终元模型
meta_model = fitcsvm(L1_train_feats, y_train, ...
    'KernelFunction', 'linear', 'Prior', 'uniform', ...
    'Standardize', true, 'BoxConstraint', best_meta_c);

%% 4. 外部验证与可视化 (保存Excel并保留4位小数)
fprintf('\n>> 执行外部验证 (External Validation)...\n');

% 获取后验概率
meta_model = fitPosterior(meta_model);
[~, pred_scores] = predict(meta_model, L1_test_feats);
pos_score = pred_scores(:, 2); 

% 计算 AUC (整体指标，不随阈值变化)
classes = unique(y_test);
pos_class = max(classes); % 假设较大的标签为正类
neg_class = min(classes);
[Xroc, Yroc, ~, AUC] = perfcurve(y_test, pos_score, pos_class);
AUCPLOT_stack.x =  Xroc;
AUCPLOT_stack.y =  Yroc;
save('AUCPLOT_stack.mat', 'AUCPLOT_stack');
% 定义阈值列表
thresholds = [0.05, 0.10, 0.15, 0.2, 0.25,0.26,0.27,0.28, 0.29,0.3,0.31,0.32,0.33,0.34, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]; 
num_thresh = length(thresholds);

% --- 初始化矩阵存储结果 ---
% 列顺序: Threshold, AUC, BAC, ACC, SENSI, SPECI, PPV, NPV
results_mat = zeros(num_thresh, 8);

for i = 1:num_thresh
    ts = thresholds(i);
    
    % 生成二值化预测
    pred_label = double(pos_score > ts);
    % 将 0/1 映射回原始标签 (如果原始标签不是0/1)
    final_preds = zeros(size(pred_label));
    final_preds(pred_label==1) = pos_class;
    final_preds(pred_label==0) = neg_class;
    
    % 计算混淆矩阵 (显式指定顺序: [Neg, Pos])
    conf_mat = confusionmat(y_test, final_preds, 'Order', [neg_class, pos_class]);
    
    % 提取 TN, FP, FN, TP
    TN = conf_mat(1,1); FP = conf_mat(1,2);
    FN = conf_mat(2,1); TP = conf_mat(2,2);
    
    % 计算指标
    SENS_val = TP / (TP + FN);              
    SPEC_val = TN / (TN + FP);              
    BAC_val  = (SENS_val + SPEC_val) / 2;           
    ACC_val  = (TP + TN) / sum(conf_mat(:));
    PPV_val  = TP / (TP + FP);              
    NPV_val  = TN / (TN + FN);              
    
    % 处理分母为0导致的 NaN
    if isnan(PPV_val), PPV_val = 0; end
    if isnan(NPV_val), NPV_val = 0; end
    if isnan(SENS_val), SENS_val = 0; end
    if isnan(SPEC_val), SPEC_val = 0; end
    
    % --- 存入矩阵 (保留4位小数) ---
    row_data = [ts, AUC, BAC_val, ACC_val, SENS_val, SPEC_val, PPV_val, NPV_val];
    results_mat(i, :) = round(row_data, 4);
    
    fprintf('Threshold: %.2f -> BAC: %.4f, ACC: %.4f\n', ts, round(BAC_val,4), round(ACC_val,4));
end

% --- 将矩阵转换为 Table 并保存为 Excel ---
col_names = {'Threshold', 'AUC', 'BAC', 'ACC', 'SENSI', 'SPECI', 'PPV', 'NPV'};
results_table = array2table(results_mat, 'VariableNames', col_names);

output_filename = 'Test_Metrics_by_Threshold.xlsx';
try
    writetable(results_table, output_filename);
    fprintf('\n✅ 结果已保存至 Excel 文件: %s (已保留4位小数)\n', output_filename);
catch ME
    fprintf('\n❌ 保存 Excel 失败 (请确保文件未被打开): %s\n', ME.message);
end

% --- 可视化 (展示 ACC 最高或默认 0.5 的结果) ---
figure('Position', [100, 100, 1000, 400]);

% 子图1: ROC 曲线
subplot(1, 2, 1);
plot(Xroc, Yroc, 'LineWidth', 2, 'Color', [0.8500 0.3250 0.0980]);
AUCPLOT_all_dis.x =  Xroc;
AUCPLOT_all_dis.y =  Yroc;
save('AUCPLOT_all_dis.mat', 'AUCPLOT_all_dis');
hold on; plot([0,1],[0,1],'k--'); hold off;
title(['ROC Curve (AUC = ' num2str(AUC,'%.4f') ')'], 'FontSize', 12, 'FontWeight', 'bold');
xlabel('False Positive Rate (1-Spec)'); ylabel('True Positive Rate (Sens)');
grid on; axis square;

% 子图2: 混淆矩阵 (默认展示 Threshold=0.4 的结果作为示例)
subplot(1, 2, 2);
idx_disp = find(thresholds == 0.4);
if isempty(idx_disp), idx_disp = round(num_thresh/2); end
ts_disp = thresholds(idx_disp);

% 重新生成用于绘图的标签
pred_label_disp = double(pos_score > ts_disp);
final_preds_disp = zeros(size(pred_label_disp));
final_preds_disp(pred_label_disp==1) = pos_class;
final_preds_disp(pred_label_disp==0) = neg_class;

cm = confusionchart(y_test, final_preds_disp, ...
    'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
cm.Title = sprintf('Confusion Matrix (Threshold=%.2f)', ts_disp);

function d = calCohensd(A, B)
    m1 = mean(A);
    m2 = mean(B);
    s1 = std(A);
    s2 = std(B);
    n1 = length(A);
    n2 = length(B);
    s_pooled = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1 + n2 - 2));
    d = (m1 - m2) / s_pooled;
end
