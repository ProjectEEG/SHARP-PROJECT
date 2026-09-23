%% Combined DCA figure: Discovery set and validation set
close all;
clear;
clc;

rng(2026);

font_name = 'Arial';
set(groot, 'defaultAxesFontName', font_name);
set(groot, 'defaultTextFontName', font_name);
set(groot, 'defaultLegendFontName', font_name);

%% File settings
left_label_file = 'p2.mat';
left_prob_file  = 'p1.mat';
right_data_file = '57features.mat';

%% Left panel: DCA from saved labels and probabilities
left_dca = build_dca_from_mat( ...
    left_label_file, ...
    left_prob_file, ...
    'full_true_labels', ...
    'full_preds_prob', ...
    'Discovery Set', ...
    'Discovery model' ...
);

%% Right panel: DCA from stacked SVM validation prediction
right_dca = build_dca_from_stacked_svm( ...
    right_data_file, ...
    'Validation Set', ...
    'Validation model' ...
);

%% Save DCA numeric results
write_dca_table(left_dca, 'Discovery_DCA_Net_Benefit_Helvetica.xlsx');
write_dca_table(right_dca, 'Validation_DCA_Net_Benefit_Helvetica.xlsx');

%% Plot two model DCA curves in one figure
fig = figure('Color', 'w', 'Position', [100, 100, 950, 760]);
ax = axes(fig);
plot_dca_combined(ax, left_dca, right_dca, font_name);

set(fig, 'Renderer', 'painters');
drawnow;

try
    exportgraphics( ...
        fig, ...
        'DCA_Combined_Helvetica.pdf', ...
        'ContentType', 'vector', ...
        'BackgroundColor', 'white' ...
    );
catch
    set(fig, 'Units', 'inches');
    fig_pos = get(fig, 'Position');
    set(fig, 'PaperUnits', 'inches');
    set(fig, 'PaperPosition', [0, 0, fig_pos(3), fig_pos(4)]);
    set(fig, 'PaperSize', [fig_pos(3), fig_pos(4)]);
    print(fig, 'DCA_Combined_Helvetica.pdf', '-dpdf', '-painters');
end

fprintf('Combined DCA figure saved as DCA_Combined_Helvetica.pdf\n');

%% ===================== Local functions =====================

function dca = build_dca_from_mat(label_file, prob_file, label_var, prob_var, panel_title, model_label)
    S_label = load(label_file);
    S_prob  = load(prob_file);

    true_labels = get_vector_from_mat(S_label, label_var);
    pred_prob   = get_vector_from_mat(S_prob, prob_var);

    true_labels = double(true_labels(:));
    pred_prob   = double(pred_prob(:));
    pred_prob   = min(max(pred_prob, 0), 1);

    if numel(true_labels) ~= numel(pred_prob)
        error( ...
            'Label and probability lengths are inconsistent: %d vs %d.', ...
            numel(true_labels), ...
            numel(pred_prob) ...
        );
    end

    classes = sort(unique(true_labels));
    pos_class = classes(end);

    [xroc, yroc, ~, auc_value] = perfcurve(true_labels, pred_prob, pos_class);
    [xroc, yroc] = add_roc_origin(xroc, yroc);

    fprintf('%s AUC = %.4f\n', panel_title, auc_value);

    dca = compute_dca(true_labels == pos_class, pred_prob);
    dca.panel_title = panel_title;
    dca.model_label = model_label;
    dca.auc = auc_value;
    dca.xroc = xroc;
    dca.yroc = yroc;
end

function dca = build_dca_from_stacked_svm(data_file, panel_title, model_label)
    S = load(data_file);

    X_train_data = S.X_train;
    X_test_data  = S.X_test;
    y_train = double(S.y1(:));
    y_test  = double(S.y2(:));

    all_classes = sort(unique([y_train; y_test]));
    neg_class = all_classes(1);
    pos_class = all_classes(end);

    cols_cog = 1:6;
    cols_sym = 7:17;
    cols_eeg = 18:26;
    cols_eeg_rs = 27:57;

    y_train_cat = categorical(y_train);
    [idx, ~] = relieff(X_train_data(:, cols_eeg_rs), y_train_cat, 15);
    n_select_rs = floor(numel(cols_eeg_rs) / 2);
    cols_eeg_rs = idx(1:n_select_rs) + cols_eeg_rs(1) - 1;

    feature_groups = {cols_cog, cols_sym, cols_eeg, cols_eeg_rs};
    group_names = {'Cognitive', 'Symptoms', 'ERP', 'RS'};

    n_train = size(X_train_data, 1);
    n_test  = size(X_test_data, 1);
    n_groups = numel(feature_groups);

    L1_train_feats = zeros(n_train, n_groups);
    L1_test_feats  = zeros(n_test, n_groups);

    n_repeats = 10;
    n_folds = 5;
    cv_partitions = cell(n_repeats, 1);

    for r = 1:n_repeats
        cv_partitions{r} = cvpartition(y_train, 'KFold', n_folds);
    end

    box_constraints = [0.0001, 0.001, 0.01, 0.1, 1, 10];

    for g = 1:n_groups
        fprintf('Processing level-0 group: %s\n', group_names{g});

        X_sub_train = X_train_data(:, feature_groups{g});
        X_sub_test  = X_test_data(:, feature_groups{g});

        best_bc = choose_best_svm_c( ...
            X_sub_train, ...
            y_train, ...
            cv_partitions, ...
            box_constraints, ...
            neg_class, ...
            pos_class ...
        );

        score_sum = zeros(n_train, 1);
        count = zeros(n_train, 1);

        for r = 1:n_repeats
            cvp = cv_partitions{r};

            for k = 1:n_folds
                tr_idx = training(cvp, k);
                te_idx = test(cvp, k);

                tmp_mdl = fitcsvm( ...
                    X_sub_train(tr_idx, :), ...
                    y_train(tr_idx), ...
                    'BoxConstraint', best_bc, ...
                    'KernelFunction', 'linear', ...
                    'Prior', 'uniform', ...
                    'Standardize', true ...
                );

                [~, sc] = predict(tmp_mdl, X_sub_train(te_idx, :));
                pos_col = find_score_column(tmp_mdl.ClassNames, pos_class, size(sc, 2));

                score_sum(te_idx) = score_sum(te_idx) + sc(:, pos_col);
                count(te_idx) = count(te_idx) + 1;
            end
        end

        raw_scores_oof = score_sum ./ count;

        final_mdl = fitcsvm( ...
            X_sub_train, ...
            y_train, ...
            'BoxConstraint', best_bc, ...
            'KernelFunction', 'linear', ...
            'Prior', 'uniform', ...
            'Standardize', true ...
        );

        [~, sc_test] = predict(final_mdl, X_sub_test);
        pos_col_test = find_score_column(final_mdl.ClassNames, pos_class, size(sc_test, 2));
        raw_scores_test = sc_test(:, pos_col_test);

        min_s = min(raw_scores_oof);
        max_s = max(raw_scores_oof);

        if max_s > min_s
            L1_train_feats(:, g) = 2 * ((raw_scores_oof - min_s) / (max_s - min_s)) - 1;
            L1_test_feats(:, g)  = 2 * ((raw_scores_test - min_s) / (max_s - min_s)) - 1;
        else
            L1_train_feats(:, g) = 0;
            L1_test_feats(:, g) = 0;
        end
    end

    meta_box_constraints = [0.0001, 0.001, 0.01, 0.1, 1, 10];
    best_meta_c = choose_best_svm_c( ...
        L1_train_feats, ...
        y_train, ...
        cv_partitions, ...
        meta_box_constraints, ...
        neg_class, ...
        pos_class ...
    );

    meta_model = fitcsvm( ...
        L1_train_feats, ...
        y_train, ...
        'KernelFunction', 'linear', ...
        'Prior', 'uniform', ...
        'Standardize', true, ...
        'BoxConstraint', best_meta_c ...
    );

    meta_model = fitPosterior(meta_model);
    [~, pred_scores] = predict(meta_model, L1_test_feats);

    pos_col = find_score_column(meta_model.ClassNames, pos_class, size(pred_scores, 2));
    pos_score = pred_scores(:, pos_col);
    pos_score = min(max(pos_score(:), 0), 1);

    [xroc, yroc, ~, auc_value] = perfcurve(y_test(:), pos_score(:), pos_class);
    [xroc, yroc] = add_roc_origin(xroc, yroc);

    fprintf('%s AUC = %.4f\n', panel_title, auc_value);

    dca = compute_dca(y_test(:) == pos_class, pos_score);
    dca.panel_title = panel_title;
    dca.model_label = model_label;
    dca.auc = auc_value;
    dca.xroc = xroc;
    dca.yroc = yroc;
end

function best_bc = choose_best_svm_c(X, y, cv_partitions, box_constraints, neg_class, pos_class)
    best_bac = -inf;
    best_bc = box_constraints(1);

    for bc = box_constraints
        bacs = [];

        for r = 1:numel(cv_partitions)
            cvp = cv_partitions{r};

            for k = 1:cvp.NumTestSets
                tr_idx = training(cvp, k);
                te_idx = test(cvp, k);

                try
                    mdl = fitcsvm( ...
                        X(tr_idx, :), ...
                        y(tr_idx), ...
                        'KernelFunction', 'linear', ...
                        'Prior', 'uniform', ...
                        'Standardize', true, ...
                        'BoxConstraint', bc ...
                    );

                    y_pred = predict(mdl, X(te_idx, :));
                    cm = confusionmat(y(te_idx), y_pred, 'Order', [neg_class, pos_class]);

                    TN = cm(1, 1);
                    FP = cm(1, 2);
                    FN = cm(2, 1);
                    TP = cm(2, 2);

                    sens = safe_divide(TP, TP + FN);
                    spec = safe_divide(TN, TN + FP);
                    bacs = [bacs; (sens + spec) / 2]; %#ok<AGROW>
                catch
                    continue;
                end
            end
        end

        if isempty(bacs)
            mean_bac = 0.5;
        else
            mean_bac = mean(bacs);
        end

        if mean_bac > best_bac
            best_bac = mean_bac;
            best_bc = bc;
        end
    end
end

function dca = compute_dca(y_binary, pred_prob)
    y_binary = double(y_binary(:));
    pred_prob = double(pred_prob(:));

    if numel(y_binary) ~= numel(pred_prob)
        error('DCA input lengths are inconsistent.');
    end

    pred_prob = min(max(pred_prob, 0), 1);

    n = numel(y_binary);
    prevalence = sum(y_binary == 1) / n;
    thresholds = 0.01:0.01:0.99;

    model_nb = zeros(numel(thresholds), 1);
    treat_all_nb = zeros(numel(thresholds), 1);
    treat_none_nb = zeros(numel(thresholds), 1);

    for i = 1:numel(thresholds)
        pt = thresholds(i);
        pred = double(pred_prob >= pt);

        TP = sum((pred == 1) & (y_binary == 1));
        FP = sum((pred == 1) & (y_binary == 0));

        model_nb(i) = TP / n - FP / n * pt / (1 - pt);
        treat_all_nb(i) = prevalence - (1 - prevalence) * pt / (1 - pt);
        treat_none_nb(i) = 0;
    end

    smooth_window = 9;
    model_nb_plot = smooth_dca_curve(model_nb, smooth_window);

    dca.thresholds = thresholds(:);
    dca.model_nb_raw = model_nb(:);
    dca.model_nb_plot = model_nb_plot(:);
    dca.treat_all_nb = treat_all_nb(:);
    dca.treat_none_nb = treat_none_nb(:);
end

function plot_dca_combined(ax, left_dca, right_dca, font_name)
    axes(ax);

    color_discovery  = [107 91 139] / 255;
    color_validation = [232 184 122] / 255;
    color_treat_all  = [0 0 0] ;
    color_treat_none = [0.45 0.45 0.45];
    
    h_none = plot( ...
        ax, ...
        left_dca.thresholds, ...
        left_dca.treat_none_nb, ...
        '--', ...
        'LineWidth', 1.8, ...
        'Color', color_treat_none ...
    );

    hold(ax, 'on');
    
    h_all = plot( ...
        ax, ...
        left_dca.thresholds, ...
        left_dca.treat_all_nb, ...
        '--', ...
        'LineWidth', 2.8, ...
        'Color', color_treat_all ...
    );

    h_left = plot( ...
        ax, ...
        left_dca.thresholds, ...
        left_dca.model_nb_plot, ...
        'LineWidth', 2.8, ...
        'Color', color_discovery ...
    );

    h_right = plot( ...
        ax, ...
        right_dca.thresholds, ...
        right_dca.model_nb_plot, ...
        'LineWidth', 2.8, ...
        'Color', color_validation ...
    );

    hold(ax, 'off');

    xlim(ax, [0.0, 0.7]);
    xticks(ax, 0.0:0.1:0.7);

    ylim(ax, [-0.1, 0.4]);
    yticks(ax, -0.1:0.1:0.4);

    format_clean_axes(ax, font_name);
    ax.FontSize = 16;
    ax.LineWidth = 2.0;
    ax.Position = [0.15, 0.16, 0.78, 0.72];

    xlabel(ax, 'Threshold probability', ...
        'FontSize', 20, ...
        'FontName', font_name);

    ylabel(ax, 'Net benefit', ...
        'FontSize', 20, ...
        'FontName', font_name);

    legend(ax, ...
        [h_left, h_right, h_all, h_none], ...
        {left_dca.model_label, right_dca.model_label, 'Treat all', 'Treat none'}, ...
        'Location', 'best', ...
        'FontSize', 18, ...
        'FontName', font_name, ...
        'Box', 'off');

    format_axis_tick_1decimal(ax);
end

function write_dca_table(dca, filename)
    dca_table = table( ...
        dca.thresholds(:), ...
        dca.model_nb_raw(:), ...
        dca.model_nb_plot(:), ...
        dca.treat_all_nb(:), ...
        dca.treat_none_nb(:), ...
        'VariableNames', { ...
            'Threshold_Probability', ...
            'Model_NetBenefit_Raw_for_CHR_Conversion', ...
            'Model_NetBenefit_Smoothed_for_CHR_Conversion', ...
            'Treat_All_as_Conversion_NetBenefit', ...
            'Treat_None_as_Conversion_NetBenefit' ...
        } ...
    );

    writetable(dca_table, filename);
    fprintf('DCA Net Benefit saved to %s\n', filename);
end

function vec = get_vector_from_mat(S, preferred_name)
    if isfield(S, preferred_name)
        vec = S.(preferred_name);
        return;
    end

    names = fieldnames(S);

    for i = 1:numel(names)
        v = S.(names{i});

        if isnumeric(v) && isvector(v)
            fprintf('Variable "%s" was not found. Using "%s" instead.\n', preferred_name, names{i});
            vec = v;
            return;
        end
    end

    error('No numeric vector variable was found in the MAT file.');
end

function y_smooth = smooth_dca_curve(y_raw, smooth_window)
    y_raw = y_raw(:);

    if smooth_window < 3
        y_smooth = y_raw;
        return;
    end

    if mod(smooth_window, 2) == 0
        smooth_window = smooth_window + 1;
    end

    try
        y_smooth = smoothdata(y_raw, 'loess', smooth_window);
    catch
        warning('smoothdata loess is unavailable. Falling back to movmean.');
        y_smooth = smoothdata(y_raw, 'movmean', smooth_window);
    end

    y_smooth = y_smooth(:);
end

function format_clean_axes(ax, font_name)
    ax.TickDir = 'out';
    ax.Box = 'off';

    ax.XGrid = 'off';
    ax.YGrid = 'off';
    ax.XMinorGrid = 'off';
    ax.YMinorGrid = 'off';

    ax.XAxisLocation = 'bottom';
    ax.YAxisLocation = 'left';

    ax.LineWidth = 1.2;
    ax.FontName = font_name;
    ax.XMinorTick = 'off';
    ax.YMinorTick = 'off';
    ax.Layer = 'top';
    ax.XAxis.Exponent = 0;
    ax.YAxis.Exponent = 0;
end

function format_axis_tick_1decimal(ax)
    ax.XTickLabel = arrayfun(@(v) sprintf('%.1f', v), ax.XTick, 'UniformOutput', false);
    ax.YTickLabel = arrayfun(@(v) sprintf('%.1f', v), ax.YTick, 'UniformOutput', false);
end

function [xroc, yroc] = add_roc_origin(xroc, yroc)
    xroc = xroc(:);
    yroc = yroc(:);

    if isempty(xroc) || isempty(yroc)
        return;
    end

    if xroc(1) ~= 0 || yroc(1) ~= 0
        xroc = [0; xroc];
        yroc = [0; yroc];
    end
end

function pos_col = find_score_column(class_names, pos_class, n_score_cols)
    pos_col = [];

    try
        pos_col = find(class_names == pos_class, 1);
    catch
        if iscell(class_names)
            pos_col = find(strcmp(class_names, num2str(pos_class)), 1);
        end
    end

    if isempty(pos_col)
        pos_col = n_score_cols;
    end
end

function val = safe_divide(a, b)
    if b == 0
        val = 0;
    else
        val = a / b;
    end
end