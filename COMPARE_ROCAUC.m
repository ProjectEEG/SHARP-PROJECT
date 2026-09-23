clear;
close all;
clc;

load('AUCPLOT_lasso.mat');
load('AUCPLOT_stack.mat');
load('AUCPLOT_nfs.mat');
load('AUCPLOT_ct.mat');

%% Font settings
font_name = 'Arial';
set(groot, 'defaultAxesFontName', font_name);
set(groot, 'defaultTextFontName', font_name);
set(groot, 'defaultLegendFontName', font_name);

%% Curve colors
color_stack = [0.8500 0.3250 0.0980];
color_lasso = [0.0000 0.4470 0.7410];
color_nfs   = [0.4660 0.6740 0.1880];
color_ct    = [0.4940 0.1840 0.5560];

%% Create figure
fig = figure('Position', [100, 100, 800, 600], 'Color', 'w');
ax = axes(fig);
hold(ax, 'on');

%% ROC curves
h2 = plot(ax, AUCPLOT_lasso.x, AUCPLOT_lasso.y, ...
    'LineWidth', 2.2, 'Color', color_lasso);

h3 = plot(ax, AUCPLOT_nfs.x, AUCPLOT_nfs.y, ...
    'LineWidth', 2.2, 'Color', color_nfs);

h4 = plot(ax, AUCPLOT_ct.x, AUCPLOT_ct.y, ...
    'LineWidth', 2.2, 'Color', color_ct);

h1 = plot(ax, AUCPLOT_stack.x, AUCPLOT_stack.y, ...
    'LineWidth', 3.5, 'Color', color_stack);

%% Diagonal reference line
plot(ax, [0, 1], [0, 1], 'k--', 'LineWidth', 1.3);

%% Legend
lgd = legend(ax, [h3, h2, h4, h1], ...
    'No feature selection (AUC = 0.676)', ...
    'LASSO-based feature selection (AUC = 0.666)', ...
    'Effect-size-based feature selection (AUC = 0.645)', ...
    'DES-based feature selection (AUC = 0.726)', ...
    'Location', 'southeast', ...
    'Box', 'off');

set(lgd, 'FontName', font_name, 'FontSize', 12);

%% Axes style
grid(ax, 'off');
box(ax, 'off');

xlim(ax, [0.0, 1.0]);
ylim(ax, [0.0, 1.0]);
xticks(ax, 0.0:0.2:1.0);
yticks(ax, 0.0:0.2:1.0);

x_tick_labels = arrayfun(@(v) sprintf('%.1f', v), ax.XTick, 'UniformOutput', false);
y_tick_labels = arrayfun(@(v) sprintf('%.1f', v), ax.YTick, 'UniformOutput', false);
y_tick_labels{1} = '';

xticklabels(ax, x_tick_labels);
yticklabels(ax, y_tick_labels);
axis(ax, 'square');

set(ax, ...
    'FontName', font_name, ...
    'FontSize', 14, ...
    'LineWidth', 1.2, ...
    'TickDir', 'out', ...
    'XColor', 'k', ...
    'YColor', 'k');

xlabel(ax, 'False positive rate (1-specificity)', ...
    'FontSize', 16, 'FontName', font_name);

ylabel(ax, 'True positive rate (sensitivity)', ...
    'FontSize', 16, 'FontName', font_name);

lgd.Position = [0.405, 0.12, 0.35, 0.15];

hold(ax, 'off');

%% Save PDF and PNG with extra white margins
set(fig, 'Renderer', 'painters');
drawnow;

out_dir = 'C:\Users\73448\Desktop\paper\chrpaper\figure';

pdf_path = fullfile(out_dir, 'ROC_Feature_Comparison.pdf');
png_path = fullfile(out_dir, 'ROC_Feature_Comparison.png');

tmp_pdf = fullfile(out_dir, 'ROC_Feature_Comparison_tmp.pdf');
tmp_png = fullfile(out_dir, 'ROC_Feature_Comparison_tmp.png');

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

margin_pt = 10;
png_resolution = 300;

%% PDF
exportgraphics(fig, tmp_pdf, ...
    'ContentType', 'vector', ...
    'BackgroundColor', 'white');

pdfcrop_exe = 'D:\texlive\2026\bin\windows\pdfcrop.exe';

old_dir = pwd;

try
    cd(out_dir);

    if exist(pdfcrop_exe, 'file')
        cmd = sprintf('"%s" --margins "%d %d %d %d" "%s" "%s"', ...
            pdfcrop_exe, ...
            margin_pt, margin_pt, margin_pt, margin_pt, ...
            'ROC_Feature_Comparison_tmp.pdf', ...
            'ROC_Feature_Comparison.pdf');
    else
        cmd = sprintf('pdfcrop --margins "%d %d %d %d" "%s" "%s"', ...
            margin_pt, margin_pt, margin_pt, margin_pt, ...
            'ROC_Feature_Comparison_tmp.pdf', ...
            'ROC_Feature_Comparison.pdf');
    end

    [status, cmdout] = system(cmd);
    cd(old_dir);

catch ME
    cd(old_dir);
    rethrow(ME);
end

if status ~= 0
    disp(cmdout);
    error('pdfcrop failed.');
end

if exist(tmp_pdf, 'file')
    delete(tmp_pdf);
end

%% PNG
exportgraphics(fig, tmp_png, ...
    'Resolution', png_resolution, ...
    'BackgroundColor', 'white');

margin_px = round(margin_pt / 72 * png_resolution);

I = imread(tmp_png);

new_h = size(I, 1) + 2 * margin_px;
new_w = size(I, 2) + 2 * margin_px;
num_c = size(I, 3);

if isinteger(I)
    I_pad = intmax(class(I)) * ones(new_h, new_w, num_c, 'like', I);
else
    I_pad = ones(new_h, new_w, num_c, 'like', I);
end

I_pad( ...
    margin_px + 1 : margin_px + size(I, 1), ...
    margin_px + 1 : margin_px + size(I, 2), ...
    :) = I;

imwrite(I_pad, png_path);

if exist(tmp_png, 'file')
    delete(tmp_png);
end

fprintf('Saved PDF: %s\n', pdf_path);
fprintf('Saved PNG: %s\n', png_path);
fprintf('Axes FontName: %s\n', ax.FontName);
fprintf('XLabel FontName: %s\n', ax.XLabel.FontName);
fprintf('YLabel FontName: %s\n', ax.YLabel.FontName);
fprintf('Legend FontName: %s\n', lgd.FontName);
