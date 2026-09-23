%load('',''....)加载脑电特征，均为channel-level，后续为按照62通道-5脑区平均（静息态）/9通道-1脑区平均（任务态），得到相应的regional-level

%1.1 静息态划分脑区通道
frontal = [1:4,11,12,17,21,22,25,26,31,32,35,36,41:44,49,50,55,56];
central = [5,6,18,33,37,38,51,52,23,24,45,46];
parietal = [7,8,15,16,19,39,40,53,54,27,28];
temporal = [57,58,13,14,59,60,29,30];
occipital = [9,10,20,47,48,34,61,62];

%1.2 静息态特征选择。第一层，基于boostrap找到具有稳健表征方向的特征
brain_region_chn = [{frontal},{central},{parietal},{temporal},{occipital}];

%注意oaec和wpli操作稍复杂，这里需要提前进行脑区平均。这里以oaec为例
oaec_chan = oaec_matrix;
oaec_region = zeros(5,5,5,n);% n为被试数，需要自行更换
lower_tri_vector = zeros(15,5,n);%5个脑区的组合数，为15
for sub = 1:n
    for fre = 1:5
        for i = 1:5
            for j = 1:5
                oaec = squeeze(oaec_chan(brain_region_chn{i},brain_region_chn{j},fre,sub));
                if i==j
                % 提取非对角线元素        
                oaec_region(i,j,fre,sub) = mean(oaec(~eye(length(brain_region_chn{i}))),'all');
                else
                oaec_region(i,j,fre,sub) = mean(oaec_chan(brain_region_chn{i},brain_region_chn{j},fre,sub),'all');
                end
            end
        end 
        matrix = squeeze(oaec_region(:,:,fre,sub));
        lower_tri_vector(:,fre,sub) = matrix(tril(true(size(matrix))));
    end
end
oaec_feature = lower_tri_vector;

%之后为第一层特征选择，这里以psd_rel_std为例
all_results = [];
data = psd_rel_std;
for fre = 1:5
    for region = 1:5
        group1 = squeeze(mean(data(fre,brain_region_chn{region},intersect(find(eeg_yind==1),find(eeg_group==1))),2));%发现集的转化组
        group2 = squeeze(mean(data(fre,brain_region_chn{region},intersect(find(eeg_yind==0),find(eeg_group==1))),2));%发现集的未转化组
        results = boostrap_feature_select(group1,group2);
        all_results = cat(1, all_results,results);
    end
end

consistency = [];
for j = 1:size(all_results,1)
    consistency = cat(1,consistency,all_results(j).consistency);%得到每个特征的5000次boostrap的符号一致性系数
end
target = find(consistency>0.8);%选择阈值为0.8

%1.3 静息态特征选择。第二层，检测目标特征的脑区内通道表征方向的一致性

%以psd_rel_std为例
data = psd_rel_std;
prob = zeros(length(target),1);
for j = 1:length(target)
    fre = floor((target(j)-1)/5) + 1;
    region = mod(target(j)-1, 5) + 1;
    group1 = squeeze(mean(data(fre,brain_region_chn{region},intersect(find(eeg_yind==1),find(eeg_group==1))),2));
    group2 = squeeze(mean(data(fre,brain_region_chn{region},intersect(find(eeg_yind==0),find(eeg_group==1))),2));
    chn_effect = [];
    for chn = brain_region_chn{region}
        grp1 = squeeze(data(fre,chn,intersect(find(eeg_yind==1),find(eeg_group==1))));
        grp2 = squeeze(data(fre,chn,intersect(find(eeg_yind==0),find(eeg_group==1))));
        chn_effect = cat(1,chn_effect,cohens_d(grp1,grp2));
    end
    if cohens_d(group1,group2)>0
        effect = chn_effect> 0;
        prob(j) = mean(effect);
    else
        effect = chn_effect< 0;
        prob(j) = mean(effect);
    end
end

%以 oaec特征为例
data = oaec_chan;
prob = zeros(length(target),1);
mask = tril(true(5));
[rows, cols] = find(mask);
for j = 1:length(target)
    fre = floor((target(j)-1)/15) + 1;
    pairs = mod(target(j)-1, 15) + 1;
    row = rows(pairs);
    col = cols(pairs);
    if row==col
        mask = ~eye(length(brain_region_chn{row}));
        oaec = data(brain_region_chn{row},brain_region_chn{col},fre,:);
        data_2d = reshape(oaec, length(brain_region_chn{row})^2, size(oaec,4));
        mask_vector = mask(:);
        group1 = squeeze(mean(data_2d(mask_vector, intersect(find(eeg_yind==1),find(eeg_group==1))),1));
        group2 = squeeze(mean(data_2d(mask_vector, intersect(find(eeg_yind==0),find(eeg_group==1))),1));
        
        chn_effect = [];
        for chn_row = brain_region_chn{row}
            for chn_col = brain_region_chn{col}
                grp1 = squeeze(data(chn_row,chn_col,fre,intersect(find(eeg_yind==1),find(eeg_group==1))));
                grp2 = squeeze(data(chn_row,chn_col,fre,intersect(find(eeg_yind==0),find(eeg_group==1))));
                chn_effect = cat(1,chn_effect,cohens_d(grp1,grp2));
            end
        end
        chn_effect = chn_effect(mask_vector);
    else
        group1 = squeeze(mean(mean(data(brain_region_chn{row},brain_region_chn{col},fre,intersect(find(eeg_yind==1),find(eeg_group==1))),1),2));
        group2 = squeeze(mean(mean(data(brain_region_chn{row},brain_region_chn{col},fre,intersect(find(eeg_yind==0),find(eeg_group==1))),1),2));
        
        chn_effect = [];
        for chn_row = brain_region_chn{row}
            for chn_col = brain_region_chn{col}
                grp1 = squeeze(data(chn_row,chn_col,fre,intersect(find(eeg_yind==1),find(eeg_group==1))));
                grp2 = squeeze(data(chn_row,chn_col,fre,intersect(find(eeg_yind==0),find(eeg_group==1))));
                chn_effect = cat(1,chn_effect,cohens_d(grp1,grp2));
            end
        end
    end

    if cohens_d(group1,group2)>0
        effect = chn_effect> 0;
        prob(j) = mean(effect);
    else
        effect = chn_effect< 0;
        prob(j) = mean(effect);
    end
end

count = sum(prob == 1);%只选择所有相关通道特征的表征方向完全相同的脑区特征，即prob == 1

%2.1 任务态划分脑区通道
load('sharp_feature.mat');% 加载脑电特征，均为channel-level
%对于erp特征，22个特征的通道index（自定），每个特征有9个对应任务的脑区通道的数值
erp_region_chn = [{[1:9]},{[10:18]},{[19:27]},{[28:36]},{[37:45]},{[46:54]},{[55:63]},{[64:72]},{[73:81]},{[82:90]},{[91:99]},{[100:108]},{[109:117]},{[118:126]},{[127:135]},{[136:144]},{[145:153]},{[154:162]},{[163:171]},{[172:180]},{[181:189]},{[190:198]}];

%2.2 任务态特征选择。第一层，基于boostrap找到具有稳健表征方向的特征
data = [];
for i = 1:length(erp_region_chn)
    data = cat(2,data,mean(x_eeg1(:,erp_region_chn{i}),2));%此处x_eeg1为发现集任务态特征，这里对各特征进行脑区平均
end
all_results = [];
for feature = 1:size(data,2)
    group1 = squeeze(data(find(eeg_yind==1),feature));
    group2 = squeeze(data(find(eeg_yind==0),feature));
    results = boostrap_feature_select(group1,group2);
    all_results = cat(1, all_results,results);
end

consistency = [];
for j = 1:size(data,2)
    consistency = cat(1,consistency,all_results(j).consistency);
end
target = find(consistency>0.8);%选择阈值为0.8

%2.3 任务态特征选择。第二层，检测目标特征的脑区内通道表征方向的一致性
prob = zeros(length(target),1);
for j = 1:length(target)
    region = target(j);
    group1 = data(find(eeg_yind==1),region);
    group2 = data(find(eeg_yind==0),region);
    chn_effect = [];
    for chn = erp_region_chn{region}
        grp1 = x_eeg1(find(eeg_yind==1),chn);
        grp2 = x_eeg1(find(eeg_yind==0),chn);
        chn_effect = cat(1,chn_effect,cohens_d(grp1,grp2));
    end
    if cohens_d(group1,group2)>0
        effect = chn_effect> 0;
        prob(j) = mean(effect);
    else
        effect = chn_effect< 0;
        prob(j) = mean(effect);
    end
end
count = sum(prob == 1);%只选择所有相关通道特征的表征方向完全相同的脑区特征，即prob == 1

