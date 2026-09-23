function results = boostrap_feature_select(group1,group2)
%boostrap特征选择
boot_cohen_d = zeros(5000, 1);
for i = 1:5000
    boot_group1 = datasample(group1, length(group1), 'Replace', true);
    boot_group2 = datasample(group2, length(group2), 'Replace', true);
    % 计算效应量
    boot_cohen_d(i) = cohens_d(boot_group1,boot_group2);
end
positive_effects = boot_cohen_d > 0;
negative_effects = boot_cohen_d < 0;
    
prop_positive = mean(positive_effects);
prop_negative = mean(negative_effects);
    
if prop_positive >= prop_negative
     dominant_direction = 'positive';
     consistency = prop_positive;
     effects = boot_cohen_d(positive_effects);
else
     dominant_direction = 'negative';
     consistency = prop_negative;
     effects = boot_cohen_d(negative_effects);
end
   % 置信区间
  ci_95 = prctile(boot_cohen_d, [2.5, 97.5]);
  ci_90 = prctile(boot_cohen_d, [5, 95]);
  % 存储结果
  results = struct();
  results.boot_cohen_d = mean(boot_cohen_d);
  results.dominant_direction = dominant_direction;
  results.consistency = consistency;
  results.ci_95 = ci_95;
  results.ci_90 = ci_90;
  results.effects = effects;
    
