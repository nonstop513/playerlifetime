# %%
# -*- coding: utf-8 -*-
"""
玩家生命周期预测 - 二分类版本
目标：预测玩家是否流失（0: 流失, 1: 留存）
流失定义：剩余转数 <= 100
"""

import sys
import io

# 设置标准输出编码为UTF-8（仅在终端环境下）
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import duckdb
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_recall_fscore_support
import xgboost as xgb

# %%
# 读取数据
db_path = r"D:\IGame\研究\玩家生命週期\101003_20260108_20260318.duckdb"
con = duckdb.connect(db_path, read_only=True)
df_variableX = con.execute("""
    SELECT *
    FROM main.VariableX
""").fetchdf()
con.close()
df_variable = df_variableX.iloc[:, [0,1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,18,19,20,21,25]].copy()

# %%
# 数据过滤和特征工程
df_mech = df_variable[
    (df_variable.iloc[:, 2] == "OKBT") &
    (df_variable['SpinCount'].between(15, 300000))
].copy()
df_mech['Balanceset'] = df_mech['BalanceCount']/df_mech['SpinCount']

# 统计每个玩家的游玩天数和日期范围
player_summary = df_mech.groupby('PlayerID').agg(
    总游玩天数=('Date', 'count'),
    开始日期=('Date', 'min'),
    结束日期=('Date', 'max')
).reset_index()
player_summary = player_summary.sort_values('总游玩天数', ascending=True)

# 新增已游玩总转数和剩余总转数栏位
df_mech = df_mech.sort_values(['PlayerID', 'Date']).reset_index(drop=True)
df_mech['played_spin'] = df_mech.groupby('PlayerID')['SpinCount'].cumsum() - df_mech['SpinCount']
df_mech['full_spin'] = df_mech.groupby('PlayerID')['SpinCount'].transform('sum')
df_mech['Remaining_spin'] = df_mech['full_spin'] - df_mech['played_spin'] - df_mech['SpinCount']
df_mech['played_days'] = df_mech.groupby('PlayerID').cumcount()  # 过去游玩天数（第一天为0）

# 计算iloc[:, 4:21]栏位的过去平均值
cols_to_avg = df_mech.columns[4:21]
for col in cols_to_avg:
    # 计算累计和与累计计数（不包含当天）
    df_mech[f'{col}_cumsum'] = df_mech.groupby('PlayerID')[col].cumsum() - df_mech[col]
    df_mech[f'{col}_cumcount'] = df_mech.groupby('PlayerID').cumcount()
    # 计算过去平均值
    df_mech[f'{col}_past'] = df_mech[f'{col}_cumsum'] / df_mech[f'{col}_cumcount']
    # 如果过去没有资料（第一天），用当天资料填充
    df_mech[f'{col}_past'] = df_mech[f'{col}_past'].fillna(df_mech[col])
    # 删除临时栏位
    df_mech.drop([f'{col}_cumsum', f'{col}_cumcount'], axis=1, inplace=True)

# 过滤掉在最新7天有游玩记录的玩家
df_mech['Date'] = pd.to_datetime(df_mech['Date'])
max_date = df_mech['Date'].max()
exclude_start_date = max_date - pd.Timedelta(days=6)  # 最新7天
exclude_players = df_mech[
    (df_mech['Date'] >= exclude_start_date) & 
    (df_mech['Date'] <= max_date)
]['PlayerID'].unique()
df_mech1 = df_mech[~df_mech['PlayerID'].isin(exclude_players)].copy()

# %%
# 选择使用的数据集
use_filtered = False  # True使用df_mech1，False使用df_mech
df_model = df_mech1 if use_filtered else df_mech

print(f"数据集选择：")
print(f"df_mech样本数: {len(df_mech)}, 玩家数: {df_mech['PlayerID'].nunique()}")
print(f"df_mech1样本数: {len(df_mech1)}, 玩家数: {df_mech1['PlayerID'].nunique()}")
print(f"当前使用: {'df_mech1 (排除最新7天玩家)' if use_filtered else 'df_mech (全部玩家)'}")

# 按玩家ID拆分训练集和验证集
df_model = df_model.copy()
df_model['PlayerID_int'] = df_model['PlayerID'].astype(int)
train_mask = df_model['PlayerID_int'] % 5 != 4
df_mech_train = df_model[train_mask].copy()
df_mech_valid = df_model[~train_mask].copy()

# 定义预测特征和目标变量
base_cols = list(df_model.columns[4:22]) + list(df_model.columns[25:42])
exclude_features = ['full_spin', 'Remaining_spin', 'PlayerID_int']
X_cols = [col for col in base_cols if col not in exclude_features] + ['played_days']

# %%
# 创建二分类目标变量
def create_binary_target(y):
    """
    将剩余转数转换为二分类
    0: 流失 (0~100转)
    1: 留存 (>100转)
    """
    return (y > 100).astype(int)

y_col = 'is_active'  # 二分类目标变量
df_model[y_col] = create_binary_target(df_model['Remaining_spin'])
df_mech_train[y_col] = create_binary_target(df_mech_train['Remaining_spin'])
df_mech_valid[y_col] = create_binary_target(df_mech_valid['Remaining_spin'])

print(f"\n目标变量: {y_col} (二分类)")
print(f"  0 = 流失 (剩余转数 0~100)")
print(f"  1 = 留存 (剩余转数 >100)")
print(f"\n特征数量: {len(X_cols)}")
print(f"排除的特征: {[col for col in base_cols if col in exclude_features]}")

print(f"\n目标变量分布（训练集）：")
churn_count = (df_mech_train[y_col] == 0).sum()
active_count = (df_mech_train[y_col] == 1).sum()
print(f"  类别0 (流失): {churn_count} ({churn_count/len(df_mech_train)*100:.1f}%)")
print(f"  类别1 (留存): {active_count} ({active_count/len(df_mech_train)*100:.1f}%)")

print(f"\n目标变量分布（验证集）：")
churn_count_v = (df_mech_valid[y_col] == 0).sum()
active_count_v = (df_mech_valid[y_col] == 1).sum()
print(f"  类别0 (流失): {churn_count_v} ({churn_count_v/len(df_mech_valid)*100:.1f}%)")
print(f"  类别1 (留存): {active_count_v} ({active_count_v/len(df_mech_valid)*100:.1f}%)")

# 准备训练和验证数据
X_train = df_mech_train[X_cols]
y_train = df_mech_train[y_col]
X_valid = df_mech_valid[X_cols]
y_valid = df_mech_valid[y_col]

print(f"\n训练集大小: {len(X_train)}")
print(f"验证集大小: {len(X_valid)}")

# %%
# ========================================
# 模型1: 随机森林 (Random Forest)
# ========================================
print("\n" + "="*60)
print("模型1: 随机森林 (Random Forest)")
print("="*60)

rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_valid)
y_pred_proba_rf = rf_clf.predict_proba(X_valid)[:, 1]

# 保存预测结果到验证集
df_mech_valid['y_pred_rf'] = y_pred_rf

# 评估指标
accuracy_rf = accuracy_score(y_valid, y_pred_rf)
auc_rf = roc_auc_score(y_valid, y_pred_proba_rf)
precision_rf, recall_rf, f1_rf, _ = precision_recall_fscore_support(y_valid, y_pred_rf, average='binary')

print(f"\n整体准确率: {accuracy_rf*100:.2f}%")
print(f"AUC-ROC: {auc_rf:.4f}")
print(f"精确率 (Precision): {precision_rf:.4f}")
print(f"召回率 (Recall): {recall_rf:.4f}")
print(f"F1分数: {f1_rf:.4f}")

# 混淆矩阵
cm_rf = confusion_matrix(y_valid, y_pred_rf)
print(f"\n混淆矩阵:")
print(f"              预测流失  预测留存")
print(f"实际流失        {cm_rf[0,0]:>6}    {cm_rf[0,1]:>6}")
print(f"实际留存        {cm_rf[1,0]:>6}    {cm_rf[1,1]:>6}")

# 分类报告
print(f"\n详细分类报告:")
print(classification_report(y_valid, y_pred_rf, target_names=['流失', '留存']))

# 特征重要性
feature_importance_rf = pd.DataFrame({
    'feature': X_cols,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\n特征重要性 Top 10:")
print(feature_importance_rf.head(10).to_string(index=False))

# %%
# ========================================
# 模型2: XGBoost
# ========================================
print("\n" + "="*60)
print("模型2: XGBoost")
print("="*60)

# 计算类别权重
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

# 转换为numpy数组
X_train_np = X_train.values
y_train_np = y_train.values
X_valid_np = X_valid.values
y_valid_np = y_valid.values

xgb_clf.fit(X_train_np, y_train_np)
y_pred_xgb = xgb_clf.predict(X_valid_np)
y_pred_proba_xgb = xgb_clf.predict_proba(X_valid_np)[:, 1]
# 保存预测结果到验证集
df_mech_valid['y_pred_xgb'] = y_pred_xgb
# 评估指标
accuracy_xgb = accuracy_score(y_valid_np, y_pred_xgb)
auc_xgb = roc_auc_score(y_valid_np, y_pred_proba_xgb)
precision_xgb, recall_xgb, f1_xgb, _ = precision_recall_fscore_support(y_valid_np, y_pred_xgb, average='binary')

print(f"\n整体准确率: {accuracy_xgb*100:.2f}%")
print(f"AUC-ROC: {auc_xgb:.4f}")
print(f"精确率 (Precision): {precision_xgb:.4f}")
print(f"召回率 (Recall): {recall_xgb:.4f}")
print(f"F1分数: {f1_xgb:.4f}")

# 混淆矩阵
cm_xgb = confusion_matrix(y_valid_np, y_pred_xgb)
print(f"\n混淆矩阵:")
print(f"              预测流失  预测留存")
print(f"实际流失        {cm_xgb[0,0]:>6}    {cm_xgb[0,1]:>6}")
print(f"实际留存        {cm_xgb[1,0]:>6}    {cm_xgb[1,1]:>6}")

# 分类报告
print(f"\n详细分类报告:")
print(classification_report(y_valid_np, y_pred_xgb, target_names=['流失', '留存']))

# 特征重要性
feature_importance_xgb = pd.DataFrame({
    'feature': X_cols,
    'importance': xgb_clf.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\n特征重要性 Top 10:")
print(feature_importance_xgb.head(10).to_string(index=False))

# %%
# ========================================
# 模型3: 逻辑回归 (Logistic Regression)
# ========================================
print("\n" + "="*60)
print("模型3: 逻辑回归 (Logistic Regression)")
print("="*60)

lr_clf = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

lr_clf.fit(X_train, y_train)
y_pred_lr = lr_clf.predict(X_valid)
y_pred_proba_lr = lr_clf.predict_proba(X_valid)[:, 1]

# 保存预测结果到验证集
df_mech_valid['y_pred_lr'] = y_pred_lr

# 评估指标
accuracy_lr = accuracy_score(y_valid, y_pred_lr)
auc_lr = roc_auc_score(y_valid, y_pred_proba_lr)
precision_lr, recall_lr, f1_lr, _ = precision_recall_fscore_support(y_valid, y_pred_lr, average='binary')

print(f"\n整体准确率: {accuracy_lr*100:.2f}%")
print(f"AUC-ROC: {auc_lr:.4f}")
print(f"精确率 (Precision): {precision_lr:.4f}")
print(f"召回率 (Recall): {recall_lr:.4f}")
print(f"F1分数: {f1_lr:.4f}")

# 混淆矩阵
cm_lr = confusion_matrix(y_valid, y_pred_lr)
print(f"\n混淆矩阵:")
print(f"              预测流失  预测留存")
print(f"实际流失        {cm_lr[0,0]:>6}    {cm_lr[0,1]:>6}")
print(f"实际留存        {cm_lr[1,0]:>6}    {cm_lr[1,1]:>6}")

# 分类报告
print(f"\n详细分类报告:")
print(classification_report(y_valid, y_pred_lr, target_names=['流失', '留存']))

# 特征系数（前10个最重要的特征）
feature_coef_lr = pd.DataFrame({
    'feature': X_cols,
    'coefficient': lr_clf.coef_[0]
}).sort_values('coefficient', ascending=False, key=abs)
print(f"\n特征系数 Top 10 (按绝对值排序):")
print(feature_coef_lr.head(10).to_string(index=False))

# %%
# ========================================
# 模型性能对比
# ========================================
print("\n" + "="*60)
print("模型性能对比")
print("="*60)

comparison_data = {
    '模型': ['随机森林', 'XGBoost', '逻辑回归'],
    '准确率': [f"{accuracy_rf*100:.2f}%", f"{accuracy_xgb*100:.2f}%", f"{accuracy_lr*100:.2f}%"],
    'AUC-ROC': [f"{auc_rf:.4f}", f"{auc_xgb:.4f}", f"{auc_lr:.4f}"],
    '精确率': [f"{precision_rf:.4f}", f"{precision_xgb:.4f}", f"{precision_lr:.4f}"],
    '召回率': [f"{recall_rf:.4f}", f"{recall_xgb:.4f}", f"{recall_lr:.4f}"],
    'F1分数': [f"{f1_rf:.4f}", f"{f1_xgb:.4f}", f"{f1_lr:.4f}"]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n")
print(comparison_df.to_string(index=False))

# 找出最佳模型
best_auc_idx = np.argmax([auc_rf, auc_xgb, auc_lr])
best_f1_idx = np.argmax([f1_rf, f1_xgb, f1_lr])
best_models = ['随机森林', 'XGBoost', '逻辑回归']

print(f"\n最佳模型:")
print(f"  AUC-ROC最高: {best_models[best_auc_idx]}")
print(f"  F1分数最高: {best_models[best_f1_idx]}")

# %%
# 隔日留存分析
print("\n" + "="*60)
print("========== 隔日留存分析 ==========")
print("="*60)

# 创建隔日留存变量
# 先对整个df_mech按PlayerID和Date排序
df_mech_sorted = df_mech.sort_values(['PlayerID', 'Date']).reset_index(drop=True)

# 创建次日日期
df_mech_sorted['next_day'] = df_mech_sorted['Date'] + pd.Timedelta(days=1)

# 为每个玩家的每个日期，检查次日是否有记录
next_day_records = df_mech_sorted.groupby('PlayerID')['Date'].apply(set).to_dict()

def has_next_day_play(row):
    player_id = row['PlayerID']
    next_day = row['next_day']
    if player_id in next_day_records:
        return 1 if next_day in next_day_records[player_id] else 0
    return 0

df_mech_sorted['has_next_day'] = df_mech_sorted.apply(has_next_day_play, axis=1)

# 将隔日留存变量合并到验证集
df_mech_valid = df_mech_valid.merge(
    df_mech_sorted[['PlayerID', 'Date', 'has_next_day']], 
    on=['PlayerID', 'Date'], 
    how='left'
)

# 填充缺失值（如果有的话）
df_mech_valid['has_next_day'] = df_mech_valid['has_next_day'].fillna(0).astype(int)

print(f"验证集样本数: {len(df_mech_valid)}")
print(f"隔日有游玩记录的样本数: {df_mech_valid['has_next_day'].sum()} ({df_mech_valid['has_next_day'].mean()*100:.2f}%)")
print(f"隔日无游玩记录的样本数: {(df_mech_valid['has_next_day']==0).sum()} ({(df_mech_valid['has_next_day']==0).mean()*100:.2f}%)")

# %%
# 分析预测流失与隔日留存的关系
print("\n" + "="*60)
print("========== 流失预测与隔日留存的关系分析 ==========")
print("="*60)

# 1. 真实流失(y=0)的样本，隔日留存率
true_0_mask = df_mech_valid['is_active'] == 0
true_0_retention = df_mech_valid[true_0_mask]['has_next_day'].mean()
print(f"\n【真实流失的样本】")
print(f"  样本数: {true_0_mask.sum()}")
print(f"  隔日留存率: {true_0_retention*100:.2f}%")

# 2. 随机森林预测流失的样本，隔日留存率
pred_rf_0_mask = df_mech_valid['y_pred_rf'] == 0
pred_rf_0_retention = df_mech_valid[pred_rf_0_mask]['has_next_day'].mean()
print(f"\n【随机森林预测流失的样本】")
print(f"  样本数: {pred_rf_0_mask.sum()}")
print(f"  隔日留存率: {pred_rf_0_retention*100:.2f}%")
print(f"  预测正确率: {(df_mech_valid[pred_rf_0_mask]['is_active'] == 0).mean()*100:.2f}%")

# 3. XGBoost预测流失的样本，隔日留存率
pred_xgb_0_mask = df_mech_valid['y_pred_xgb'] == 0
pred_xgb_0_retention = df_mech_valid[pred_xgb_0_mask]['has_next_day'].mean()
print(f"\n【XGBoost预测流失的样本】")
print(f"  样本数: {pred_xgb_0_mask.sum()}")
print(f"  隔日留存率: {pred_xgb_0_retention*100:.2f}%")
print(f"  预测正确率: {(df_mech_valid[pred_xgb_0_mask]['is_active'] == 0).mean()*100:.2f}%")

# 4. Logistic预测流失的样本，隔日留存率
pred_lr_0_mask = df_mech_valid['y_pred_lr'] == 0
pred_lr_0_retention = df_mech_valid[pred_lr_0_mask]['has_next_day'].mean()
print(f"\n【Logistic预测流失的样本】")
print(f"  样本数: {pred_lr_0_mask.sum()}")
print(f"  隔日留存率: {pred_lr_0_retention*100:.2f}%")
print(f"  预测正确率: {(df_mech_valid[pred_lr_0_mask]['is_active'] == 0).mean()*100:.2f}%")

# %%
# 共线性分析
print("\n" + "="*60)
print("========== 共线性分析 ==========")
print("="*60)

from scipy.stats import pearsonr, spearmanr, chi2_contingency

# 1. 真实流失 vs 隔日留存
true_y_binary = (df_mech_valid['is_active'] == 0).astype(int)
pearson_corr_true, pearson_p_true = pearsonr(true_y_binary, df_mech_valid['has_next_day'])
spearman_corr_true, spearman_p_true = spearmanr(true_y_binary, df_mech_valid['has_next_day'])

print(f"\n【真实流失 vs 隔日留存】")
print(f"  Pearson相关系数: {pearson_corr_true:.4f} (p-value: {pearson_p_true:.4e})")
print(f"  Spearman相关系数: {spearman_corr_true:.4f} (p-value: {spearman_p_true:.4e})")

contingency_table_true = pd.crosstab(true_y_binary, df_mech_valid['has_next_day'])
chi2_true, p_true, dof_true, expected_true = chi2_contingency(contingency_table_true)
print(f"  卡方检验: χ²={chi2_true:.2f}, p-value={p_true:.4e}")
print(f"\n  列联表:")
print(contingency_table_true)

# 2. 随机森林预测流失 vs 隔日留存
pred_rf_binary = (df_mech_valid['y_pred_rf'] == 0).astype(int)
pearson_corr_rf, pearson_p_rf = pearsonr(pred_rf_binary, df_mech_valid['has_next_day'])
spearman_corr_rf, spearman_p_rf = spearmanr(pred_rf_binary, df_mech_valid['has_next_day'])

print(f"\n【随机森林预测流失 vs 隔日留存】")
print(f"  Pearson相关系数: {pearson_corr_rf:.4f} (p-value: {pearson_p_rf:.4e})")
print(f"  Spearman相关系数: {spearman_corr_rf:.4f} (p-value: {spearman_p_rf:.4e})")

contingency_table_rf = pd.crosstab(pred_rf_binary, df_mech_valid['has_next_day'])
chi2_rf, p_rf, dof_rf, expected_rf = chi2_contingency(contingency_table_rf)
print(f"  卡方检验: χ²={chi2_rf:.2f}, p-value={p_rf:.4e}")
print(f"\n  列联表:")
print(contingency_table_rf)

# 3. XGBoost预测流失 vs 隔日留存
pred_xgb_binary = (df_mech_valid['y_pred_xgb'] == 0).astype(int)
pearson_corr_xgb, pearson_p_xgb = pearsonr(pred_xgb_binary, df_mech_valid['has_next_day'])
spearman_corr_xgb, spearman_p_xgb = spearmanr(pred_xgb_binary, df_mech_valid['has_next_day'])

print(f"\n【XGBoost预测流失 vs 隔日留存】")
print(f"  Pearson相关系数: {pearson_corr_xgb:.4f} (p-value: {pearson_p_xgb:.4e})")
print(f"  Spearman相关系数: {spearman_corr_xgb:.4f} (p-value: {spearman_p_xgb:.4e})")

contingency_table_xgb = pd.crosstab(pred_xgb_binary, df_mech_valid['has_next_day'])
chi2_xgb, p_xgb, dof_xgb, expected_xgb = chi2_contingency(contingency_table_xgb)
print(f"  卡方检验: χ²={chi2_xgb:.2f}, p-value={p_xgb:.4e}")
print(f"\n  列联表:")
print(contingency_table_xgb)

# %%
# 详细交叉分析
print("\n" + "="*60)
print("========== 详细交叉分析 ==========")
print("="*60)

analysis_groups = {
    '真实流失 & 隔日留存': (df_mech_valid['is_active'] == 0) & (df_mech_valid['has_next_day'] == 1),
    '真实流失 & 隔日流失': (df_mech_valid['is_active'] == 0) & (df_mech_valid['has_next_day'] == 0),
    '真实留存 & 隔日留存': (df_mech_valid['is_active'] == 1) & (df_mech_valid['has_next_day'] == 1),
    '真实留存 & 隔日流失': (df_mech_valid['is_active'] == 1) & (df_mech_valid['has_next_day'] == 0)
}

print("\n【各组合样本数】")
for group_name, mask in analysis_groups.items():
    count = mask.sum()
    pct = mask.mean() * 100
    print(f"  {group_name}: {count} ({pct:.2f}%)")

print("\n【随机森林在各组合的预测表现】")
for group_name, mask in analysis_groups.items():
    if mask.sum() > 0:
        group_data = df_mech_valid[mask]
        pred_0_rate = (group_data['y_pred_rf'] == 0).mean() * 100
        pred_correct_rate = (group_data['y_pred_rf'] == group_data['is_active']).mean() * 100
        print(f"  {group_name}:")
        print(f"    预测为流失的比例: {pred_0_rate:.2f}%")
        print(f"    预测正确率: {pred_correct_rate:.2f}%")

print("\n【XGBoost在各组合的预测表现】")
for group_name, mask in analysis_groups.items():
    if mask.sum() > 0:
        group_data = df_mech_valid[mask]
        pred_0_rate = (group_data['y_pred_xgb'] == 0).mean() * 100
        pred_correct_rate = (group_data['y_pred_xgb'] == group_data['is_active']).mean() * 100
        print(f"  {group_name}:")
        print(f"    预测为流失的比例: {pred_0_rate:.2f}%")
        print(f"    预测正确率: {pred_correct_rate:.2f}%")

# %%
# 时间序列分析：每日流失趋势
print("\n" + "="*60)
print("========== 时间序列分析：每日流失趋势 ==========")
print("="*60)

# 合并训练集和验证集进行完整的时间序列分析
df_all_with_pred = pd.concat([
    df_mech_train[['PlayerID', 'Date', 'is_active']],
    df_mech_valid[['PlayerID', 'Date', 'is_active', 'y_pred_rf', 'y_pred_xgb', 'has_next_day']]
], ignore_index=True)

# 填充训练集缺失的预测列
if 'y_pred_rf' not in df_all_with_pred.columns:
    df_all_with_pred['y_pred_rf'] = np.nan
if 'y_pred_xgb' not in df_all_with_pred.columns:
    df_all_with_pred['y_pred_xgb'] = np.nan
if 'has_next_day' not in df_all_with_pred.columns:
    df_all_with_pred['has_next_day'] = np.nan

# 为训练集也添加隔日留存数据
df_mech_train_with_retention = df_mech_train.merge(
    df_mech_sorted[['PlayerID', 'Date', 'has_next_day']], 
    on=['PlayerID', 'Date'], 
    how='left'
)
df_mech_train_with_retention['has_next_day'] = df_mech_train_with_retention['has_next_day'].fillna(0).astype(int)

# 合并完整数据（包含训练集和验证集）
df_full = pd.concat([
    df_mech_train_with_retention[['PlayerID', 'Date', 'is_active', 'has_next_day']],
    df_mech_valid[['PlayerID', 'Date', 'is_active', 'has_next_day']]
], ignore_index=True)

# 只在验证集上才有预测值，所以单独处理验证集的时间序列
daily_stats_valid = df_mech_valid.groupby('Date').agg(
    真实流失率=('is_active', lambda x: (x == 0).mean()),
    RF预测流失率=('y_pred_rf', lambda x: (x == 0).mean()),
    XGBoost预测流失率=('y_pred_xgb', lambda x: (x == 0).mean()),
    LR预测流失率=('y_pred_lr', lambda x: (x == 0).mean()),
    实际隔日流失率=('has_next_day', lambda x: (x == 0).mean()),
    样本数=('PlayerID', 'count')
).reset_index()

# 全量数据的每日统计（用于更完整的真实流失率）
daily_stats_full = df_full.groupby('Date').agg(
    真实流失率_全量=('is_active', lambda x: (x == 0).mean()),
    实际隔日流失率_全量=('has_next_day', lambda x: (x == 0).mean()),
    样本数_全量=('PlayerID', 'count')
).reset_index()

print("\n【验证集每日流失趋势统计】")
print(daily_stats_valid.head(10))

print("\n【每日统计摘要（验证集）】")
print(f"日期范围: {daily_stats_valid['Date'].min()} 到 {daily_stats_valid['Date'].max()}")
print(f"平均真实流失率: {daily_stats_valid['真实流失率'].mean()*100:.2f}%")
print(f"平均RF预测流失率: {daily_stats_valid['RF预测流失率'].mean()*100:.2f}%")
print(f"平均XGBoost预测流失率: {daily_stats_valid['XGBoost预测流失率'].mean()*100:.2f}%")
print(f"平均实际隔日流失率: {daily_stats_valid['实际隔日流失率'].mean()*100:.2f}%")

# %%
# 导出二分类模型关键指标
print("\n" + "="*60)
print("========== 导出二分类模型关键指标 ==========")
print("="*60)

# 准备时间序列数据
timeseries_data = {
    "dates": daily_stats_valid['Date'].dt.strftime('%Y-%m-%d').tolist(),
    "真实流失率": (daily_stats_valid['真实流失率'] * 100).round(2).tolist(),
    "RF预测流失率": (daily_stats_valid['RF预测流失率'] * 100).round(2).tolist(),
    "XGBoost预测流失率": (daily_stats_valid['XGBoost预测流失率'] * 100).round(2).tolist(),
    "LR预测流失率": (daily_stats_valid['LR预测流失率'] * 100).round(2).tolist(),
    "实际隔日流失率": (daily_stats_valid['实际隔日流失率'] * 100).round(2).tolist(),
    "样本数": daily_stats_valid['样本数'].tolist()
}

# 收集所有关键指标
report_metrics = {
    "报告生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "模型类型": "二分类 (流失预测)",
    "数据规模": {
        "原始数据行数": len(df_variableX),
        "过滤后样本数": len(df_mech),
        "唯一玩家数": df_mech['PlayerID'].nunique(),
        "训练集样本数": len(df_mech_train),
        "验证集样本数": len(df_mech_valid),
        "训练集玩家数": df_mech_train['PlayerID'].nunique(),
        "验证集玩家数": df_mech_valid['PlayerID'].nunique()
    },
    "类别分布_训练集": {
        "流失(0)": {
            "样本数": int((y_train == 0).sum()),
            "占比": f"{(y_train == 0).mean()*100:.2f}%"
        },
        "留存(1)": {
            "样本数": int((y_train == 1).sum()),
            "占比": f"{(y_train == 1).mean()*100:.2f}%"
        }
    },
    "类别分布_验证集": {
        "流失(0)": {
            "样本数": int((y_valid == 0).sum()),
            "占比": f"{(y_valid == 0).mean()*100:.2f}%"
        },
        "留存(1)": {
            "样本数": int((y_valid == 1).sum()),
            "占比": f"{(y_valid == 1).mean()*100:.2f}%"
        }
    },
    "模型性能对比": {
        "随机森林": {
            "准确率": f"{accuracy_rf*100:.2f}%",
            "AUC-ROC": f"{auc_rf:.4f}",
            "精确率": f"{precision_rf:.4f}",
            "召回率": f"{recall_rf:.4f}",
            "F1分数": f"{f1_rf:.4f}"
        },
        "XGBoost": {
            "准确率": f"{accuracy_xgb*100:.2f}%",
            "AUC-ROC": f"{auc_xgb:.4f}",
            "精确率": f"{precision_xgb:.4f}",
            "召回率": f"{recall_xgb:.4f}",
            "F1分数": f"{f1_xgb:.4f}"
        },
        "逻辑回归": {
            "准确率": f"{accuracy_lr*100:.2f}%",
            "AUC-ROC": f"{auc_lr:.4f}",
            "精确率": f"{precision_lr:.4f}",
            "召回率": f"{recall_lr:.4f}",
            "F1分数": f"{f1_lr:.4f}"
        }
    },
    "混淆矩阵": {
        "随机森林": cm_rf.tolist(),
        "XGBoost": cm_xgb.tolist(),
        "逻辑回归": cm_lr.tolist()
    },
    "特征重要性_随机森林": feature_importance_rf.head(10).to_dict('records'),
    "特征重要性_XGBoost": feature_importance_xgb.head(10).to_dict('records'),
    "时间序列数据": timeseries_data
}

# 保存为JSON文件
json_path = r"D:\IGame\研究\玩家生命週期\二分类模型_关键指标.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(report_metrics, f, ensure_ascii=False, indent=2)

# 保存时间序列数据为JS文件（用于HTML直接引用）
js_path = r"D:\IGame\研究\玩家生命週期\binary_data.js"
with open(js_path, 'w', encoding='utf-8') as f:
    f.write("// 二分类模型时间序列数据\n")
    f.write("// 此文件由 lifetime_bin.py 自动生成\n\n")
    f.write(f"const binaryTimeSeriesData = {json.dumps(timeseries_data, ensure_ascii=False, indent=2)};\n")
    f.write("\nconsole.log('二分类时间序列数据已加载:', binaryTimeSeriesData.dates.length, '天');\n")

# 保存为可读的文本文件
txt_path = r"D:\IGame\研究\玩家生命週期\二分类模型_关键指标.txt"
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("玩家生命周期预测 - 二分类模型关键指标汇总\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"报告生成时间: {report_metrics['报告生成时间']}\n")
    f.write(f"模型类型: {report_metrics['模型类型']}\n\n")
    
    f.write("【数据规模】\n")
    for key, value in report_metrics['数据规模'].items():
        f.write(f"  {key}: {value:,}\n")
    
    f.write("\n【类别分布 - 训练集】\n")
    for key, value in report_metrics['类别分布_训练集'].items():
        f.write(f"  {key}: {value['样本数']:,}个 ({value['占比']})\n")
    
    f.write("\n【类别分布 - 验证集】\n")
    for key, value in report_metrics['类别分布_验证集'].items():
        f.write(f"  {key}: {value['样本数']:,}个 ({value['占比']})\n")
    
    f.write("\n【模型性能对比】\n")
    for model_name, metrics in report_metrics['模型性能对比'].items():
        f.write(f"\n  {model_name}:\n")
        for metric_name, value in metrics.items():
            f.write(f"    {metric_name}: {value}\n")
    
    f.write("\n【混淆矩阵】\n")
    for model_name, matrix in report_metrics['混淆矩阵'].items():
        f.write(f"\n  {model_name}:\n")
        matrix_arr = np.array(matrix)
        f.write(f"                预测流失  预测留存\n")
        f.write(f"    实际流失      {matrix_arr[0,0]:>6}    {matrix_arr[0,1]:>6}\n")
        f.write(f"    实际留存      {matrix_arr[1,0]:>6}    {matrix_arr[1,1]:>6}\n")
    
    f.write("\n【特征重要性 - 随机森林 Top 10】\n")
    for item in report_metrics['特征重要性_随机森林']:
        f.write(f"  {item['feature']}: {item['importance']:.6f}\n")
    
    f.write("\n【特征重要性 - XGBoost Top 10】\n")
    for item in report_metrics['特征重要性_XGBoost']:
        f.write(f"  {item['feature']}: {item['importance']:.6f}\n")

print(f"\n✓ 关键指标已导出:")
print(f"  JSON格式: {json_path}")
print(f"  文本格式: {txt_path}")
print(f"\n这些文件包含所有二分类模型的评估数据！")
# %%
