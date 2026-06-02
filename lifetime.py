# %%
import sys
import io

# 设置标准输出编码为UTF-8（解决Windows终端中文显示问题）
# 仅在终端环境下设置，Notebook环境跳过
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import duckdb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Gaussian
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, accuracy_score, confusion_matrix
from scipy.stats import pearsonr
import xgboost as xgb
import json
from datetime import datetime

db_path = r"D:\IGame\研究\玩家生命週期\101003_20260108_20260318.duckdb"
con = duckdb.connect(db_path, read_only=True)
df_variableX = con.execute("""
    SELECT *
    FROM main.VariableX
""").fetchdf()
con.close()
df_variable = df_variableX.iloc[:, [0,1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,18,19,20,21,25]].copy()# %%
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
# 选择使用的数据集：df_mech（全部玩家）或 df_mech1（排除最新7天有游玩记录的玩家）
use_filtered = False  # True使用df_mech1，False使用df_mech
df_model = df_mech1 if use_filtered else df_mech

print(f"数据集选择：")
print(f"df_mech样本数: {len(df_mech)}, 玩家数: {df_mech['PlayerID'].nunique()}")
print(f"df_mech1样本数: {len(df_mech1)}, 玩家数: {df_mech1['PlayerID'].nunique()}")
print(f"当前使用: {'df_mech1 (排除最新7天玩家)' if use_filtered else 'df_mech (全部玩家)'}")

# 按玩家ID拆分训练集和验证集
df_model = df_model.copy()  # 避免SettingWithCopyWarning
df_model['PlayerID_int'] = df_model['PlayerID'].astype(int)
train_mask = df_model['PlayerID_int'] % 5 != 4
df_mech_train = df_model[train_mask].copy()
df_mech_valid = df_model[~train_mask].copy()

# 定义预测特征和目标变量
# X特征：iloc[:, 4:22]和iloc[:, 25:42]，加上played_days
# 需要排除会造成数据泄漏的变量：full_spin, Remaining_spin
base_cols = list(df_model.columns[4:22]) + list(df_model.columns[25:42])
# 排除目标变量和相关变量
exclude_features = ['full_spin', 'Remaining_spin', 'PlayerID_int']
X_cols = [col for col in base_cols if col not in exclude_features] + ['played_days']

# 创建离散目标变量（5个区间）
def create_discrete_bins(y):
    """
    将剩余转数转换为5个离散类别
    0: 0~100
    1: 101~500
    2: 501~1000
    3: 1001~5000
    4: 5001~
    """
    bins = []
    for val in y:
        if val <= 100:
            bins.append(0)
        elif val <= 500:
            bins.append(1)
        elif val <= 1000:
            bins.append(2)
        elif val <= 5000:
            bins.append(3)
        else:
            bins.append(4)
    return np.array(bins)

# 识别最新3天有游玩记录的玩家
max_date = df_mech['Date'].max()
recent_start_date = max_date - pd.Timedelta(days=2)
recent_players = df_mech[
    (df_mech['Date'] >= recent_start_date) & 
    (df_mech['Date'] <= max_date)
]['PlayerID'].unique()

y_col = 'Remaining_spin_bin'  # 离散目标变量
# 创建目标变量
df_model[y_col] = create_discrete_bins(df_model['Remaining_spin'])
df_mech_train[y_col] = create_discrete_bins(df_mech_train['Remaining_spin'])
df_mech_valid[y_col] = create_discrete_bins(df_mech_valid['Remaining_spin'])

# 统计最新3天玩家中有多少原本是0
recent_zero_train = ((df_mech_train['PlayerID'].isin(recent_players)) & (df_mech_train[y_col] == 0)).sum()
recent_zero_valid = ((df_mech_valid['PlayerID'].isin(recent_players)) & (df_mech_valid[y_col] == 0)).sum()

# 对最新3天有游玩记录的玩家，如果实际值为0，改为1（因为他们还在活跃，不应该被视为低价值）
df_model.loc[(df_model['PlayerID'].isin(recent_players)) & (df_model[y_col] == 0), y_col] = 1
df_mech_train.loc[(df_mech_train['PlayerID'].isin(recent_players)) & (df_mech_train[y_col] == 0), y_col] = 1
df_mech_valid.loc[(df_mech_valid['PlayerID'].isin(recent_players)) & (df_mech_valid[y_col] == 0), y_col] = 1

print(f"\n最新3天有游玩记录的玩家数: {len(recent_players)}")
print(f"训练集中最新3天玩家样本数: {df_mech_train['PlayerID'].isin(recent_players).sum()}, 其中{recent_zero_train}个从0改为1")
print(f"验证集中最新3天玩家样本数: {df_mech_valid['PlayerID'].isin(recent_players).sum()}, 其中{recent_zero_valid}个从0改为1")

print(f"\n目标变量: {y_col} (离散分类)")
print(f"特征数量: {len(X_cols)}")
print(f"排除的特征: {[col for col in base_cols if col in exclude_features]}")
print(f"\n目标变量分布（训练集）：")
for i, label in enumerate(['0~100', '101~500', '501~1000', '1001~5000', '5001~']):
    count = (df_mech_train[y_col] == i).sum()
    pct = (df_mech_train[y_col] == i).mean() * 100
    print(f"  类别{i} ({label}): {count} ({pct:.1f}%)")

print(f"目标变量: {y_col}")
print(f"特征数量: {len(X_cols)}")
print(f"排除的特征: {[col for col in base_cols if col in exclude_features]}")

# 自定义分类准确率计算函数（针对离散预测）
def calculate_discrete_accuracy(df_valid, y_true_col, y_pred_col, df_all=None):
    """
    计算离散分类的准确率
    df_valid: 验证集数据框
    y_true_col: 实际类别列名
    y_pred_col: 预测类别列名
    df_all: 原始完整数据框，用于获取真实的最新日期
    
    规则：
    - 最新3天有游玩记录的玩家：如果原本为0已改为1，只要预测值>0就算正确
    - 其他玩家：预测类别必须完全匹配才算正确
    """
    df = df_valid.copy()
    
    # 使用原始数据获取真实的最新日期
    if df_all is not None:
        max_date = df_all['Date'].max()
        recent_start_date = max_date - pd.Timedelta(days=2)  # 最新3天
        recent_players = df_all[
            (df_all['Date'] >= recent_start_date) & 
            (df_all['Date'] <= max_date)
        ]['PlayerID'].unique()
    else:
        max_date = df['Date'].max()
        recent_start_date = max_date - pd.Timedelta(days=2)
        recent_players = df[
            (df['Date'] >= recent_start_date) & 
            (df['Date'] <= max_date)
        ]['PlayerID'].unique()
    
    # 标记两种类型的玩家
    df['has_recent_play'] = df['PlayerID'].isin(recent_players)
    
    # 计算准确率
    correct = 0
    total = len(df)
    recent_correct = 0
    recent_total = 0
    other_correct = 0
    other_total = 0
    
    for idx, row in df.iterrows():
        y_true = row[y_true_col]
        y_pred = row[y_pred_col]
        
        if row['has_recent_play']:
            # 最新3天有游玩记录：只要预测>0就算对（因为还在活跃）
            recent_total += 1
            if y_pred > 0:  # 只要预测还会玩（任何非0类别）
                correct += 1
                recent_correct += 1
            elif y_true == y_pred:  # 如果实际和预测都是其他值也算对
                correct += 1
                recent_correct += 1
        else:
            # 其他玩家：必须完全匹配
            other_total += 1
            if y_true == y_pred:
                correct += 1
                other_correct += 1
    
    accuracy = correct / total if total > 0 else 0
    recent_accuracy = recent_correct / recent_total if recent_total > 0 else 0
    other_accuracy = other_correct / other_total if other_total > 0 else 0
    
    return accuracy, recent_accuracy, other_accuracy, recent_total, other_total

# %%
# 使用随机森林分类模型进行预测
print("\n========== 随机森林分类模型 ==========")

# 准备训练集和验证集数据
X_train = df_mech_train[X_cols].copy()
y_train = df_mech_train[y_col].copy()
X_valid = df_mech_valid[X_cols].copy()
y_valid = df_mech_valid[y_col].copy()

# 处理缺失值和无穷值
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_train = X_train.fillna(X_train.median())
X_valid = X_valid.replace([np.inf, -np.inf], np.nan)
X_valid = X_valid.fillna(X_train.median())

# 训练随机森林分类模型
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
# 使用numpy数组训练
X_train_np = X_train.values
X_valid_np = X_valid.values
y_train_np = y_train.values

rf_clf.fit(X_train_np, y_train_np)

# 预测
y_pred = rf_clf.predict(X_valid_np)
df_mech_valid['y_pred_discrete'] = y_pred

# 计算准确率
accuracy, recent_acc, other_acc, recent_total, other_total = calculate_discrete_accuracy(
    df_mech_valid, y_col, 'y_pred_discrete', df_mech
)

print(f"\n整体准确率: {accuracy*100:.2f}%")
print(f"最新3天有游玩记录的玩家 (n={recent_total}): {recent_acc*100:.2f}%")
print(f"其他玩家 (n={other_total}): {other_acc*100:.2f}%")

# 详细分类报告
print(f"\n分类报告:")
print(classification_report(y_valid, y_pred, 
                          target_names=['0~100', '101~500', '501~1000', '1001~5000', '5001~'],
                          zero_division=0))

# 按类别分析预测正确程度
print(f"\n========== 各类别预测正确程度分析 ==========")
class_labels = ['0~100', '101~500', '501~1000', '1001~5000', '5001~']
for class_id in range(5):
    # 找出实际值为该类别的样本
    actual_mask = y_valid == class_id
    actual_count = actual_mask.sum()
    
    if actual_count > 0:
        # 这些样本中预测正确的数量
        correct_count = ((y_valid == class_id) & (y_pred == class_id)).sum()
        accuracy_rate = correct_count / actual_count * 100
        
        # 这些样本被预测为各类别的分布
        pred_dist = pd.Series(y_pred[actual_mask]).value_counts().sort_index()
        
        print(f"\n类别 {class_id} ({class_labels[class_id]}):")
        print(f"  实际样本数: {actual_count}")
        print(f"  预测正确数: {correct_count} ({accuracy_rate:.1f}%)")
        print(f"  预测分布:")
        for pred_class, count in pred_dist.items():
            if pred_class < len(class_labels):
                print(f"    预测为{pred_class}({class_labels[pred_class]}): {count} ({count/actual_count*100:.1f}%)")
    else:
        print(f"\n类别 {class_id} ({class_labels[class_id]}): 无样本")

# 二分类分析：预测0 vs 其他
print(f"\n========== 二分类分析（0 vs 其他）==========")
y_valid_binary = (y_valid > 0).astype(int)  # 0->0, 其他->1
y_pred_binary = (y_pred > 0).astype(int)

binary_accuracy = accuracy_score(y_valid_binary, y_pred_binary)
print(f"二分类准确率: {binary_accuracy*100:.2f}%")

print(f"\n二分类混淆矩阵:")
print(f"                预测=0    预测>0")
binary_cm = confusion_matrix(y_valid_binary, y_pred_binary)
print(f"实际=0      {binary_cm[0,0]:6d}    {binary_cm[0,1]:6d}")
print(f"实际>0      {binary_cm[1,0]:6d}    {binary_cm[1,1]:6d}")

# 计算各项指标
tn, fp, fn, tp = binary_cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n二分类指标:")
print(f"精确率(Precision): {precision*100:.2f}% - 预测为>0的样本中，实际为>0的比例")
print(f"召回率(Recall):    {recall*100:.2f}% - 实际为>0的样本中，被正确预测的比例")
print(f"F1分数:           {f1*100:.2f}%")
print(f"类别0准确率:       {tn/(tn+fp)*100:.2f}% - 实际为0的样本中，被正确预测的比例")
print(f"类别>0准确率:      {tp/(tp+fn)*100:.2f}% - 实际为>0的样本中，被正确预测的比例")

# 考虑最新3天玩家规则的二分类准确率
print(f"\n========== 考虑最新3天玩家规则的二分类准确率 ==========")
max_date = df_mech['Date'].max()
recent_start_date = max_date - pd.Timedelta(days=2)
recent_players = df_mech[
    (df_mech['Date'] >= recent_start_date) & 
    (df_mech['Date'] <= max_date)
]['PlayerID'].unique()

df_mech_valid['is_recent'] = df_mech_valid['PlayerID'].isin(recent_players)
binary_correct = 0
recent_binary_correct = 0
other_binary_correct = 0
recent_binary_total = 0
other_binary_total = 0

for idx, row in df_mech_valid.iterrows():
    y_true_val = row[y_col]
    y_pred_val = row['y_pred_discrete']
    is_recent = row['is_recent']
    
    y_true_bin = 1 if y_true_val > 0 else 0
    y_pred_bin = 1 if y_pred_val > 0 else 0
    
    if is_recent:
        recent_binary_total += 1
        if y_pred_bin > 0:  # 最新3天玩家只要预测>0就算对
            binary_correct += 1
            recent_binary_correct += 1
    else:
        other_binary_total += 1
        if y_true_bin == y_pred_bin:
            binary_correct += 1
            other_binary_correct += 1

total_samples = len(df_mech_valid)
binary_acc_with_rule = binary_correct / total_samples if total_samples > 0 else 0
recent_binary_acc = recent_binary_correct / recent_binary_total if recent_binary_total > 0 else 0
other_binary_acc = other_binary_correct / other_binary_total if other_binary_total > 0 else 0

print(f"整体二分类准确率（考虑规则）: {binary_acc_with_rule*100:.2f}%")
print(f"最新3天玩家 (n={recent_binary_total}): {recent_binary_acc*100:.2f}%")
print(f"其他玩家 (n={other_binary_total}): {other_binary_acc*100:.2f}%")

# 特征重要性
feature_importance = pd.DataFrame({
    'feature': X_cols,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\n前10个重要特征：")
print(feature_importance.head(10))

# 混淆矩阵
cm = confusion_matrix(y_valid, y_pred)
print(f"\n混淆矩阵（随机森林）：")
print(cm)

# %%
# XGBoost分类模型
print("\n" + "="*60)
print("========== XGBoost分类模型 ==========")
print("="*60)

# 转换为numpy数组（解决XGBoost兼容性问题）
X_train_np = X_train.values
X_valid_np = X_valid.values
y_train_np = y_train.values

# 训练XGBoost模型
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)
xgb_clf.fit(X_train_np, y_train_np)

# 预测
y_pred_xgb = xgb_clf.predict(X_valid_np)
df_mech_valid['y_pred_xgb'] = y_pred_xgb

# 计算准确率
accuracy_xgb, recent_acc_xgb, other_acc_xgb, _, _ = calculate_discrete_accuracy(
    df_mech_valid, y_col, 'y_pred_xgb', df_mech
)

print(f"\n整体准确率: {accuracy_xgb*100:.2f}%")
print(f"最新3天有游玩记录的玩家: {recent_acc_xgb*100:.2f}%")
print(f"其他玩家: {other_acc_xgb*100:.2f}%")

# 详细分类报告
print(f"\n分类报告:")
print(classification_report(y_valid, y_pred_xgb, 
                          target_names=['0~100', '101~500', '501~1000', '1001~5000', '5001~'],
                          zero_division=0))

# 按类别分析预测正确程度
print(f"\n========== 各类别预测正确程度分析（XGBoost）==========")
class_labels = ['0~100', '101~500', '501~1000', '1001~5000', '5001~']
for class_id in range(5):
    # 找出实际值为该类别的样本
    actual_mask = y_valid == class_id
    actual_count = actual_mask.sum()
    
    if actual_count > 0:
        # 这些样本中预测正确的数量
        correct_count = ((y_valid == class_id) & (y_pred_xgb == class_id)).sum()
        accuracy_rate = correct_count / actual_count * 100
        
        # 这些样本被预测为各类别的分布
        pred_dist = pd.Series(y_pred_xgb[actual_mask]).value_counts().sort_index()
        
        print(f"\n类别 {class_id} ({class_labels[class_id]}):")
        print(f"  实际样本数: {actual_count}")
        print(f"  预测正确数: {correct_count} ({accuracy_rate:.1f}%)")
        print(f"  预测分布:")
        for pred_class, count in pred_dist.items():
            if pred_class < len(class_labels):
                print(f"    预测为{pred_class}({class_labels[pred_class]}): {count} ({count/actual_count*100:.1f}%)")
    else:
        print(f"\n类别 {class_id} ({class_labels[class_id]}): 无样本")

# 二分类分析
y_pred_xgb_binary = (y_pred_xgb > 0).astype(int)
binary_accuracy_xgb = accuracy_score(y_valid_binary, y_pred_xgb_binary)
print(f"\n二分类准确率: {binary_accuracy_xgb*100:.2f}%")

# 混淆矩阵
cm_xgb = confusion_matrix(y_valid, y_pred_xgb)
print(f"\n混淆矩阵（XGBoost）：")
print(cm_xgb)

# %%
# Ordinal Logistic回归
print("\n" + "="*60)
print("========== Ordinal Logistic回归 ==========")
print("="*60)

try:
    from mord import LogisticAT
    
    print("开始训练Ordinal Logistic模型（此过程可能需要几分钟）...")
    
    # 训练Ordinal Logistic模型（使用numpy数组）
    ord_clf = LogisticAT(alpha=1.0)
    ord_clf.fit(X_train_np, y_train_np)
    
    # 预测
    y_pred_ord = ord_clf.predict(X_valid_np)
    df_mech_valid['y_pred_ord'] = y_pred_ord
    
    # 计算准确率
    accuracy_ord, recent_acc_ord, other_acc_ord, _, _ = calculate_discrete_accuracy(
        df_mech_valid, y_col, 'y_pred_ord', df_mech
    )
    
    print(f"\n整体准确率: {accuracy_ord*100:.2f}%")
    print(f"最新3天有游玩记录的玩家: {recent_acc_ord*100:.2f}%")
    print(f"其他玩家: {other_acc_ord*100:.2f}%")
    
    # 详细分类报告
    print(f"\n分类报告:")
    print(classification_report(y_valid, y_pred_ord, 
                              target_names=['0~100', '101~500', '501~1000', '1001~5000', '5001~'],
                              zero_division=0))
    
    # 按类别分析预测正确程度
    print(f"\n========== 各类别预测正确程度分析（Ordinal Logistic）==========")
    class_labels = ['0~100', '101~500', '501~1000', '1001~5000', '5001~']
    for class_id in range(5):
        # 找出实际值为该类别的样本
        actual_mask = y_valid == class_id
        actual_count = actual_mask.sum()
        
        if actual_count > 0:
            # 这些样本中预测正确的数量
            correct_count = ((y_valid == class_id) & (y_pred_ord == class_id)).sum()
            accuracy_rate = correct_count / actual_count * 100
            
            # 这些样本被预测为各类别的分布
            pred_dist = pd.Series(y_pred_ord[actual_mask]).value_counts().sort_index()
            
            print(f"\n类别 {class_id} ({class_labels[class_id]}):")
            print(f"  实际样本数: {actual_count}")
            print(f"  预测正确数: {correct_count} ({accuracy_rate:.1f}%)")
            print(f"  预测分布:")
            for pred_class, count in pred_dist.items():
                if pred_class < len(class_labels):
                    print(f"    预测为{pred_class}({class_labels[pred_class]}): {count} ({count/actual_count*100:.1f}%)")
        else:
            print(f"\n类别 {class_id} ({class_labels[class_id]}): 无样本")
    
    # 二分类分析
    y_pred_ord_binary = (y_pred_ord > 0).astype(int)
    binary_accuracy_ord = accuracy_score(y_valid_binary, y_pred_ord_binary)
    print(f"\n二分类准确率: {binary_accuracy_ord*100:.2f}%")
    
    # 混淆矩阵
    cm_ord = confusion_matrix(y_valid, y_pred_ord)
    print(f"\n混淆矩阵（Ordinal Logistic）：")
    print(cm_ord)
    
    has_ordinal = True
    
except ImportError:
    print("\n注意：需要安装mord库才能使用Ordinal Logistic回归")
    print("可以运行: pip install mord")
    has_ordinal = False
    accuracy_ord = 0
    recent_acc_ord = 0
    other_acc_ord = 0
    binary_accuracy_ord = 0

# %%
# 模型对比总结
print("\n" + "="*60)
print("========== 模型对比总结 ==========")
print("="*60)

comparison_data = {
    '模型': ['随机森林', 'XGBoost', 'Ordinal Logistic'],
    '整体准确率': [f"{accuracy*100:.2f}%", f"{accuracy_xgb*100:.2f}%", f"{accuracy_ord*100:.2f}%" if has_ordinal else "N/A"],
    '最新3天玩家': [f"{recent_acc*100:.2f}%", f"{recent_acc_xgb*100:.2f}%", f"{recent_acc_ord*100:.2f}%" if has_ordinal else "N/A"],
    '其他玩家': [f"{other_acc*100:.2f}%", f"{other_acc_xgb*100:.2f}%", f"{other_acc_ord*100:.2f}%" if has_ordinal else "N/A"],
    '二分类准确率': [f"{binary_accuracy*100:.2f}%", f"{binary_accuracy_xgb*100:.2f}%", f"{binary_accuracy_ord*100:.2f}%" if has_ordinal else "N/A"]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n")
print(comparison_df.to_string(index=False))

print(f"\n最佳模型:")
if has_ordinal:
    best_model_idx = np.argmax([accuracy, accuracy_xgb, accuracy_ord])
    best_models = ['随机森林', 'XGBoost', 'Ordinal Logistic']
else:
    best_model_idx = np.argmax([accuracy, accuracy_xgb])
    best_models = ['随机森林', 'XGBoost']
print(f"  整体准确率最高: {best_models[best_model_idx]}")

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
# 方法：self join，找出同一玩家在次日是否有记录
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
# 分析真实y=0和预测y=0与隔日留存的关系
print("\n" + "="*60)
print("========== 类别0预测与隔日留存的关系分析 ==========")
print("="*60)

# 1. 真实y=0的样本，隔日留存率
true_0_mask = df_mech_valid[y_col] == 0
true_0_retention = df_mech_valid[true_0_mask]['has_next_day'].mean()
print(f"\n【真实类别0的样本】")
print(f"  样本数: {true_0_mask.sum()}")
print(f"  隔日留存率: {true_0_retention*100:.2f}%")

# 2. 随机森林预测为0的样本，隔日留存率
pred_rf_0_mask = df_mech_valid['y_pred_discrete'] == 0
pred_rf_0_retention = df_mech_valid[pred_rf_0_mask]['has_next_day'].mean()
print(f"\n【随机森林预测为0的样本】")
print(f"  样本数: {pred_rf_0_mask.sum()}")
print(f"  隔日留存率: {pred_rf_0_retention*100:.2f}%")
print(f"  预测正确率: {(df_mech_valid[pred_rf_0_mask][y_col] == 0).mean()*100:.2f}%")

# 3. XGBoost预测为0的样本，隔日留存率
pred_xgb_0_mask = df_mech_valid['y_pred_xgb'] == 0
pred_xgb_0_retention = df_mech_valid[pred_xgb_0_mask]['has_next_day'].mean()
print(f"\n【XGBoost预测为0的样本】")
print(f"  样本数: {pred_xgb_0_mask.sum()}")
print(f"  隔日留存率: {pred_xgb_0_retention*100:.2f}%")
print(f"  预测正确率: {(df_mech_valid[pred_xgb_0_mask][y_col] == 0).mean()*100:.2f}%")

# 4. 如果有Ordinal Logistic结果
if has_ordinal:
    pred_ord_0_mask = df_mech_valid['y_pred_ord'] == 0
    pred_ord_0_retention = df_mech_valid[pred_ord_0_mask]['has_next_day'].mean()
    print(f"\n【Ordinal Logistic预测为0的样本】")
    print(f"  样本数: {pred_ord_0_mask.sum()}")
    print(f"  隔日留存率: {pred_ord_0_retention*100:.2f}%")
    print(f"  预测正确率: {(df_mech_valid[pred_ord_0_mask][y_col] == 0).mean()*100:.2f}%")

# %%
# 共线性分析
print("\n" + "="*60)
print("========== 共线性分析 ==========")
print("="*60)

# 计算相关系数
from scipy.stats import pearsonr, spearmanr, chi2_contingency

# 1. 真实y=0 与 隔日留存的关系
true_y_binary = (df_mech_valid[y_col] == 0).astype(int)
pearson_corr_true, pearson_p_true = pearsonr(true_y_binary, df_mech_valid['has_next_day'])
spearman_corr_true, spearman_p_true = spearmanr(true_y_binary, df_mech_valid['has_next_day'])

print(f"\n【真实y=0 vs 隔日留存】")
print(f"  Pearson相关系数: {pearson_corr_true:.4f} (p-value: {pearson_p_true:.4e})")
print(f"  Spearman相关系数: {spearman_corr_true:.4f} (p-value: {spearman_p_true:.4e})")

# 卡方检验
contingency_table_true = pd.crosstab(true_y_binary, df_mech_valid['has_next_day'])
chi2_true, p_true, dof_true, expected_true = chi2_contingency(contingency_table_true)
print(f"  卡方检验: χ²={chi2_true:.2f}, p-value={p_true:.4e}")
print(f"\n  列联表:")
print(contingency_table_true)

# 2. 随机森林预测y=0 与 隔日留存的关系
pred_rf_binary = (df_mech_valid['y_pred_discrete'] == 0).astype(int)
pearson_corr_rf, pearson_p_rf = pearsonr(pred_rf_binary, df_mech_valid['has_next_day'])
spearman_corr_rf, spearman_p_rf = spearmanr(pred_rf_binary, df_mech_valid['has_next_day'])

print(f"\n【随机森林预测y=0 vs 隔日留存】")
print(f"  Pearson相关系数: {pearson_corr_rf:.4f} (p-value: {pearson_p_rf:.4e})")
print(f"  Spearman相关系数: {spearman_corr_rf:.4f} (p-value: {spearman_p_rf:.4e})")

contingency_table_rf = pd.crosstab(pred_rf_binary, df_mech_valid['has_next_day'])
chi2_rf, p_rf, dof_rf, expected_rf = chi2_contingency(contingency_table_rf)
print(f"  卡方检验: χ²={chi2_rf:.2f}, p-value={p_rf:.4e}")
print(f"\n  列联表:")
print(contingency_table_rf)

# 3. XGBoost预测y=0 与 隔日留存的关系
pred_xgb_binary = (df_mech_valid['y_pred_xgb'] == 0).astype(int)
pearson_corr_xgb, pearson_p_xgb = pearsonr(pred_xgb_binary, df_mech_valid['has_next_day'])
spearman_corr_xgb, spearman_p_xgb = spearmanr(pred_xgb_binary, df_mech_valid['has_next_day'])

print(f"\n【XGBoost预测y=0 vs 隔日留存】")
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

# 创建四种组合的分析
analysis_groups = {
    '真实=0 & 隔日留存': (df_mech_valid[y_col] == 0) & (df_mech_valid['has_next_day'] == 1),
    '真实=0 & 隔日流失': (df_mech_valid[y_col] == 0) & (df_mech_valid['has_next_day'] == 0),
    '真实>0 & 隔日留存': (df_mech_valid[y_col] > 0) & (df_mech_valid['has_next_day'] == 1),
    '真实>0 & 隔日流失': (df_mech_valid[y_col] > 0) & (df_mech_valid['has_next_day'] == 0)
}

print("\n【各组合样本数】")
for group_name, mask in analysis_groups.items():
    count = mask.sum()
    pct = mask.mean() * 100
    print(f"  {group_name}: {count} ({pct:.2f}%)")

# 对每个组合，看随机森林和XGBoost的预测表现
print("\n【随机森林在各组合的预测表现】")
for group_name, mask in analysis_groups.items():
    if mask.sum() > 0:
        group_data = df_mech_valid[mask]
        pred_0_rate = (group_data['y_pred_discrete'] == 0).mean() * 100
        pred_correct_rate = (group_data['y_pred_discrete'] == group_data[y_col]).mean() * 100
        print(f"  {group_name}:")
        print(f"    预测为0的比例: {pred_0_rate:.2f}%")
        print(f"    预测正确率: {pred_correct_rate:.2f}%")

print("\n【XGBoost在各组合的预测表现】")
for group_name, mask in analysis_groups.items():
    if mask.sum() > 0:
        group_data = df_mech_valid[mask]
        pred_0_rate = (group_data['y_pred_xgb'] == 0).mean() * 100
        pred_correct_rate = (group_data['y_pred_xgb'] == group_data[y_col]).mean() * 100
        print(f"  {group_name}:")
        print(f"    预测为0的比例: {pred_0_rate:.2f}%")
        print(f"    预测正确率: {pred_correct_rate:.2f}%")

# %%
# 时间序列分析：每日类别0比例趋势
print("\n" + "="*60)
print("========== 时间序列分析：每日类别0趋势 ==========")
print("="*60)

# 为训练集也添加隔日留存数据
df_mech_train_with_retention = df_mech_train.merge(
    df_mech_sorted[['PlayerID', 'Date', 'has_next_day']], 
    on=['PlayerID', 'Date'], 
    how='left'
)
df_mech_train_with_retention['has_next_day'] = df_mech_train_with_retention['has_next_day'].fillna(0).astype(int)

# 合并完整数据（包含训练集和验证集）
df_full = pd.concat([
    df_mech_train_with_retention[[y_col, 'Date', 'has_next_day']],
    df_mech_valid[[y_col, 'Date', 'has_next_day']]
], ignore_index=True)

# 验证集的每日统计（带预测值）
daily_stats_valid = df_mech_valid.groupby('Date').agg(
    真实类别0率=(y_col, lambda x: (x == 0).mean()),
    RF预测类别0率=('y_pred_discrete', lambda x: (x == 0).mean()),
    XGBoost预测类别0率=('y_pred_xgb', lambda x: (x == 0).mean()),
    实际隔日流失率=('has_next_day', lambda x: (x == 0).mean()),
    样本数=('PlayerID', 'count')
).reset_index()

# 如果有Ordinal Logistic结果
if has_ordinal:
    daily_stats_valid['Ordinal预测类别0率'] = df_mech_valid.groupby('Date')['y_pred_ord'].apply(lambda x: (x == 0).mean()).values

# 全量数据的每日统计
daily_stats_full = df_full.groupby('Date').agg(
    真实类别0率_全量=(y_col, lambda x: (x == 0).mean()),
    实际隔日流失率_全量=('has_next_day', lambda x: (x == 0).mean()),
    样本数_全量=('Date', 'count')
).reset_index()

print("\n【验证集每日类别0趋势统计】")
print(daily_stats_valid.head(10))

print("\n【每日统计摘要（验证集）】")
print(f"日期范围: {daily_stats_valid['Date'].min()} 到 {daily_stats_valid['Date'].max()}")
print(f"平均真实类别0率: {daily_stats_valid['真实类别0率'].mean()*100:.2f}%")
print(f"平均RF预测类别0率: {daily_stats_valid['RF预测类别0率'].mean()*100:.2f}%")
print(f"平均XGBoost预测类别0率: {daily_stats_valid['XGBoost预测类别0率'].mean()*100:.2f}%")
print(f"平均实际隔日流失率: {daily_stats_valid['实际隔日流失率'].mean()*100:.2f}%")

# %%
# 导出研究报告关键指标
print("\n" + "="*60)
print("========== 导出研究报告关键指标 ==========")
print("="*60)

import json
from datetime import datetime

# 准备时间序列数据
timeseries_data = {
    "dates": daily_stats_valid['Date'].dt.strftime('%Y-%m-%d').tolist(),
    "真实类别0率": (daily_stats_valid['真实类别0率'] * 100).round(2).tolist(),
    "RF预测类别0率": (daily_stats_valid['RF预测类别0率'] * 100).round(2).tolist(),
    "XGBoost预测类别0率": (daily_stats_valid['XGBoost预测类别0率'] * 100).round(2).tolist(),
    "实际隔日流失率": (daily_stats_valid['实际隔日流失率'] * 100).round(2).tolist(),
    "样本数": daily_stats_valid['样本数'].tolist()
}

if has_ordinal:
    timeseries_data['Ordinal预测类别0率'] = (daily_stats_valid['Ordinal预测类别0率'] * 100).round(2).tolist()

# 收集所有关键指标
report_metrics = {
    "报告生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        f"类别{i}({['0-100', '101-500', '501-1000', '1001-5000', '5001+'][i]})": {
            "样本数": int((y_train == i).sum()),
            "占比": f"{(y_train == i).mean()*100:.2f}%"
        } for i in range(5)
    },
    "类别分布_验证集": {
        f"类别{i}({['0-100', '101-500', '501-1000', '1001-5000', '5001+'][i]})": {
            "样本数": int((y_valid == i).sum()),
            "占比": f"{(y_valid == i).mean()*100:.2f}%"
        } for i in range(5)
    },
    "模型性能对比": {
        "随机森林": {
            "整体准确率": f"{accuracy*100:.2f}%",
            "最新3天玩家": f"{recent_acc*100:.2f}%",
            "其他玩家": f"{other_acc*100:.2f}%",
            "二分类准确率": f"{binary_accuracy*100:.2f}%"
        },
        "XGBoost": {
            "整体准确率": f"{accuracy_xgb*100:.2f}%",
            "最新3天玩家": f"{recent_acc_xgb*100:.2f}%",
            "其他玩家": f"{other_acc_xgb*100:.2f}%",
            "二分类准确率": f"{binary_accuracy_xgb*100:.2f}%"
        }
    },
    "混淆矩阵": {
        "随机森林": cm.tolist(),
        "XGBoost": cm_xgb.tolist()
    },
    "最新3天玩家统计": {
        "玩家数": len(recent_players),
        "训练集样本数": int(df_mech_train['PlayerID'].isin(recent_players).sum()),
        "验证集样本数": int(df_mech_valid['PlayerID'].isin(recent_players).sum())
    },
    "时间序列数据": timeseries_data
}

# 如果有Ordinal Logistic结果
if has_ordinal:
    report_metrics["模型性能对比"]["Ordinal_Logistic"] = {
        "整体准确率": f"{accuracy_ord*100:.2f}%",
        "最新3天玩家": f"{recent_acc_ord*100:.2f}%",
        "其他玩家": f"{other_acc_ord*100:.2f}%",
        "二分类准确率": f"{binary_accuracy_ord*100:.2f}%"
    }
    report_metrics["混淆矩阵"]["Ordinal_Logistic"] = cm_ord.tolist()

# 保存为JSON文件
json_path = r"D:\IGame\研究\玩家生命週期\研究报告_关键指标.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(report_metrics, f, ensure_ascii=False, indent=2)

# 保存时间序列数据为JS文件（用于HTML直接引用）
js_path = r"D:\IGame\研究\玩家生命週期\multiclass_data.js"
with open(js_path, 'w', encoding='utf-8') as f:
    f.write("// 五分类模型时间序列数据\n")
    f.write("// 此文件由 lifetime.py 自动生成\n\n")
    f.write(f"const multiclassTimeSeriesData = {json.dumps(timeseries_data, ensure_ascii=False, indent=2)};\n")
    f.write("\nconsole.log('五分类时间序列数据已加载:', multiclassTimeSeriesData.dates.length, '天');\n")

# 保存为可读的文本文件
txt_path = r"D:\IGame\研究\玩家生命週期\研究报告_关键指标.txt"
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("玩家生命周期预测研究 - 关键指标汇总\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"报告生成时间: {report_metrics['报告生成时间']}\n\n")
    
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
    
    f.write("\n【最新3天玩家统计】\n")
    for key, value in report_metrics['最新3天玩家统计'].items():
        f.write(f"  {key}: {value:,}\n")
    
    f.write("\n【混淆矩阵】\n")
    for model_name, matrix in report_metrics['混淆矩阵'].items():
        f.write(f"\n  {model_name}:\n")
        matrix_arr = np.array(matrix)
        for row in matrix_arr:
            f.write(f"    {row}\n")

print(f"\n✓ 关键指标已导出:")
print(f"  JSON格式: {json_path}")
print(f"  文本格式: {txt_path}")
print(f"\n这些文件包含所有需要填入报告的数据！")
# %%