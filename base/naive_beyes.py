# coding:UTF-8
# RuHe  2025/4/3 21:06

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('./data/data.csv', sep=',')
# 使用切片删除第一列
df = df.iloc[:, 1:]
df['好瓜'] = df['好瓜'].map({'是': 1, '否': 0})  # 将标签转换为0和1
# print(df.head())
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)


# print("df_test:\n", df_test)


def predict(data, sample):
    probability_1 = 1
    probability_0 = 1  # 初始化判断为好瓜坏瓜的概率
    # 计算每个特征的条件概率
    for feature in sample.columns[:-1]:
        if sample[feature].dtype == 'object':  # 使用拉普拉斯修正避免0概率乘因子,分母可约去
            probability_1 *= (len(data[(data[feature] == sample[feature][0]) & (data['好瓜'] == 1)]) + 1) / (
                    len(data[data['好瓜'] == 1]) + len(data[feature].unique()))
            probability_0 *= (len(data[(data[feature] == sample[feature][0]) & (data['好瓜'] == 0)]) + 1) / (
                    len(data[data['好瓜'] == 0]) + len(data[feature].unique()))
            # print(feature, ": probability_1: ", probability_1)
            # print(feature, ": probability_0: ", probability_0)
        else:  # 连续值的处理
            # 计算均值
            mean1 = data[data['好瓜'] == 1][feature].mean()
            mean0 = data[data['好瓜'] == 0][feature].mean()
            # print("均值:", mean)
            # 计算方差
            variance1 = data[data['好瓜'] == 1][feature].var()
            variance0 = data[data['好瓜'] == 0][feature].var()
            # print("方差:", variance)
            probability_1 *= 1 / np.sqrt(2 * np.pi * variance1) * np.exp(
                -(sample[feature][0] - mean1) ** 2 / (2 * variance1))
            probability_0 *= 1 / np.sqrt(2 * np.pi * variance0) * np.exp(
                -(sample[feature][0] - mean0) ** 2 / (2 * variance0))
            # print(feature, ": probability_1: ", probability_1)
            # print(feature, ": probability_0: ", probability_0)
    return probability_1 > probability_0  # 判断为好瓜还是坏瓜


accuracy = 0
right = 0
error = 0
for i in range(len(df_test)):
    sample = df_test.iloc[i:i + 1]  # 取出第i个样本
    # 重置索引
    sample = sample.reset_index(drop=True)  # drop删除原索引
    # print("第{}个样本: \n".format(i+1), sample)
    prediction = predict(df_train, sample)
    if prediction ^ sample["好瓜"][0]:
        error += 1
    else:
        right += 1
    # print(f'第{i + 1}个样本的预测结果: {"好瓜" if prediction else "坏瓜"}')
    # print(f'第{i + 1}个样本的真实结果: {"好瓜" if sample["好瓜"][0] else "坏瓜"}')
    # print(f'第{i + 1}个样本的预测结果是否正确: {"正确" if prediction == sample["好瓜"][0] else "错误"}')

accuracy = right / (right + error)
print("准确率: ", accuracy)
sample1 = pd.DataFrame(
    {'色泽': '青绿', '根蒂': '蜷缩', '敲声': '浊响', '纹理': '清晰', '脐部': '凹陷', '触感': '硬滑',
     '密度': 0.697, '含糖率': 0.46, '好瓜': '?'}, index=[0])
prediction = predict(df, sample1)
print(f'新样本的预测结果: {"好瓜" if prediction else "坏瓜"}')