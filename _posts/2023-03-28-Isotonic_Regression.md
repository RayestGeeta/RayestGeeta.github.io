---
title: 保序回归(Isotonic Regression)
author: Rayest
date: 2023-03-27 01:08:00 +0800
categories: [Recommendation]
tags: [technology]
math: true
---

## 1. __模型校准__
### 1.1 为什么要校准
- 推荐系统或广告算法中，预测用户购买通常是二分类的有监督问题，要不买或者不买，几乎没有是以用户实际购买概率来进行建模(很难做)。通常模型预估输出的值经过`sigmoid`之后值域在`[0,1]`, 我们称这个值为概率，数学上表示为 $P(Y=c \mid X)$，X代表某条样本，$Y=C$ 是预测该样本分类为C，但其实在电商场景中这个概率很难说是用户购买某个商品的真实概率，原因：
  - 多目标：通常电商场景或其他推荐等场景，要兼顾点击率，加车率，访购率目标，再通过LTR融合为一个分数，最终使用该分数排序，但这个分数已经不具备任何的概率意义了。
  - 数据采样：在绝大数推荐或广告场景中，有非常明显的长尾效应，往往在模型训练时要对负样本做采样。模型预估出的概率分数和真实概率有很大的偏差。
  - 模型局限性：特征不充分，模型未完全达到SOTA等因素，模型的输出不是最优解
- 当然预估概率存在偏差是常见的，比如在推荐大多数场景中，不需要关注预估概率与真实的偏差，只用来排序。但是仍有一些场景需要做预估校准：
  - 广告点击排序：广告点击率预估实际上是保证用户体验，推出用户可能感兴趣的，但平台需要考虑推送广告的收益，通常排序公式为$pctr^\alpha * bid^\beta$，这个时候就需要考虑`pctr`的真实概率。
  - 电商排序：和广告排序很类似，如果只用`pctcvr`排序会出现一个问题，往往推给用户的都是低价商品，但电商场景中`GMV`是很重要的。通常电商场景排序公式为$pctcvr^\alpha * price^\beta$


### 1.2 保序回归


## 2. __工业界保序回归做法__

- 构造数据
  - 模拟比较真实的电商数据，共100
  
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  from matplotlib.collections import LineCollection
  from sklearn.linear_model import LinearRegression
  from sklearn.isotonic import IsotonicRegression
  from sklearn.utils import check_random_state

  n = 100000

  model_predict = np.random.random(n)
  model_predict.sort()

  p = np.linspace(0.01, 0.2, n)
  order_label = np.random.binomial(1, p)
  ```

- 分桶
  ```python
  # 分桶前分数预处理：
  def func(x):
      return np.log10(1+1000*x)/3
      
  rankData['new_score'] = func(rankData['rankscore'].values)
  rankData.sort_values(by = ['new_score'], na_position='first', inplace=True)
  N_bucket = 4000

  bins = [i/N_bucket for i in range(int(N_bucket + 1 - 1000))] + [i/N_bucket for i in range(int(N_bucket + 1 - 1000), N_bucket+1, 2)]

  bucket_res = pd.cut(low_rankData['new_score'].values, bins=bins)
  low_rankData['bucket'] = bucket_res.codes
  low_groupData = low_rankData.groupby('bucket').agg(['mean','count'])
  ```
- PVA
  - sklearn直接调用：
  ```python
  ir = IsotonicRegression()
  high_y_ = ir.fit_transform(high_groupData.index, high_groupData['orderLabel']['mean'])
  ```
  - 具体代码实现(参考):
  ```python
  def isotonic_regression(x, y):
      n = len(x)  # 样本数量
      p = [0] * n  # 输出预测值
      
      # 对x排序
      indices = list(range(n))
      indices.sort(key=lambda i: x[i])
      
      # 初始化y_mean为y的累计和，方便后续均值求解
      y_mean = [y[indices[0]]]
      for i in range(1, n):
          y_mean.append(y_mean[-1] + y[indices[i]])
          
      # 计算分段平均值
      breakpoints = []
      for i in range(n):
          if (i == 0 or x[indices[i]] > x[indices[i - 1]]):
              slope = y_mean[i] / (i + 1)
              breakpoints.append(slope)
      
      # 对每个输入x，计算对应的保序回归值
      k = len(breakpoints)
      for i in range(n):
          j = 0
          while j < k and x[indices[i]] >= x[indices[j]]:
              j += 1
          p[indices[i]] = breakpoints[j - 1]
          
      return p
  ```
- 线性插值
  ```python
  # 高价权重生成
  def sir_weight(p1, p2 ,r1, r2):
      a = (r2-r1)/(p2-p1)
      b = r1 - a*p1
      return a, b

  N_bucket = 3000
  bins = [i/N_bucket for i in range(int(N_bucket + 1 - 1000))] + [i/N_bucket for i in range(int(N_bucket + 1 - 1000), N_bucket+1, 2)]
  high_bin_nums = len(bins) - 1

  high_bucket_weight_bias = {}

  bin_bucket, high_weight, high_bias = [], [], []

  flag = True
  for i in range(len(high_groupData)-1):
      bucket_i = high_groupData.index[i]
      bucket_i_1 = high_groupData.index[i+1]
      
      pi = bins[bucket_i]
      pi_1 = bins[bucket_i_1]
      
      ri = high_y_[i]
      ri_1 = high_y_[i+1]
      
      weight, bias = sir_weight(pi, pi_1, ri, ri_1)
      
      if flag and weight != 0.0:
          for i in range(bucket_i):
              high_bucket_weight_bias[i] = [weight, bias]
          flag = False


      for i in range(bucket_i, bucket_i_1):
          high_bucket_weight_bias[i] = [weight, bias]

              
  last_bucket = high_groupData.index[-1]
  for i in range(last_bucket, high_bin_nums+100):
      high_bucket_weight_bias[i] = high_bucket_weight_bias[i-1]
      
  ```


## 3. __存在的问题__


## 参考文档
[稀疏CTR离线化分桶](https://zhuanlan.zhihu.com/p/450221388)

[使用 Isotonic Regression 校准分类器](http://vividfree.github.io/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/2015/12/21/classifier-calibration-with-isotonic-regression)

[Calibrating User Response Predictions in Online Advertising](/assets/files/sub_214.pdf)