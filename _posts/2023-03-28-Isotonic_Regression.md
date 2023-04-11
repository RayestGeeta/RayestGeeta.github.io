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
  - 模拟比较真实的电商数据：共10w条数据，`model_predict`是指模型的打分，随机生成`0-1`之间的随机数。`order_label`是用户真实是否购买的标签，并且随着`model_predict`增大，`order_label`是`1`的概率也更高。
  
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  from matplotlib.collections import LineCollection
  from sklearn.linear_model import LinearRegression
  from sklearn.isotonic import IsotonicRegression
  from sklearn.utils import check_random_state

  n = 100000

  model_predict = np.random.random(n)
  model_predict_raw = np.clip(model_predict_raw, 0, 0.99)
  # 使打分向较小值倾斜
  model_predict_raw = model_predict_raw**2
  model_predict_raw.sort()

  p = np.linspace(0.01, 0.2, n)
  order_label = np.random.binomial(1, p)
  ```

- 分桶
  - 离散化预处理
    - 问题：大多数推荐或广告场景数据都非常稀疏，点击率通常不足10%，多目标融合(通常是多个目标相乘)会使模型打分进一步缩放，我们发现模型打分会非常密集，长尾效应严重。
    - 解决方法：对比度增强算法。简单来说是对数值做映射变换，常用的方式有`对数变换/指数变化/灰度拉伸`，具体介绍可以看[保序回归的工程问题-CTR的离散化方法](https://zhuanlan.zhihu.com/p/450221388)

    - 这里选择对数变换：$s=c \cdot \log _{v+1}(1+v \cdot r)$

  ```python
  # 分桶前分数预处理：
  def func(x):
    return np.log10(1+99*x)/2 
  model_predict = func(model_predict_raw)
  ```

  ![no process](/img/no_process_bucket.jpg){: w="370" h="200" .left}
  ![process](/img/process_bucket.jpg){: w="370" h="200" .normal}

  - 分桶方式
    - `等频`：也称为分位数分桶。也就是每个桶有一样多的样本，但可能出现数值相差太大的样本放在同个桶的情况。
    - `等距`：每个桶的宽度是固定的，即值域范围是固定的，这种适合样本分布比较均匀的情况。我们这里用等距分桶，上面经过离线化预处理，样本相对比较均匀。
    - 其他：`模型分桶`，聚类或树模型等。
  
  ```python
  # 等距分桶
  N_bucket = 100

  bins = [i/N_bucket for i in range(int(N_bucket + 1))]
  bucket_res = pd.cut(model_predict, bins=bins)

  rank_data = pd.DataFrame(np.array([model_predict,order_label,model_predict_raw,  bucket_res.codes]).T, columns = ['model_predict', 'order_label','model_predict_raw', 'bucket'])

  rank_data_group = rank_data.groupby('bucket').agg(['mean','count'])[:]

  rank_data_group['model_predict']['count'].plot()
  plt.ylabel('count')
  plt.xlabel('bucket')
  plt.title('process bucket')
  ```

- PAV(The Pool Adjacent Violators Algorithm)
  - 流程：
    - 学术：对于一个无序数字序列，PAV会从该序列的首元素往后观察，一旦出现乱序现象停止该轮观察，从该乱序元素开始逐个吸收元素组成一个序列，直到该序列所有元素的平均值小于或等于下一个待吸收的元素。
    - 大白话：类似一种合并桶的操作，从第一个桶开始，对于后面违反单调递增的桶与前一个桶合并，最终形成一个单调自增的序列。
    - 数学推导：[保序回归（isotonic regression）](https://zhuanlan.zhihu.com/p/569977824)
    - pool adjacent violators algorithm(PAVA)(后续update)
  - sklearn直接调用：
  
  ```python
  ir = IsotonicRegression()
  y_ = ir.fit_transform(rank_data_group.index, rank_data_group['model_predict']['mean'])
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

  - 评估：
    - mae/mse：我们可以看一下`order_label`和`保序回归前后`的差距
  
    ```python
    # 评估
    from sklearn import metrics

    mse_fix = metrics.mean_squared_error(rank_data_group['order_label']['mean'], y_)
    mae_fix = metrics.mean_absolute_error(rank_data_group['order_label']['mean'], y_)


    mse_rank = metrics.mean_squared_error(rank_data_group['order_label']['mean'], rank_data_group['model_predict']['mean'])
    mae_rank = metrics.mean_absolute_error(rank_data_group['order_label']['mean'], rank_data_group['model_predict']['mean'])

    print('mse:', round(mse_fix, 6),  round(mse_rank, 6))
    print('mae:', round(mae_fix, 6),  round(mae_rank, 6))

    ```

    ```shell
    mse: 5.5e-05 0.233557
    mae: 0.005333 0.419035
    ```

    - 可视化：明显看到保序回归分数和真实的`order_label`很接近了。
    ![offlien fix](/img/offline_fix.jpg){: w="450" h="250" .normal}

    ```python
    #可视化

    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # ax1.bar(x=rank_data_group.index, height=rank_data_group['order_label']['count'], alpha=1)

    ax2.plot(rank_data_group.index, rank_data_group['order_label']['mean'], 'r.', markersize=5)
    ax2.plot(rank_data_group.index, y_, 'b.', markersize=5)
    ax2.plot(rank_data_group.index, rank_data_group['model_predict_raw']['mean'], 'g.', markersize=4)

    plt.legend(('real_ctcvr', 'fix_ctcvr', 'rank_raw'))
    ax1.set_ylabel('number of sample')
    ax2.set_ylabel('ctcvr')
    ax1.set_xlabel('bucket')
    plt.title('offline fix (rank_mse:rankMse, fix_mse:fixMse)'.replace('rankMse', str(round(mse_rank, 4))).replace('fixMse', str(round(mse_fix, 6))))
    plt.savefig('offline_fix.jpg', dpi=100)
    plt.show()
    ```
  
- 线性映射:
  - 要做一个`model_predict`到`保序回归分数`的映射，为了不破环原有的排序效果，要保证映射是严格单调递增的。
  - 这里用的方法是：每个桶做一个线性映射函数，对于第`N`个桶，该桶的区间是`[bn_0, bn_1]`，保序回归第`N`个桶分数为`Rn`，第`N+1`个桶分数为`Rn+1`：
    - weight:  `(Rn+1 - Rn)/(bn_1 - bn_0)`
    - bias:  `Rn - weight*bn_0`
    - 使用: `Fix_score = func(model_predict_raw) * weight + bias`

  ```python
  # 映射权重生成

  # 生成线性映射的weight,bias
  def sir_weight(p1, p2 ,r1, r2):
      a = (r2-r1)/(p2-p1)
      b = r1 - a*p1
      return a, b

  # 桶区间
  N_bucket = 100
  bins = [i/N_bucket for i in range(int(N_bucket + 1))]
  bin_nums = len(bins) - 1

  bucket_weight_bias = {}

  bin_bucket, weight, bias = [], [], []

  flag = True
  for i in range(len(rank_data_group)-1):
      bucket_i = int(rank_data_group.index[i])
      bucket_i_1 = int(rank_data_group.index[i+1])
      
      # 取出第i桶的区间
      pi = bins[bucket_i]
      pi_1 = bins[bucket_i_1]
      
      # 取出第i个和i+1个保序回归分数
      ri = y_[i]
      ri_1 = y_[i+1]
      
      # 求weihgt, bias
      weight, bias = sir_weight(pi, pi_1, ri, ri_1)
      
      # 桶稀疏可能导致前几个桶无数据，那么直接用后面桶的数据
      if flag and weight != 0.0:
          for i in range(bucket_i):
              bucket_weight_bias[i] = [weight, bias]
          flag = False


      for i in range(bucket_i, bucket_i_1):
          bucket_weight_bias[i] = [weight, bias]

  # 最后一个桶使用倒数第二个桶的数据            
  last_bucket = int(rank_data_group.index[-1])
  for i in range(last_bucket, bin_nums+100):
      bucket_weight_bias[i] = bucket_weight_bias[i-1]
      
  ```

  - 保序性验证:
    - 随机生成10w个单调递增在`[0,1]`之间的数，通过保序回归后看一下分数的分区：符合预期，我们模拟的真实`ctcvr`也是在`[0, 0.2]`范围内。
  ![test IR](/img/test_ir.jpg){: w="450" h="250" .normal}

  ```python
  # 测试保序性

  test_x = [i/100000 for i in range(1,100000)]
  test_x2 = func(np.array(test_x))

  N_bucket = 100
  bins = [i/N_bucket for i in range(int(N_bucket + 1))]
  bucket_res = pd.cut(np.array(test_x2), bins=bins)

  y = [bucket_weight_bias[bucket][0]*x+bucket_weight_bias[bucket][1]  for x, bucket in zip(test_x2[:], bucket_res.codes[:])]
  plt.plot(y, label = 'fix')
  plt.plot(test_x, label = 'raw')
  plt.xlabel('bucket')
  plt.ylabel('fix_score')
  plt.title('Test Isotonic Regression')
  
  ```

## 3. __存在的问题__

- 数据分布发生变化：在具体使用过程中，不同时间甚至是随着时间的推移，模型打分的分布是会发生变化的，这可能会使原先的分桶方式效果变得不佳，目前还没有一种很好的自动化调节方式(或许使用聚类分桶会缓解)
- 桶稀疏：发现有些桶校准效果不是很好，主要原因还是因为桶内样本数量偏少不太置信。如果使用等频分桶会使部分桶区间变得极小，如果拟合不准，这个区间内的桶会非常敏感。
- 避免不了的调参：校准之后，我们往往会再乘上其他权重，但是由于校准每天其实都会发生变化，其他权重因子过些时间也会变得不适用。
- 线上响应不及时：特殊时期，群体用户行为会有较大的不同(节庆等)，毕竟保序回归是离线进行的，这种情况下效果会不佳。

## 参考文档

[稀疏CTR离线化分桶](https://zhuanlan.zhihu.com/p/450221388)

[使用 Isotonic Regression 校准分类器](http://vividfree.github.io/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/2015/12/21/classifier-calibration-with-isotonic-regression)

[Calibrating User Response Predictions in Online Advertising](/assets/files/sub_214.pdf)

[保序回归（isotonic regression）](https://zhuanlan.zhihu.com/p/569977824)
