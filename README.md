# Learning Path of Data Analysis
## 学习清单
1.Udacity数据分析（入门） -->2018.04 已完成（[证书](https://confirm.udacity.com/LGUKXDDT)）

- Python入门 ;  pandas ； SQL ; 数据分析入门 ； 数据可视化 ；统计学基础   

2.Udacity数据分析（进阶） -->2018.07 已完成([证书](https://confirm.udacity.com/KFHYJG6D))

  - 数据挖掘 ； 数据评估及清洗；R语言；Tableau可视化

3.Udacity数据科学家 -->2019.09 已完成([证书](https://confirm.udacity.com/42FDMFVD))

- 有监督学习；深度学习；无监督学习；软件工程；数据工程；统计实验设计；推荐系统；Spark及云服务

## 项目清单

### 为Sparkify寻找“危险”用户

**技能**：Spark、有监督学习

**描述**：利用AWS与Spark对12GB的Sparkify用户数据进行清洗、特征构造及建模，预测用户是否会取消会员订阅，最终取得了0.7045的F1-score。

**项目开源链接**：[Github](https://github.com/CapAllen/Sparkify)

### 音乐软件推荐系统设计

**技能**：推荐系统、有监督学习

**描述**：利用Knowledge Based，Collaborative Filtering Based ，Content Based以及机器学习的方法对音乐软件Sparkify构造新老用户的推荐系统。

**项目开源链接**：[Github](https://github.com/CapAllen/Learning-Path/tree/master/%E4%B8%BASparkify%E6%9E%84%E5%BB%BA%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F)

### 优化Starbucks营销策略

**技能**：实验设计、有监督学习

**描述**：利用A/B-test及有监督学习模型优化Starbucks营销的增量回应率(IRR)与净营收增量(NIR)，IRR由0.0077提升至0.0228，NIR由-759.95提升至298.10；

**项目开源链接**：[Github](https://github.com/CapAllen/Learning-Path/tree/master/%E4%B8%BAStarbucks%E4%BC%98%E5%8C%96%E8%90%A5%E9%94%80%E7%AD%96%E7%95%A5)

### 挖掘客户细分

**技能**：无监督学习、特征工程

**描述**：利用Bertelsmann提供的德国人口普查数据与Arvato提供的客户数据，使用无监督学习方法去做市场细分，筛选潜在客户，并进行精准营销。

**项目开源链接**：[Github](https://github.com/CapAllen/Learning-Path/tree/master/%E4%B8%BAArvato%E5%88%9B%E5%BB%BA%E5%AE%A2%E6%88%B7%E7%BB%86%E5%88%86)

### 图像分类器

**技能**：深度学习、Pytorch

**描述**：利用深度学习算法CNN构建花卉图像分类器，并部署成命令行应用。我在本项目中，利用Pytorch对数据进行加载和预处理，并利用多种CNN算法训练图像分类器，最终达到了80%以上的准确率。

**项目开源链接**：[Github](https://github.com/CapAllen/Learning-Path/tree/master/%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%99%A8)

### 为慈善机构寻找捐献者

**技能**：有监督学习、scikit-learn

**描述**：依据人口普查数据，挖掘更有可能像慈善机构捐款的人物特征。我在本项目中对慈善机构CharityML 提供的数据进行清洗、特征选择与构造，对比利用多种有监督学习方法，并择优调优，最终实现了正确率0.8707，F1-score 0.8683的结果。

**项目开源链接**：[Github](https://github.com/CapAllen/Learning-Path/tree/master/%E4%B8%BA%E6%85%88%E5%96%84%E6%9C%BA%E6%9E%84%E5%AF%BB%E6%89%BE%E6%8D%90%E7%8C%AE%E8%80%85)

### @WeRateDogs

**技能**：数据分析、可视化、pandas、matplotlib、seaborn

**描述**： 通过不同的方式收集推特用户 [@dog_rates](https://twitter.com/dog_rates) 的档案，对收集的数据进行清洗，完成分析及可视化！其中数据清洗部分是难点。

**链接**：[Github](https://github.com/CapAllen/Learning-Path/tree/master/%40WeRateDogs)

### FBI枪支数据分析

**技能**：数据分析、可视化

**描述**：此项中使用了两个数据集，一个来自联邦调查局 (FBI) 的全国即时犯罪背景调查系统 (NICS)，另一个数据集收纳了美国的人口普查的州级数据 (U.S. census data)，对两个数据集进行探索性数据分析，进行可视化，得出结论。

**链接**：[GitHub](https://github.com/CapAllen/Learning-Path/tree/master/FBI%E6%9E%AA%E6%94%AF%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90)

### 分析A/B测试结果

**技能**：统计学、statsmodels

**描述**：利用某电子商务网站运行的 A/B 测试的数据，利用统计学知识来帮助公司弄清楚他们是否应该使用新的页面，保留旧的页面，或者应该将测试时间延长，之后再做出决定。

**链接**：[Github](https://github.com/CapAllen/Learning-Path/tree/master/%E5%88%86%E6%9E%90AB%E6%B5%8B%E8%AF%95%E7%BB%93%E6%9E%9C)

### 拍拍贷数据分析

**技能**：R语言数据分析

**描述**：在p2p企业中，用户还款逾期会对公司的资金链产生冲击，欠款金额越高，逾期时间越久，冲击就会越大，为了减少这种冲击，我们应该尽可能的降低逾期事件的概率。 所以在此项目中，我们的目标是通过对现有数据的分析及可视化，并引入了新变量`逾期比`指出具有哪些特征的标容易逾期。

**链接**：[Github](https://github.com/CapAllen/Learning-Path/tree/master/%E6%8B%8D%E6%8B%8D%E8%B4%B7%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90)

### 探索未来气候发展趋势

**技能**：Excel、SQL

**描述**：使用SQL从数据库中提取数据，并在Excel中完成数据处理和可视化，得出结论。

**链接**：[Github](https://github.com/CapAllen/Learning-Path/tree/master/%E6%8E%A2%E7%B4%A2%E6%9C%AA%E6%9D%A5%E6%B0%94%E5%80%99%E5%8F%91%E5%B1%95%E8%B6%8B%E5%8A%BF)

### 2016年8月上海市摩拜数据分析

**技能**：Tableau

**描述**：使用Tableau创建一个数据分析故事，并做分享。

**链接**：[Github](https://github.com/CapAllen/Learning-Path/tree/master/2016%E5%B9%B48%E6%9C%88%E4%B8%8A%E6%B5%B7%E5%B8%82%E6%91%A9%E6%8B%9C%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90)

## 心得与总结

- [Airbnb-in-Beijing](https://github.com/CapAllen/Airbnb-in-Beijing)
- [Disaster-Response-Pipelines](https://github.com/CapAllen/Disaster-Response-Pipelines)

- [Python Code of Mini-Batch Gradient Descent](http://www.capallen.top/dsnd/2018/12/02/mini-batch-gradient-descent/)

- [Matplotlib可视化](http://www.capallen.top/dand-vip/2018/08/28/%E7%AC%AC%E4%B8%83%E5%91%A8-Matplotlib%E5%8F%AF%E8%A7%86%E5%8C%96/)
- [Pandas数据融合](http://www.capallen.top/dand-vip/2018/08/21/%E7%AC%AC%E5%85%AD%E5%91%A8-2-%E6%95%B0%E6%8D%AE%E8%9E%8D%E5%90%88/)

