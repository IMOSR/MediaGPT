# Media LLaMA ：中文自媒体大模型

虽然LLaMA模型在通用领域通过指令微调已经展示出了令人印象深刻的性能，但对于自媒体创作、直播和运营等领域，由于缺乏专业的训练数据，其能力仍有待提高。为了解决这个问题，我们提出了Media LLaMA，一个针对自媒体领域进行特殊训练的模型。

Media LLaMA首先在大规模自媒体语料上进行连续预训练，系统地学习自媒体的知识体系。然后，我们借助ChatGPT收集了一批关于抖音运营、短视频创作、巨量千川投放、直播运营和直播话术技巧等领域知识问题的分析和回答，并利用这些数据对模型进行指令微调，使模型习得如何将自媒体知识应用到实际场景中。

我们的模型具有以下能力：

1. **掌握自媒体知识：** 能够理解抖音运营、短视频创作、巨量千川投放、直播运营等领域的核心概念和策略。

2. **适用于实际操作：** 能够以通俗易懂的语言解释自媒体概念，并进行基础的自媒体运营咨询，涵盖内容创作、平台运营、广告投放等领域。

为了推动中文自媒体大模型的开放研究，我们将开源一系列**自媒体领域的指令微调数据**和**基于LLaMA训练的中文自媒体大模型的参数**。

## 模型效果

    Human:

    请介绍一下抖音带货的运营策略

    Assistant:
    抖音带货的运营策略可以从以下几个方面进行： 
    1.了解目标受众，根据受众的兴趣和需求来选择适合的产品；
    2.与优质的主播合作，提高直播间的影响力和吸引力；
    3.利用抖音平台的特点，如短视频引流和直播间互动等，将产品销售地带向用户；
    4.定期分析数据，了解用户对产品的反应和偏好，根据数据调整运营策略；
    5.与供应商和品牌建立良好的合作关系，以获得更优惠的佣金和商品资源。


    Human:

    如何做好停留模型来补充销量和转化？

    Assistant:
    要在直播间中做好停留模型的补充，可以考虑以下方面：
    1.提供有吸引力的内容，给观众带来更多的价值和信息，增加他们停留的动机和感兴趣；
    2.与观众进行互动，回答他们的问题，建立良好的交流和互动关系；
    3.提供购买商品的便利，如设置商品链接、提醒观众下单等，增加转化率。
    4.提供与直播内容相关的其他内容，如教程、折扣促销等，吸引观众的关注和参与。



## 更新计划

- [x] (20230629)开源抖音创作、运营和直播相关的指令微调数据

- [x] (20230629)开源7B的指令微调模型

- [ ] 开源更大规模指令微调数据集(预计十万级别)
- [ ] 开源13B的指令微调模型
- [ ] 开源33B的指令微调模型
- [ ] 开源通用自媒体领域的预训练模型

## 训练数据

### 自媒体领域纯文本语料

我们收集了部分公开的自媒体运营、短视频创作等数据用于模型的连续训练。

### 自媒体领域指令微调数据

我们将一些关于抖音运营、短视频创作、巨量千川投放、直播运营等领域的问题输入ChatGPT，并让其生成对每个问题的回答。

#### 开源数据概览
| 文件名                | 规模    | 描述               |
|--------------------|-------|------------------|
| [data/tiktok_v1](https://github.com/IMOSR/Media-LLaMA/tree/master/data/tiktok_v1) | 3,000 | ChatGPT生成的抖音知识问答 |


#### 数据
我们将都培训相关知识输入ChatGPT，让其生成对应的数据，代码地址: generate_data/generate_tiktok.py，样例如下：

    Human:

    如何在直播间中选择和组合产品?

    Assistant:
    1. 定一个北极星指标，选择一到两个引流品和三到五个爆品，还有三到四个利润款。
    2. 引流品应具备一见钟情的特点，不需要过多解释和讲解，吸引观众主动关注和购买。
    3. 爆品是卖点强、热销的产品，具有薄利多销的特点，可以帮助提升销量。
    4. 利润款是能够带来高利润和溢价的产品，让观众愿意为之付出更多。
    5. 根据直播间的阶段和目标进行选择和组合产品，如前期引流品较多，后期利润款较多。
    6. 根据产品的定位和特点来进行组合，确保产品之间有协调性和互补性。\n7. 考虑直播间的流量和观众需求，选择适合的产品来满足他们的购买欲望。
    8. 不断观察和分析产品的表现和销售数据，进行调整和优化选择。
    9. 考虑市场趋势和潮流，选择具有竞争力和吸引力的产品。
    10. 根据产品的类别、价格和受众群体进行筛选和安排，确保产品的多样性和吸引力。

    Human:

    请介绍一下抖音带货的运营策略。

    Assistant:
    1. 确定目标受众，了解目标受众的喜好、需求和购买习惯。
    2. 建立一个吸引人的个人品牌形象，包括昵称、头像、简介等。
    3. 精选高质量的商品，确保商品的品质和性价比。
    4. 制定合适的推广计划，包括直播时间、直播内容和促销活动。
    5. 运用直播间工具，如商品链接、优惠券和红包，增加转化率。
    6. 与粉丝互动，回答问题、提供购买建议，并提供售后服务。
    7. 定期进行数据分析，了解直播效果，并根据数据调整策略。
    8. 与品牌商合作，提供更多的优惠和资源。
    9. 不断学习和提升自己的直播技巧，增加吸引力和影响力。
    10. 持续关注市场和竞争对手，及时调整策略和推广内容。
  
  
    
## 模型参数

我们目前公开了以下版本的Media LLaMA：
* `media-alpaca-lora-7b-beta0.1`: 以[Chinese-LLaMA-7B](https://github.com/ymcui/Chinese-LLaMA-Alpaca)为基础，未经过自媒体语料连续训练，使用通用指令和自媒体指令进行SFT。

### 下载地址：
链接：https://pan.baidu.com/s/1tEuj0SvwJK4czQPCE6gI9w?pwd=onfo 
提取码：onfo

## 训练和测试教程

训练和测试教程请参考: [train/finetune_media_alpaca_lora.ipynb](https://github.com/IMOSR/Media-LLaMA/tree/master/finetune_media_alpaca_lora.ipynb)


## 讨论群

* 1 如果二维码过去加群主微信：yydsa0007 备注：智媒大模型
* 2 扫码

![img_1.png](img_1.png)
## 致谢

本项目的开放过程中，获得了以下项目的文档和帮助，用到的代码不在项目中一一标出了，在此表示感谢。

https://github.com/ymcui/Chinese-LLaMA-Alpaca


## 局限性和使用限制

本项目内容仅供用于学术研究，不得用于商业以及其他可能对社会带来危害的用途。使用涉及第三方代码的部分时，请严格遵循相应的开源协议。

本项目中使用的数据由ChatGPT生成，未经严格验证，可能会存在错误内容，在使用时请注意甄别。

本项目中的模型输出并非专业自媒体运营建议，可能会包含错误内容。如需自媒体运营援助，请向专业人士寻求帮助。

## 引用

如果您使用了本项目的内容，或者认为本项目对您的研究有帮助，请引用本项目。

