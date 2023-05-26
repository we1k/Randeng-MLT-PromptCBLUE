

为推动LLM在医疗领域的发展和落地，华东师范大学王晓玲教授团队联合阿里巴巴天池平台，复旦大学附属华山医院，东北大学，哈尔滨工业大学（深圳），鹏城实验室与同济大学推出**PromptCBLUE**评测基准, 对[CBLUE](https://tianchi.aliyun.com/dataset/95414)基准进行二次开发，将16种不同的医疗场景NLP任务全部转化为基于提示的语言生成任务,形成首个中文医疗场景的LLM评测基准。**PromptCBLUE**将作为CCKS-2023的评测任务之一，依托于天池大赛平台进行评测。

考虑到目前的LLM训练可能涉及商业数据，大规模模型开源受到各种外在条件的限制，我们将对PromptCBLUE评测开放两个赛道：
- 通用赛道：接受来自企业，高校，开源社区，各类研究团队或者个人对自研的LLM进行评测，不需要开源其模型。评测地址：[PromptCBLUE通用赛道评测网站](https://tianchi.aliyun.com/competition/entrance/532085/introduction)
- 开源赛道：接受各类参赛团队提交评测，但是其必须使用开源的大模型底座，且只能使用开源的或者可以全部提交至比赛组织方审核的数据集进行训练/微调。评测地址：[PromptCBLUE通用赛道评测网站](https://tianchi.aliyun.com/competition/entrance/532084/introduction)



## 数据集详情

### PromptCBLUE总体统计


| PromptCBLUE      | -     |
|-------------|-------|
| 版本号         | v0.1  |
| prompt 模板数量 | 94    |
| 训练集         | 68500 |
| 验证集         | 10270 |
| 测试集A        | 10270 |
| 测试集B        | 10270    |



## References

- [CBLUE基准](https://tianchi.aliyun.com/dataset/95414)
- [ChatGLM-6b模型](https://github.com/THUDM/ChatGLM-6B)
