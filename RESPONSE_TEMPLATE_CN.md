# Reviewer 1 代码问题回复模板

本模板须在真实数据核查和重跑之后填写。修订包的软件运行测试不能替代
原稿结果的复现。不要在尚未完成时提交以下完成时态的回复。

## 正式回复需要填写的证据

| 内容 | 应填入的真实证据 |
|---|---|
| 问题原因 | 是公开发布错误、实际实现不一致，还是二者均存在 |
| 修订版本 | Release tag 与完整 Git commit |
| 方法对应 | 论文公式到代码类或函数的对应表 |
| 执行命令 | 从干净环境到三套数据训练的实际成功命令 |
| 数据证据 | 来源版本、实体映射、校验和与最终统计 |
| 数值影响 | 重跑前后 Table 2、消融和其他受影响图表的变化 |

## 英文候选回复

Thank you for identifying the discrepancy between the manuscript and the
released repository. We audited the repository against the experimental
records and identified [verified cause and affected scope]. We have provided
a revised release at [release URL and full commit], including an executable
training entry point, dataset-specific configurations, dependency information,
and a mapping from each manuscript component to its implementation.

The revised code includes separate activating and inhibitory message channels,
the sign-aware gate, parameterized signed-intent hypergraph propagation,
node-level and graph-level contrastive regularization, intent decorrelation,
and the two-class decoder. We corrected [verified implementation or release
issues] and documented all implementation choices that were previously
underspecified.

We [verified or recomputed] the affected experiments using [actual data
versions and protocol]. The resulting configurations, input checksums,
validation-selected checkpoints, and per-pair test predictions are available
at [location]. Changes to the reported results are summarized in [table or
supplement location]. We revised the corresponding claims accordingly.

## 填写时的限制

如果采用本包重新实现了部分方法，应明确说明 revised implementation，
不能称为找回了原始源码。若发现原实验也存在错误，应说明修正和重算，
不能笼统归因于上传问题。没有真实重跑记录时，不写 results remain unchanged。
