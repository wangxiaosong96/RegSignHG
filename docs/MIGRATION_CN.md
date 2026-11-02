# 将修订包放入现有仓库

## 应放在仓库根目录的内容

解压本包，进入 `RegSignHG_revision` 文件夹，把里面的内容放入仓库的
revision 分支。根目录应直接看到 `README.md`、`train.py`、`evaluate.py`、
`predict.py`、`requirements.txt`、`configs/`、`regsignhg/`、`scripts/`、
`tests/`、`docs/` 和 `data/README.md`。

只把 zip 文件上传到 GitHub 不会形成可直接运行的源码仓库。
不应在 README 根目录命令仍然写 `python train.py` 时又把所有代码放进
多余的一层子文件夹。

## 文件职责对应

| 旧包职责或文件 | 修订位置 |
|---|---|
| run.py 的训练流程 | train.py 和 regsignhg/engine.py |
| parameter.py | configs/*.yaml 和 regsignhg/config.py |
| generate.py 的加载部分 | regsignhg/data.py |
| graph.py 和 algorithm.py | regsignhg/model.py |
| util.py 损失和指标 | regsignhg/losses.py 和 metrics.py |
| 旧架构展示 | README 的 Mermaid 架构图；论文图仍需用准确的 Figure 2 源文件 |
| 论文描述对应 | docs/METHOD_CODE_MAP.md |

保留旧版本历史。若保留旧 MVGCN 或 CSIH 源文件，明确归入 legacy 并说明
它们不是新默认训练入口；README 不再指向不存在的脚本或旧模型流程。
仓库中的旧 flow chart.png 应替换为与最终方法一致的图，不能只改图片文件名。

## 接入真实数据后的顺序

先审计三套数据来源、实体映射和哈希，再核实真实特征与元数据。
把示例配置替换为验证集选定的设置，运行模型并保留所有输出。
根据真实结果修改论文，再将精确 commit 与 release tag 写入回复信。
本包没有替你上传、覆盖或发布远程仓库。
