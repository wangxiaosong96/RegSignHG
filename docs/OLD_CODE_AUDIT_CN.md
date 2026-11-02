# 已核对的旧代码包问题

以下结论来自本次读取的三个历史压缩包。它们不能代表当前远程仓库的完整状态，
也不能证明哪一份代码实际产生了投稿结果。原包未被覆盖。

| 文件 | SHA256 |
|---|---|
| CSIH_all_python_files.zip | 4760ea151ef15cb57f140c79488742f1737394547508d84830081bf9f0c5132a |
| CSIH_GitHub_modules.zip | 4a1cac9071a452b8e8f95baf3f235ea4125e47a8e3912eb1e8d7ca25dfb03b4e |
| CSIH_reproducible_code.zip | b2ddeb8705cf39637fb55f533371f02a8de8de56948c2a0e19922b29d793c776 |

## 1 较新的模块包已有 run.py 但仍缺复现证据

`CSIH_all_python_files.zip/run.py` 使用内部分层验证集选择模型，
不是所有旧包都使用测试集选模型。不过，它主要打印汇总指标，没有保存
checkpoint、逐对预测、精确划分和完整环境。`train.py` 缺失可以通过
新增入口解决，但这些运行证据还需要在新训练中生成。

## 2 图级损失退化

`CSIH_all_python_files.zip/util.py` 第 38-50 行，以及模块包的同名函数，
先对 z2 的节点顺序翻转，再取均值。节点置换不改变均值，所以负图向量
与正图向量相同，两个 logits 相同，二分类交叉熵在数学上等于 log(2)。
不能用它证明 GraphCL 带来原稿所示性能变化。

新实现先改变输入特征和负图符号，再通过网络重新编码；测试检查了负图
readout 不同且 graph loss 对参数具有非零梯度。正视图仍保持符号。

## 3 Uni-Mol 提取器是占位函数

`CSIH_all_python_files.zip/extract_unimol.py` 的
`encode_with_your_unimol_pipeline` 直接抛出 NotImplementedError。
它不能支持“代码已包含真实 Uni-Mol 特征生成”的声明。
新包提供官方 API 调用、严格加载真实本地权重和特征哈希记录，但当前
没有原始预训练 checkpoint，因此该外部提取流程尚未经过实际权重验证。

## 4 缺少特征时会自动使用单位阵

`CSIH_all_python_files.zip/generate.py` 在特征路径缺失时创建 np.eye。
特征仅按数组读取，没有强制验证行对应的实体 ID。
真实实验修订中应拒绝缺失特征，并通过 IDs 对齐，而不是默默换成 one-hot。
新加载器对缺失、重复 ID、非有限值及来源元数据做显式检查。

## 5 另一个更早的包存在解码维度错误

`CSIH_reproducible_code.zip/CSIH_reproducible_code/src/model.py` 中 decoder
首层声明输入为 2*hidden_dim，但 predict_pairs 拼接 drug、gene、乘积
三个向量，实际为 3*hidden_dim。该路径会导致矩阵乘法维度不匹配。
新实现按稿件构造四部分向量，首层输入为 4*hidden_dim，并输出两类 logits。

## 6 更早的包直接使用测试 AUC 选择轮次

`CSIH_reproducible_code.zip/CSIH_reproducible_code/src/train.py` 第 80-90 行
在训练期间反复评估测试集，按其 AUC 更新 best。它不能作为独立测试结果。
该版本对单个池化图向量应用批内 InfoNCE，还缺少有效负图。

新训练器使用独立验证边选权重，最后才评分测试集。测试还验证了：在
固定划分下改变测试标签，不会改变训练权重或最佳轮次。

## 7 旧模型与当前提交稿仍有结构差异

较新的 `graph.py` 与 `algorithm.py` 使用 late sign fusion、三部分
配对向量和单个 sigmoid logit。提交稿写的是超图前门控、四部分配对
向量和两类 softmax。单纯把 CSIH 或 MVGCN 的类名替换成 RegSignHG
不会消除这些差异。新代码采用提交稿的结构，并明确列出了未充分定义的
归一化、readout 和增强选择。

## 8 不能由修订包得出的结论

本次没有取得三套真实训练数据和原稿预测，不能确认真实数据统计、LINCS
与 DrugBank 的重复问题、论文性能或消融幅度。应在可追溯的真实文件上
重跑后，再决定哪些表图需要更正，并向编辑说明结果变化。

