# RegSignHG 训练入口与代码修订使用说明

这个包已补训练入口、模型、配置、运行记录和论文模块对应表。
它是根据提交稿重新整理的修订实现，不是找回的原始实验代码。
真实三套数据、原稿 checkpoint 和逐折预测没有包含在本包中。
示例超参数也不能自动认定为论文实际配置。

## 一 应该怎样替换旧仓库

先保留原仓库历史，创建一个 revision 分支，将本包作为可审查的新实现。
不要把旧模型改名为 RegSignHG 后宣称已恢复原稿结果。
新 `train.py` 直接调用 `regsignhg/engine.py`；所有组件都可在
`docs/METHOD_CODE_MAP.md` 对照公式查看。

本包修正了旧包中的实际问题：缺少持久化运行证据、图级损失退化、
旧版本解码器输入维度不匹配、旧版本按测试集 AUC 选择轮次。
逐项位置和新旧差异见 `docs/OLD_CODE_AUDIT_CN.md`。
这些发现针对已核对的旧压缩包，不代表已经核查当前公开 GitHub 的全部内容。

## 二 先验证软件流程

解压后进入 `RegSignHG_revision`。先用 Python 3.12 创建虚拟环境：

```bash
python -m venv .venv
```

Windows 命令提示符运行 `.venv\Scripts\activate` 激活环境，
Linux/macOS 运行 `source .venv/bin/activate`。激活后再执行：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/make_toy_data.py --out data/toy
python train.py --config configs/smoke.yaml --output outputs/smoke --allow-synthetic
python evaluate.py --checkpoint outputs/smoke/seed_7/fold_1/best.pt --allow-synthetic
```

上述数据是随机生成的软件测试数据，分数没有生物学意义，不得填入论文。
重跑时选择新的输出目录，程序会拒绝覆盖已有实验记录。

## 三 换成真实数据

每个数据集目录需提供：`interactions.csv`、`drug_features.npz`、
`gene_features.npz`、`provenance.json`。
格式见 `data/README.md`。必须保留每行特征对应的药物或基因 ID。
不能凭排序猜测 `.npy` 特征与实体的对应关系。

如果已有真实 NPY 特征和准确的行顺序表，使用 `scripts/import_features.py`
转换；如果没有真实预训练特征，可按 README 的提取器准备，但应核对
实际 checkpoint、蛋白 isoform 和分子预处理，并重新运行受影响的实验。
缺少特征时程序会报错，不会自动使用 one-hot、DGI 或随机数组代替。

```bash
python scripts/audit_data.py --data-dir data/DGIdb --output dgidb_audit.json
python train.py --config configs/dgidb.yaml --output outputs/dgidb_revision
python train.py --config configs/drugbank.yaml --output outputs/drugbank_revision
python train.py --config configs/lincs.yaml --output outputs/lincs_revision
```

只检查 Table 1 所需计数时可先运行：

```bash
python scripts/audit_data.py --interactions data/DGIdb/interactions.csv
```

它会读取真实文件统计，不会写入或凑齐原稿的节点数与边数。

## 四 每次运行会保存什么

`best.pt` 保存验证集选择的最佳权重；`history.csv` 保存每轮各损失和验证 AUC；
`test_predictions.csv` 保存每个测试药物基因对的真实符号和两类概率；
`resolved_config.json`、`environment.json`、`data_audit.json`、`all_splits.json`
保存配置、环境、数据哈希与精确划分。

所有表格数值应从这些预测文件计算。没有对应运行证据的旧结果不能因为
新程序可以运行就认定已经复现。不同种子的平均值、折间标准差与种子均值
标准差在 `summary.json` 中分开标记，程序不生成估计方差或显著性星号。

## 五 需要在论文方法中同步说明的选择

本修订实现采用正负消息的对称度归一化、超图之前的门控、四部分配对向量、
两类 softmax、全节点均值 readout。负图在输入特征和训练边符号上破坏后重新
编码，避免均值池化后的置换不变性。正视图不翻转边的符号。

超边分配为完整 softmax 后保留 top-r，不重新归一化；解相关采用每个通道的
列归一化。增强概率为固定设置，不能称为已经实现的 adaptive augmentation。
更多实现细节及与公式的对应在模块表中列明。

本包仅实现按边的交叉验证。实体冷启动、真实数据构建、基线适配、三分类、
Figure 5 稀疏分桶及 Figure 6 生物学富集尚需各自的实验流程。
`export_hypergraphs.py` 可导出真实原型成员，但不自动给意图附加通路名称。

## 六 回复审稿人前的最后一步

完成真实数据审计并按最终配置重跑后，再把 Git commit、环境、数据哈希、
逐折预测和结果变化写入回复。若原训练也使用了存在问题的流程，应明确说明
重算范围和变化。`docs/RESPONSE_TEMPLATE_CN.md` 提供需要实测信息填写的模板。

