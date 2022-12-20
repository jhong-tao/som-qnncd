# som-qnncd
A cognitive diagnostic method
research on counstructinon method of learning path and learning progression based on cognitive diagnosis assessment

# note
## label compare with expert

- 10_3 的数据 label与expert的一致性也比较低（high）
- 20_5 的数据 label与expert的一致性比较低 小于0.8（high）

- 10_3 的数据 label与expert的一致性还可以都大于0.85（low）
- 20_5 的数据 label与expert的一致性比较低 小于0.8（low）

- 20_3和30_5 的数据 low，和high 都还可以
- 10_3的数据low还可以，high比较一般
- 在所有数据集中都呈现，low的一致性比high的一致性更高
- 可用的数据为 10_3, 20_3, 30_5

## models compare with expert

- 整体规律与label compare with expert 相同

## models compare with label

- 10_3 的数据与label的一致性为0.8多 （high）
- 20_5 的低于0.8 不行 （high）
- 20_5 大于0.8 还可以 （low）
- 总结 在所有数据集中都呈现，low的一致性比high的一致性更高
- 可用的数据为 10_3, 20_3, 30_5  (20_5 low)


## experiment

- 横向比较
  - 20_3 和 30_5 的high比较，随着模型参数变复杂，传统模型预测准确率在下降
  - 20_3 和 30_5 的low比较，随着模型参数变复杂，传统模型预测准确率在下降

- 纵向比较
    - 20_3 low 的预测准确率比high的准确率高
    - 30_5 low 的预测准确率比high的准确率高

# 单独比较各种NN
- high 10_3_100 acdm label
- high 10_3_100 acdm expert
- high 10_3_300 acdm label
- high 10_3_1000 acdm expert
- high 10_3_1000 dina label, expert
- high 10_3_100 gdina label
- high 10_3_100 rrum label
- high 20_3_50 gdina label, expert
- high 20_3_100 gdina label, expert
- high 20_3_100 acdm label, expert
- high 20_3_100 rrum label, expert
- high 20_3_300 rrum label, expert
- high 20_5_50 acdm label, expert
- high 20_5_300 acdm label, expert
- high 20_5_1000 acdm label, expert
- high 20_5_100 gdina label, expert
- high 20_5_300 gdina label, expert
- high 20_5_100 rrum label, expert
- high 30_5_300 acdm label, expert
- high 30_5_100 acdm label, expert
- high 30_5_300 acdm label, expert
- high 30_5_300 dina label, expert
- high 30_5_50 gdina label, expert
- high 30_5_300 gdina label, expert

- low 10_3_100 acdm label, expert
- low 10_3_300 acdm label, expert
- low 10_3_300 dina label, expert
- low 10_3_100 gdina label, expert
- low 10_3_1000 gdina label, expert
- low 10_3_100 rrum label, expert
- low 10_3_50 rrum label, expert
- low 10_3_1000 rrum label, expert
- low 20_3_50 rrum label, expert
- low 20_3_300 acdm label, expert
- low 20_3_50 acdm label, expert
- low 20_3_300 acdm label, expert
- low 20_5_100 acdm label, expert
- low 20_5_100 dina label, expert
- low 20_5_300 dina label, expert
- low 20_5_50 gdina label, expert
- low 20_5_300 gdina label, expert
- low 20_5_50 rrum label, expert
- low 20_5_100 rrum label, expert
- low 20_5_1000 rrum label, expert
- low 30_5_300 dina label, expert
- low 30_5_100 rrum label, expert

