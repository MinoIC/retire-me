#### retire-me

股价预测结合 FIRE 理论来帮助提前退休的项目

- 数据集来自 yahoo finance 的道琼斯工业平均指数中的 30 家公司的股价日线历史
- 运行环境为 python3.8，keras 前端和 tensorflow 后端，需要 cudnn
- 每次训练后会将模型保存为 MODEL_NAME.h5，预测结果保存为 predict.png
- 数据集下载后可直接放在项目目录下，程序会读取相对目录

#### run

安装 lstm.py 中所需的依赖后，可以直接通过 python ./lstm.py 来运行并直接输出一个预测结果且保存截图为 predict.png