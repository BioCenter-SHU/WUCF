# WUCF
# 数据集架构

数据集总共有三个文件夹，分别为renji、luodian、huashan。两个分类，high代表高风险（低-混合回声），low代表低风险（高回声）。

renji

|---- high

|---- low

luodian

|---- high

|---- low

huashan

|---- high

|---- low

# 运行环境

- python=3.8.5
- pytorch==1.12.1
- torchvision==0.13.1
- torchaudio==0.12.1
- cudatoolkit=11.3
- numpy==1.22.3
- torchmetrics==1.0.1
- matplotlib==3.5.2
- scikit-learn==1.1.1

详情请见文件 `环境安装指令.txt`

# DCLCN

![image.png](Read%20Me%20914d5639974d425388a226b8a631ae29/image.png)

### 训练过程：

a，使用源中心的数据和标签来训练特征提取器和两个分类器（预训练）初始化标签

b，源中心+标签，目标中心无标签，加入训练。mmd损失，要拉近目标域和源域的数据特征。

多样性损失，使用两个分类器分别输出目标域的一个结果，来计算距离。

KL（分类）损失，是源域和目标域的数据都会有的一个，这边的话会有上一轮下来的一个伪标签，是使用上一轮的伪标签来和这一次的进行一个kl的计算。

交叉熵损失：源域和他的真实标签。

兼容损失：最开始的时候，（源中心和目标中心）会有一个初始化的那个标签预测，使用那个进行计算。

c，定下来的标签来作为目标中心真实标签来微调。

### 运行

在新的服务器上运行的时候，第一次会报错，重新运行一次就顺利了。

`python [main.py](http://main.py)` 

可根据下面的输入进行参数修改。

```python
parser.add_argument('--all_use', type=str, default='no', metavar='N',
                    help='use all training data? in usps adaptation')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                    help='source only or not')
parser.add_argument('--eval_only', action='store_true', default=False,
                    help='evaluation only option')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate (default: 0.0002)')
parser.add_argument('--max_epoch', type=int, default=110, metavar='N',
                    help='how many epochs')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num_k', type=int, default=4, metavar='N',
                    help='hyper paremeter for generator update')
parser.add_argument('--one_step', action='store_true', default=False,
                    help='one step training with gradient reversal layer')
parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')
parser.add_argument('--resume_epoch', type=int, default=100, metavar='N',
                    help='epoch to resume')
parser.add_argument('--save_epoch', type=int, default=10, metavar='N',
                    help='when to restore the model')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='save_model or not')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--source', type=str, default='huashan', metavar='N',
                    help='source dataset')
parser.add_argument('--target', type=str, default='renji', metavar='N', help='target dataset')
parser.add_argument('--use_abs_diff', action='store_true', default=False,
                    help='use absolute difference value as a measurement')
parser.add_argument('--dataset_folder', type=str, default='/root/dataset/data/', help='where your dataset stored')
```

### 消融实验

将mian.py中**`from** solver_noise_mdd **import** Solver_Noise` 修改为**`from** solver_noise **import** Solver_Noise` 后运行main.py文件。

### 修改参数

修改solver_noise_mdd.py中315行 `loss = loss_s + alpha*lc_s + beta*lo_s + alpha*lc_t - loss_dis`

