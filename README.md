# nGIA3

#### 介绍
nGIA2，线性运行时间的生物序列聚类软件。
类似Linclust，先根据k-mer把序列分成小组，然后再在小组内进行序列比对。
不同之处是重复多次“分组-比对”过程，以修正之前分组中可能存在的分组错误。
通过GPU加速，取得了超过Linclust十倍的运行速度。
数据集尺寸可以大于显存，但不能超过内存。
能自动判断蛋白或者基因序列，自动选择内部最优参数，全傻瓜化操作。

#### 软件架构
分为两个部分：
1.makeDB生成数据库。
2.cluster生成聚类结果。

#### 安装教程

1. 编译安装makeDB
  cd makeDB && make && cd ..
2. 编译安装cluster
  cd cluster && make && cd ..

#### 使用说明

1. makeDB使用说明
usage: ./makeDB/makeDB -f <fasta> -p <packed>  ...  
option:  
  -f    fasta file (string) *  
  -p    packed file (string) *  
  * is necessary.  
-f 是输入的fasta格式文件  
-p 是生成的打包后的序列  
每个数据集只需要打包一次，之后任意相似度的聚类都只把打包后的数据作为输入。  
2. cluster使用说明  
usage: ./cluster/cluster -i <identity> -p <packed> -r <result>  ...  
option:  
  -i    identity 1-99 (int32_t) *  
  -p    packed file (string) *  
  -r    result file (string) *  
  * is necessary.  
-i 是相似度，identity  
-p 是打包后的序列  
-r 是生成的结果  
3. 生成结果解释  
生成的结果类似如下：  
>1  
ACGGT  
  >2  
其中：  
“>”开头的是代表序列的序列名。  
字母开头的是代表序列的碱基或氨基酸  
空格开头的是代表序列所在类内的普通序列，只给出了序列名。  