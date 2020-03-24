# 工作计划：
    - 第一阶段工作：完成模块1-6；
    - 第二阶段工作：完成模块7-9；
    - 第三阶段工作：完成所有模块联调；
# 第一阶段工作具体任务分配：
    1. 刘向阳：
        - utils.cython_bbox
        - utils.blob
        - rpn_msr
    2. 梁卓：
        - roi_pooling_layer
    3. 吕江浩：
        - networks
    4. 颜豪杰：
        - utils.timer
        - utils.cython_nms
        - utils.nms
        - utils.boxes_grid
        - nms
    5. 沙建松：
        - fast_rcnn

# Faster-Rcnn-TF/lib 代码分析
    1. utils ：
        - 通用工具包；
        - 模型部分
            - fast_rcnn:
                - test.py: utils.timer/ utils.cython_nms/ utils.boxes_grid/ utils.blob
                - train.py:utils.timer
            - rpn_msr:utils.cython_bbox
        - 数据部分
            - datasets中绝大部分包出现调用:utils.cython_bbox/ utils.boxes_grid
            - gt_data_layer: utils.blob/ utils.cython_bbox/ utils.boxes_grid
            - roi_data_layer:utils.blob/ utils.cython_bbox
    2. rpn_msr ：
        - 这就是RPN的核心代码部分，有生成proposals和anchor的方法；
        - 用于生成提议窗口；
    3. roi_pooling_layer ：
        - TF自定义op；
        - 相当于SPP-NET的一个精简版：实现从原图区域映射到conv5区域最后pooling到固定大小的功能；
        - SPP-NET： 解决RCNN的缺点，通过对于一张图像我们只需要提一次卷积层特征，然后将每个Region Proposal的卷积层特征（通过位置映射取出）输入到全连接层做后续操作，这样做可以节省大量时间。SPP-NET的另一个工作时解决了尺度问题。
    4. networks ：
        - network.py：基础模块，封装了调用tf Python api 实现的不同layer函数，如conv()/relu()等，共15种layer op函数，如 setup/load/feed/getoutput/get_unique_name/make_var/validate_padding共7中layer 设置函数；
        - VGGnet_test.py： 基于network.py构建的 tf 网络模型；
        - VGGnet_train.py：基于network.py构建的 tf 网络模型；
        - factory：networks模块的对外接口，用于返回test或train；
    5. nms ：
        - 做非极大抑制的部分，有gpu和cpu两种实现方式；
    6. fast_rcnn ：
        - config.py：负责cnn训练的配置选项
        - python的训练和测试脚本；

## 相对独立，训练过程有关：
    7. datasets ：
        - 负责数据库读取；
        - factory.py：工厂类，用类生成imdb类并且返回数据库共网络训练和测试使用；
        - imdb.py：这里是数据库读写类的基类，分装了许多db的操作，但是具体的一些文件读写需要继承继续读写；
        - pascal_voc.py：文件操作函数；
    8. roi_data_layer ：
        - 原版中所用的数据获取层；
        - 与模型训练fast_rcnn/train.py紧耦合；
    9. gt_data_layer:
        - Faster-Rcnn-TF中，基于roi_data_layer自定义的数据获取层；
        - 与模型训练fast_rcnn/train.py紧耦合；

# Faster-Rcnn-TF/data
    - 用来存放pretrained模型，比如imagenet上的，以及读取文件的cache缓存
# Faster-Rcnn-TF/experiments
    - 存放配置文件以及运行的log文件，另外这个目录下有scripts可以用end2end或者alt_opt两种方式训练。
        - 交替训练（alt_opt）
        - 近似联合训练（end-to-end）：速度更快，正确率可能也更高
# Faster-Rcnn-TF/models
    - 里面存放了三个模型文件，小型网络的ZF，大型网络VGG16，中型网络VGG_CNN_M_1024。推荐使用VGG16，如果使用端到端的approximate joint training方法，开启CuDNN，只需要3G的显存即可。
# Faster-Rcnn-TF/output
    - 这里存放的是训练完成后的输出目录，默认会在faster_rcnn_end2end文件夹下
# Faster-Rcnn-TF/tools
    - 里面存放的是训练和测试的Python文件。
    - demo.py
        --model VGGnet_fast_rcnn_iter_70000
        --cpu
        : 测试
    - train_net.py
        --gpu 1
        --solver models/hs/faster_rcnn_end2end/solver.prototxt      #模型的配置文件
        --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel #初始化的权重文件，Imagenet上预训练好的模型
        --imdb hs                                                   #hs为自定义数据库名字
        --iters 80000
        --cfg experiments/cfgs/faster_rcnn_end2end.yml
        ： 训练







