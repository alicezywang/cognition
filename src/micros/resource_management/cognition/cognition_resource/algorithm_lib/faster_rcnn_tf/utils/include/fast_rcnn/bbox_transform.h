#ifndef BBOX_TRANSFORM_H
#define BBOX_TRANSFORM_H
#include <vector>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
namespace fast_rcnn {
using namespace std;
typedef Eigen::Tensor<float, 2, Eigen::RowMajor> Tensor2f;
typedef Eigen::Tensor<float, 1, Eigen::RowMajor> Tensor1f;
typedef Eigen::DSizes<Eigen::Index, 2> Dimensions;
//#函数作用：返回anchor相对于GT的（dx,dy,dw,dh）四个回归值，shape（len（anchors），4）
Tensor2f bbox_transform(Tensor2f ex_rois, Tensor2f gt_rois);//用坐标值计算偏移量

//#boxes为anchor信息，deltas为'rpn_bbox_pred'层信息
//#函数作用:得到改善后的anchor的信息（x1,y1,x2,y2）
Tensor2f bbox_transform_inv(Tensor2f boxes, Tensor2f deltas);

//# 修剪/剪辑boxes
Tensor2f clip_boxes(const Tensor2f &boxes, const Tensor1f &im_shape);//用偏移量更新坐标值

//对于变量来说:
    //extern关键字: 表明该变量可作用于不同编译单元,multiple definition错误
        //原因是在编译阶段，每个源文件都是独立编译的，他们会生成独立的.o文件
        //链接阶段会报错
    //const 关键字: 表明修饰的全局常量只能作用于本编译单元
    //static关键字: 表明修饰的对象只能作用于本编译单元,一般放在src,给其他模块造成不必要的信息污染
    //extern const关键字: 表明该常量可以作用于其他编译单元中

}//namespace
#endif // BBOX_TRANSFORM_H
