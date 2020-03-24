#include <fstream>
#include <iomanip>
#include "fast_rcnn/bbox_transform.h"
namespace fast_rcnn{
using namespace std;

//#函数作用：返回anchor相对于GT的（dx,dy,dw,dh）四个回归值，shape（len（anchors），4）
Tensor2f bbox_transform(Tensor2f ex_rois, Tensor2f gt_rois){//ex_rois, gt_rois
    assert(ex_rois.dimensions() == gt_rois.dimensions());

    //#计算每一个anchor的width与height
    auto ex_widths  = ex_rois.chip(2,1) - ex_rois.chip(0,1) + 1.0f; //在第2维度取2
    auto ex_heights = ex_rois.chip(3,1) - ex_rois.chip(1,1) + 1.0f; //.constant(1.0f)
    //#计算每一个anchor中心点x，y坐标
    auto ex_ctr_x  = ex_rois.chip(0,1) + 0.5f * ex_widths;
    auto ex_ctr_y  = ex_rois.chip(1,1) + 0.5f * ex_heights;
    //#注意：当前的GT不是最一开始传进来的所有GT，而是与对应anchor最匹配的GT，可能有重复信息
    //#计算每一个GT的width与height
    auto gt_widths  = gt_rois.chip(2,1) - gt_rois.chip(0,1) + 1.0f; //在第2维度取2
    auto gt_heights = gt_rois.chip(3,1) - gt_rois.chip(1,1) + 1.0f;
    //#计算每一个GT的中心点x，y坐标
    auto gt_ctr_x = gt_rois.chip(0,1) +  0.5f * gt_widths;
    auto gt_ctr_y = gt_rois.chip(1,1) +  0.5f * gt_heights;
    //#要对bbox进行回归需要4个量，dx、dy、dw、dh，分别为横纵平移量、宽高缩放量
    //#此回归与fast-rcnn回归不同，fast要做的是在cnn卷积完之后的特征向量进行回归，dx、dy、dw、dh都是对应与特征向量
    //#此时由于是对原图像可视野中的anchor进行回归，更直观
    //#定义 Tx=Pwdx(P)+Px Ty=Phdy(P)+Py Tw=Pwexp(dw(P)) Th=Phexp(dh(P))
    //#P为anchor，T为target，最后要使得T～G，G为ground-True
    //#回归量dx(P)，dy(P)，dw(P)，dh(P)，即dx、dy、dw、dh
    Tensor1f targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths;
    Tensor1f targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights;
    Tensor1f targets_dw = (gt_widths / ex_widths).log();
    Tensor1f targets_dh = (gt_heights / ex_heights).log();
    //#targets_dx, targets_dy, targets_dw, targets_dh都为（anchors.shape[0]，）大小
    //#所以targets为（anchors.shape[0]，4）
    Tensor2f targets(targets_dx.dimension(0),4);
    targets.chip(0,1) = targets_dx;
    targets.chip(1,1) = targets_dy;
    targets.chip(2,1) = targets_dw;
    targets.chip(3,1) = targets_dh;
    return targets;
}

//#函数作用:得到改善后的anchor的信息（x1,y1,x2,y2）
Tensor2f bbox_transform_inv(Tensor2f boxes, Tensor2f deltas){
    //#boxes.shape[0]=K*A=Height*Width*A
    if(boxes.dimension(0) == 0) {
        return deltas.constant(0.0f);
    }

    //boxes = boxes.astype(deltas.dtype, copy=False) #数据类型转换
    //#得到Height*Width*A个anchor的宽，高，中心点的x，y坐标
    auto widths  = boxes.chip(2,1) - boxes.chip(0,1) + 1.0f; //shape:(n,)
    auto heights = boxes.chip(3,1) - boxes.chip(1,1) + 1.0f;
    auto ctr_x = boxes.chip(0,1) + 0.5f * widths;
    auto ctr_y = boxes.chip(1,1) + 0.5f * heights;

    //#deltas共有84列，依次存（dx,dy,dw,dh）,每一行表示一个anchor
    //#0::4表示先取第一个元素，以后每4个取一个，所以取的index为（0,4,8,12,16...）
    //deltas为84列;按列取值后的形状为(n,21)
    Eigen::array<Eigen::DenseIndex, 2> strides({1,4});
    Tensor2f dx = deltas.stride(strides);

    int deltas_dim0 = deltas.dimension(0);   //n
    int deltas_dim1 = deltas.dimension(1)/4; //84/4
    Tensor2f dy(deltas_dim0,deltas_dim1);
    Tensor2f dw(deltas_dim0,deltas_dim1);
    Tensor2f dh(deltas_dim0,deltas_dim1);
    for(int i=0; i<deltas_dim1; ++i){
        //dx.chip(i,1) = deltas.chip(i*4, 1);
        dy.chip(i,1) = deltas.chip(i*4+1, 1);
        dw.chip(i,1) = deltas.chip(i*4+2, 1);
        dh.chip(i,1) = deltas.chip(i*4+3, 1);
    }

    //#预测后的中心点，与w与h: 同型矩阵相乘
    Eigen::array<int, 2> dims({deltas_dim0, 1});
    Eigen::array<int, 2> bcast({1, deltas_dim1});
    auto widths_tem  = widths.reshape(dims);    //shape:(deltas_dim0,deltas_dim1)
    auto heights_tem = heights.reshape(dims);
    auto ctr_x_tem   = ctr_x.reshape(dims);
    auto ctr_y_tem   = ctr_y.reshape(dims);
    auto widths_bc   = widths_tem.broadcast(bcast);
    auto heights_bc  = heights_tem.broadcast(bcast);
    auto ctr_x_bc    = ctr_x_tem.broadcast(bcast);
    auto ctr_y_bc    = ctr_y_tem.broadcast(bcast);

    auto pred_ctr_x = dx * widths_bc  + ctr_x_bc; //同型矩阵相乘,相加
    auto pred_ctr_y = dy * heights_bc + ctr_y_bc;
    auto pred_w = dw.exp() * widths_bc;
    auto pred_h = dh.exp() * heights_bc;

    //#预测后的（x1,y1,x2,y2）存入 pred_boxes
    Tensor2f pred_boxes(deltas); //用boxes指定形状
    pred_boxes.setZero();
    for(int i=0; i<deltas_dim1; ++i){
        pred_boxes.chip(i*4  ,1) = pred_ctr_x.chip(i,1) - pred_w.chip(i,1) * 0.5f;
        pred_boxes.chip(i*4+1,1) = pred_ctr_y.chip(i,1) - pred_h.chip(i,1) * 0.5f;
        pred_boxes.chip(i*4+2,1) = pred_ctr_x.chip(i,1) + pred_w.chip(i,1) * 0.5f;
        pred_boxes.chip(i*4+3,1) = pred_ctr_y.chip(i,1) + pred_h.chip(i,1) * 0.5f;
    }
    return pred_boxes;
}

//# 修剪/剪辑boxes
Tensor2f clip_boxes(const Tensor2f &boxes, const Tensor1f &im_shape){
    /*****
    Clip boxes to image boundaries.
    *****/
    //#im_shape[0]为图片高，im_shape[1]为图片宽
    //#使得boxes位于图片内
    //# x1 >= 0
    int size = boxes.dimension(0);
    int im_heights = im_shape(0) - 1;
    int im_widths  = im_shape(1) - 1;
    ///方法2:折中实现方式;功能快速实现
    Tensor2f new_boxes = boxes;
    for(int i=0; i<size; ++i){
        if(new_boxes(i,0) > im_widths){
            new_boxes(i,0) = im_widths;
        }
        else if(new_boxes(i,0) < 0){
            new_boxes(i,0) = 0;
        }
        if(new_boxes(i,1) > im_heights){
            new_boxes(i,1) = im_heights;
        }
        else if(new_boxes(i,1) < 0){
            new_boxes(i,1) = 0;
        }
        if(new_boxes(i,2) > im_widths){
            new_boxes(i,2) = im_widths;
        }
        else if(new_boxes(i,2) < 0){
            new_boxes(i,2) = 0;
        }
        if(new_boxes(i,3) > im_heights){
            new_boxes(i,3) = im_heights;
        }
        else if(new_boxes(i,3) < 0){
            new_boxes(i,3) = 0;
        }
    }
    return new_boxes;
}

} //namespace
