/** 输入参数：
// rpn_cls_score:1xhxwx18
// gt_boxes:Nx5 {4个坐标，类别}//2维数组
// im_info:1x3,图片的高、宽、scalef
// data:1xHxWx3
// 运行：
// _anchors:9x4,使用3个ratio（0.5,1,2）和3个scale(8,16,32)对基础框（[0,0,15,15]）做变换，得到9种框。
// height,width:缩小16倍后的特征图的高宽
主要作用：给每个位置的9个anchor生成表示正负样本的label和回归的目标值，以及权重，提供给RPN进行训练
==============================================================================*/

// An anchor_target_layer Op.

#include <stdio.h>
#include <cfloat>
#include <functional>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
#include "generate_anchors.h"
#include "bbox.h"
#include "fast_rcnn/config.h"
#include "fast_rcnn/bbox_transform.h"

using namespace std;
using namespace tensorflow;

REGISTER_OP("AnchorTarget")
    .Attr("feat_stride: int")
    .Input("rpn_cls_score: float")
    .Input("gt_boxes: float")
    .Input("im_info: float")
    .Input("data: float")
    .Output("rpn_labels: float")
    .Output("rpn_bbox_targets: float")
    .Output("rpn_bbox_inside_weights: float")
    .Output("rpn_bbox_outside_weights: float")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape);

typedef Eigen::Tensor<float,1,Eigen::RowMajor> Tensor1f;
typedef Eigen::Tensor<float,2,Eigen::RowMajor> Tensor2f;
typedef Eigen::Tensor<float,3,Eigen::RowMajor> Tensor3f;
typedef Eigen::Tensor<float,4,Eigen::RowMajor> Tensor4f;
typedef Eigen::Tensor<int,1,Eigen::RowMajor> Tensor1i;
typedef Eigen::Tensor<int,4,Eigen::RowMajor> Tensor4i;

static Tensor2f operator +(const vector<vector<float> > &A)
{
  int i, j, m=A.size(), n=A[0].size();
  Tensor2f t(m, n);
  for(i=0; i<m; i++)
  {
    for(j=0; j<n; j++)
    {
      t(i,j) = A[i][j];
    }
  }
  return t;
}

static vector <vector<float> > operator +(const Tensor2f &A)
{
  int i,j,m=A.dimension(0),n=A.dimension(1);
  vector<vector <float>> t(m,vector<float> (n));
  for(i=0;i<m;i++)
  {
    for(j=0;j<n;j++)
    {
      t[i][j]=A(i,j);
    }
  }
  return t;
}

static Tensor1i operator+(const std::vector<int> &array)
{
  int i, n=array.size();
  Tensor1i t(n);
  for(i=0; i<n; i++)
  {
    t(i) = array[i];
  }
  return t;
}

class AnchorTargetOp : public OpKernel {
 public:
  explicit AnchorTargetOp(OpKernelConstruction* context) : OpKernel(context)
  {
    // Get the feat_stride
    OP_REQUIRES_OK(context, context->GetAttr("feat_stride", &feat_stride_));
    // Check that feat_stride is positive
    OP_REQUIRES(context, feat_stride_ >= 0, errors::InvalidArgument("Need feat_stride >= 0, got ", feat_stride_));
  }

  void Compute(OpKernelContext* context) override
  {
    // Grab the input tensor
    const Tensor& rpn_cls_score = context->input(0);
    const Tensor& gt_boxes = context->input(1);
    const Tensor& im_info = context->input(2);
    const Tensor& data = context->input(3);

    auto rpn_cls_score_flat = rpn_cls_score.flat<float>();
    auto gt_boxes_flat = gt_boxes.flat<float>();
    auto im_info_flat = im_info.flat<float>();
    auto data_flat = data.flat<float>();

    //std::cout << "im_info_flat = " << im_info_flat << std::endl;
    // data should have 4 dimensions.
    OP_REQUIRES(context, rpn_cls_score.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));
    // data should have 2 dimensions.
    OP_REQUIRES(context, gt_boxes.dims() == 2,
                errors::InvalidArgument("data must be 2-dimensional"));
    // data should have 2 dimensions.
    OP_REQUIRES(context, im_info.dims() == 2,
                errors::InvalidArgument("data must be 2-dimensional"));
    // rois should have 4 dimensions.
    OP_REQUIRES(context, data.dims() == 4,
                errors::InvalidArgument("rois must be 4-dimensional"));

/****************************************************************************/
//不推荐使用类，避免调用构造的麻烦
    anchors_gen genacs;
    Tensor2f anchors_ = +(genacs.generate_anchors());
    //Tensor2f anchors_ = generate_anchors();
    Tensor2f gt_box = gt_boxes.tensor<float,2>();
    int num_anchors = anchors_.dimension(0);

    int height = rpn_cls_score.dim_size(1);
    int width  = rpn_cls_score.dim_size(2);
    int allowed_border = 0;

//求出偏移量
    Tensor3f shifts_dims(height,width,4);
    for (int j = 0; j < height; j++){
      for(int i = 0; i < width; i++){
            for(int k = 0; k < 4; k++)
            switch(k)
            {
                case 0:
                {
                   shifts_dims(j,i,k) = i * feat_stride_;
                   break;
                }
                case 1:
                {
                   shifts_dims(j,i,k) = j * feat_stride_;
                   break;
                }
                case 2:
                {
                   shifts_dims(j,i,k) = i * feat_stride_;
                   break;
                }
                case 3:
                {
                   shifts_dims(j,i,k) = j * feat_stride_;
                   break;
                }
                default:
                   break;
            }
//        shifts_dims(j,i,4) = {i,j,i,j} * feat_stride_;
      }
    }
    Eigen::array<int, 3> three_dims({1,height*width,4});
    Tensor3f shifts = shifts_dims.reshape(three_dims);

    int A = num_anchors;
    int K = shifts.dimension(1);
//reshape成（1,A,4）
    Eigen::array<int, 3> three_dim_anchors({1,A,4});
    Tensor3f anchors_re = anchors_.reshape(three_dim_anchors);
//转置操作,shifts_tra(K,1,4)
    Tensor3f shifts_tra(K,1,4);
    for(int m = 0; m < K; m++){
        for(int n = 0; n < 4; n++){
            shifts_tra(m,0,n) = shifts(0,m,n);
        }
    }
/*******************************************************************/
//通过broadcast机制，将标准anchors与偏移量shifts相加，得到原图上的A×K个anchors的坐标
    Eigen::array<int, 3> aa({1,A,1});
    Eigen::array<int, 3> bb({K,1,1});
    auto cc = shifts_tra.broadcast(aa);
    auto dd = anchors_re.broadcast(bb);
    Tensor3f all_anchors = cc + dd;
    Eigen::array<int, 2> ee({K*A,4});
    Tensor2f all_anchors_re = all_anchors.reshape(ee);

/********************************************************************/
//通过边界参数进行初步筛选，这里参数为0,即不能超出图片边界
    int total_anchors = K*A;

    int in = 0;
    for(int a = 0; a < total_anchors; a++) {
        bool inside_expression = all_anchors_re(a,0) >= allowed_border
                              && all_anchors_re(a,1) >= allowed_border
                              && all_anchors_re(a,2) < im_info.tensor<float,2>()(0,1)
                              && all_anchors_re(a,3) < im_info.tensor<float,2>()(0,0);
        if(inside_expression) in++;
    }
    Tensor2f inside_anchors(in,4);
    vector<pair<int,int> > ins;
    int cnt = 0;
    for(int a = 0; a < total_anchors; a++) {
        bool inside_expression = all_anchors_re(a,0) >= allowed_border
                              && all_anchors_re(a,1) >= allowed_border
                              && all_anchors_re(a,2) < im_info.tensor<float,2>()(0,1)
                              && all_anchors_re(a,3) < im_info.tensor<float,2>()(0,0);
        if(inside_expression) {
            inside_anchors(cnt,0) = all_anchors_re(a,0);
            inside_anchors(cnt,1) = all_anchors_re(a,1);
            inside_anchors(cnt,2) = all_anchors_re(a,2);
            inside_anchors(cnt,3) = all_anchors_re(a,3);
            ins.push_back(make_pair(cnt,a));
            cnt++;
        }
    }

    //std::ofstream ofs;
    //ofs.open("test_tensor_inside_anchors.txt");
    //for(int i=0; i<in; i++)
    //{
    //    ofs << fixed << setprecision(4) << inside_anchors(i,0) << std::endl << inside_anchors(i,1) << std::endl << inside_anchors(i,2) << std::endl << inside_anchors(i,3) << std::endl;
    //}
    //ofs.close();

/************************************************************************/
//    labels = np.empty((len(inds_inside), ), dtype=np.float32)
//    labels.fill(-1)
//标记优先级：
//首先计算出每个anchor在所有gt_boxes上的LOU，选出最大值，当最大值小于LOU < 0.3，将当前anchor所在序号标签置为0;
//当最大LOU值 >= 0.7 将anchor所在序号标签置为1;此时可能会有多个anchors对应一个gt_boxes;
//选出每个gt_boxes在所有anchors的LOU最大值， 将anchor所在序号标签置为1;
//LOU在0.3-0.7之间的标签初始化为-1.
    Tensor1i labels(in);
    labels.setConstant(-1);
    vector<vector<float> > temp=+(inside_anchors);
    // vector<vector<float> > temp_get=+(gt_boxes.tensor<float,2>()); 
    Tensor2f overlaps_ = +(overlaps(temp, +(gt_boxes.tensor<float,2>())));
    // Eigen::array<int, 1> laps({1});  //row max
    // Eigen::array<int, 1> laps_c({0});
    // Eigen::Tensor<int, 1> max_overlaps = overlaps_.maximum(laps);
    // Eigen::Tensor<int, 1> gt_max_overlaps = overlaps_.maximum(laps_c);
    int num_gt = gt_boxes.dim_size(0);
    Tensor1f max_overlaps(in);
    max_overlaps.setZero();
    Tensor1f gt_max_overlaps(num_gt);
    gt_max_overlaps.setZero();
    Tensor1f argmax_overlaps(in);
    argmax_overlaps.setZero();
    std::vector<int> gt_argmax_overlaps;

    //std::cout << "overlaps_.shape() = " << overlaps_.dimension(0) << ", " << overlaps_.dimension(1) << std::endl; 

    //ofs.open("test_tensor_overlaps.txt");
    //ofs << fixed << setprecision(4) << overlaps_ << std::endl;
    //ofs.close();

    for (int r1 = 0; r1 < in; r1++)
    {
      vector<float> row1;
      vector<vector<float>::iterator> vi;
      for(int c1 = 0; c1 < num_gt; c1++)
      {
        row1.push_back(overlaps_(r1,c1));
      }
      for(vector<float>::iterator it = row1.begin(); it != row1.end(); it++)
      {
        vi.push_back(it);
      }
      sort(vi.begin(),vi.end(),[](vector<float>::iterator &a,vector<float>::iterator &b) {return *a > *b;});
      max_overlaps(r1) = **vi.begin();
      argmax_overlaps(r1) = (*vi.begin() - row1.begin());
      //row1.clear();
      //vi.clear();
      //当最大值小于LOU < 0.3，将anchors所在序号标签置为0，当最大LOU值 >= 0.7 将anchors所在序号标签置为1

      if(max_overlaps(r1) < fast_rcnn::cfg.TRAIN.RPN_NEGATIVE_OVERLAP)
        labels(r1) = 0;
      else if(max_overlaps(r1) >= fast_rcnn::cfg.TRAIN.RPN_POSITIVE_OVERLAP)
      {
        labels(r1) = 1;
        //std::cout << "positive index: " << r1 << std::endl;
      }
    }

    //ofs.open("test_tensor_max_overlaps.txt");
    //for(int i=0; i<in; i++)
    //{
    //    ofs << fixed << setprecision(4) << max_overlaps(i) << std::endl;
    //}
    //ofs.close();

    //ofs.open("test_tensor_argmax_overlaps.txt");
    //for(int i=0; i<in; i++)
    //{
    //    ofs << fixed << setprecision(4) << argmax_overlaps(i) << std::endl;
    //}
    //ofs.close();

    for(int c1 = 0; c1 < num_gt; c1++)
    {
      vector<float> col1;
      vector<vector<float>::iterator> vi_c;
      for(int r1 = 0; r1 < in; r1++)
      {
        col1.push_back(overlaps_(r1,c1));
      }
      for(vector<float>::iterator it = col1.begin(); it != col1.end(); it++)
      {
        vi_c.push_back(it);
      }
      sort(vi_c.begin(),vi_c.end(),[](vector<float>::iterator &a,vector<float>::iterator &b) {return *a > *b;});
      gt_max_overlaps(c1) = **vi_c.begin();
      for(int r2 = 0; r2 < in; r2++)
      {
        if(overlaps_(r2,c1)==gt_max_overlaps(c1))
        {
          gt_argmax_overlaps.push_back(r2);
          labels(r2) = 1;  //选出gt_boxes与所有anchors的LOU最大值，标记该anchors的索引
        }
      }
    }

    sort(gt_argmax_overlaps.begin(), gt_argmax_overlaps.end());
    //std::cout << "gt_argmax_overlaps = " << +gt_argmax_overlaps << std::endl;

    //ofs.open("test_tensor_labels.txt");
    //for(int i=0; i<in; i++)
    //{
    //    ofs << labels(i) << std::endl;
    //}
    //ofs.close();

/************************************************************************/
    int num_fg = fast_rcnn::cfg.TRAIN.RPN_BATCHSIZE * fast_rcnn::cfg.TRAIN.RPN_FG_FRACTION;
    int num_bg = 0;
    int count = 0;
    int count_p = 0;
//此时正向样本数
    vector<int> one_label;
    for (int n = 0; n < in; n++){
      if (labels(n) == 1){
        one_label.push_back(n);
      }
    }
    //std::cout << "one_label.size() = " << one_label.size() << std::endl;
//如果正向样本数大于128,随机抽取其差值个样本，将其置为-1
    if(one_label.size() > num_fg){
      srand((unsigned)time(0));
      for (int sub = 0; sub < (one_label.size()-num_fg); sub++){
        int t = rand()%one_label.size();   //此处修改为在0-one_label.size()处取值（labels为1的数组的下标），节省大量时间
        if(labels(one_label[t]) == 1)
          labels(one_label[t]) = -1;
        else
          sub--;
      }
    }
//此时正向样本数
    vector<int> one_indx;
    for (int n = 0; n < one_label.size(); n++){
      if (labels(one_label[n]) == 1){
        one_indx.push_back(one_label[n]);
      }
    }

//求出此时负样本数
    vector<int> zero_label;
    for (int n = 0; n < in; n++){
      if(labels(n) == 0)
        zero_label.push_back(n);
    }
    //std::cout << "zero_label.size() = " << zero_label.size() << std::endl;
//如果负样本数大于（256-正样本数），随机抽取其差值个样本，将其置为-1
    num_bg = fast_rcnn::cfg.TRAIN.RPN_BATCHSIZE - one_indx.size();
    if(zero_label.size() > num_bg){
      srand((unsigned)time(0));
      for (int sub = 0; sub < (zero_label.size() - num_bg); sub++){
        int t = rand()%zero_label.size();
        if(labels(zero_label[t]) == 0)
          labels(zero_label[t]) = -1;
        else
          sub--;
      }
    }
//求出此时负样本数
    vector<int> zero_indx;
    for (int n = 0; n < zero_label.size(); n++){
      if (labels(zero_label[n]) == 0){
        zero_indx.push_back(zero_label[n]);
      }
    }


/*********************************************************************/
//计算anchor和他对应的最大的LOU的gtbox之间的偏移量
    Eigen::array<int,2> off1({0,0});
    Eigen::array<int,2> ext1({num_gt,4});
    Tensor2f gt_box_f = gt_box.slice(off1,ext1);
    Tensor2f gt_bbox(in,4);
    for (int i = 0; i < in; i++){
        gt_bbox.chip(i,0) = gt_box_f.chip(argmax_overlaps(i),0);
    }  
    Tensor2f bbox_targets(in, 4);
    bbox_targets.setZero();
    bbox_targets = fast_rcnn::bbox_transform(inside_anchors, gt_bbox);//此处需将gt_boxes取 (N,4)
    Tensor2f bbox_inside_weights(in, 4);
    bbox_inside_weights.setZero();
    Tensor1f ones(4);
    ones.setConstant(1.0f);

    //ofs.open("test_tensor_bbox_targets.txt");
    //for(int i=0; i<in; i++)
    //{
    //    ofs << fixed << setprecision(4) << bbox_targets(i, 0) << std::endl << bbox_targets(i, 1) << std::endl << bbox_targets(i, 2) << std::endl << bbox_targets(i, 3) << std::endl;
    //}
    //ofs.close();

    for(int n = 0; n < one_indx.size(); n++)
        bbox_inside_weights.chip(one_indx[n],0) = ones;

    Tensor2f bbox_outside_weights(in, 4);
    bbox_outside_weights.setZero();
    Tensor1f positive_weights(4);
    positive_weights.setZero();
    Tensor1f negative_weights(4);
    positive_weights.setZero();

    float num_examples = one_indx.size() + zero_indx.size();
    float num_ones = one_indx.size();

    if(fast_rcnn::cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0){
      positive_weights = ones * 1.0f / num_examples;
      negative_weights = ones * 1.0f / num_examples;
    }
    else if((fast_rcnn::cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) && (fast_rcnn::cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1)){

      float f=fast_rcnn::cfg.TRAIN.RPN_POSITIVE_WEIGHT;

      positive_weights = ones * f / num_ones;
      negative_weights = ones * (1.0f-f) / (num_examples - num_ones);
    }

    for(int n = 0; n < one_indx.size(); n++)
        bbox_outside_weights.chip(one_indx[n],0) = positive_weights;
    for(int n = 0; n < zero_indx.size(); n++)
        bbox_outside_weights.chip(zero_indx[n],0) = negative_weights;

/***************************************************************************/
    Tensor1i labels_t(total_anchors);
    labels_t.setConstant(-1);
    Tensor2f bbox_target_t(total_anchors,4);
    bbox_target_t.setZero();
    Tensor2f bbox_inside_weights_t(total_anchors,4);
    bbox_inside_weights_t.setZero();
    Tensor2f bbox_outside_weights_t(total_anchors,4);
    bbox_outside_weights_t.setZero();
    for (int i = 0; i < ins.size(); i++){
      labels_t(ins[i].second) = labels(ins[i].first);
      bbox_target_t.chip(ins[i].second,0) = bbox_targets.chip(ins[i].first,0);
      bbox_inside_weights_t.chip(ins[i].second,0) = bbox_inside_weights.chip(ins[i].first,0);
      bbox_outside_weights_t.chip(ins[i].second,0) = bbox_outside_weights.chip(ins[i].first,0);
    }
//labels
    Eigen::array<int, 4> four_dims_f({1,height,width,A*4});
    Eigen::array<int, 4> four_dims({1,height,width,A});
    Tensor4i labels_r = labels_t.reshape(four_dims);
    Tensor4f bbox_target_r = bbox_target_t.reshape(four_dims_f);
    Tensor4f bbox_inside_weights_r = bbox_inside_weights_t.reshape(four_dims_f);
    Tensor4f bbox_outside_weights_r = bbox_outside_weights_t.reshape(four_dims_f);

    Tensor4f rpn_labels_r(1,A,height,width);
    Tensor4f rpn_bbox_targets(1,A*4,height,width) ;
    Tensor4f rpn_bbox_inside_weights(1,A*4,height,width) ;
    Tensor4f rpn_bbox_outside_weights(1,A*4,height,width) ;

    for(int i = 0; i < height; i++){
      for(int j = 0; j < width; j++){
        for(int k = 0; k < 4*A; k++){
            rpn_bbox_targets(0,k,i,j) = bbox_target_r(0,i,j,k);
            rpn_bbox_inside_weights(0,k,i,j) = bbox_inside_weights_r(0,i,j,k);
            rpn_bbox_outside_weights(0,k,i,j) = bbox_outside_weights_r(0,i,j,k);
        }
        for(int l = 0; l < A; l++){
            rpn_labels_r(0,l,i,j) = labels_r(0,i,j,l);
        }
      }
    }

    Eigen::array<int, 4> four_dims_l({1,1,A*height,width});
    Tensor4f rpn_labels = rpn_labels_r.reshape(four_dims_l);

    Tensor *outputs_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, {1,1,A*height,width}, &outputs_tensor));
    outputs_tensor->tensor<float, 4>() = rpn_labels;  
    OP_REQUIRES_OK(context, context->allocate_output(1, {1,A*4,height,width}, &outputs_tensor));
    outputs_tensor->tensor<float, 4>() = rpn_bbox_targets;     
    OP_REQUIRES_OK(context, context->allocate_output(2, {1,A*4,height,width}, &outputs_tensor));
    outputs_tensor->tensor<float, 4>() = rpn_bbox_inside_weights;
    OP_REQUIRES_OK(context, context->allocate_output(3, {1,A*4,height,width}, &outputs_tensor));
    outputs_tensor->tensor<float, 4>() = rpn_bbox_outside_weights;
    //std::cout << "------- AnchorTarget Op End ---------" << std::endl;

  }
private:
  int feat_stride_;
};

Status AnchorTargetGrad(const Scope &scope, const Operation &op, const std::vector<Output> &grad_inputs, std::vector<Output> *grad_outputs)
{
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}

REGISTER_KERNEL_BUILDER(Name("AnchorTarget").Device(DEVICE_CPU), AnchorTargetOp);
REGISTER_GRADIENT_OP("AnchorTarget", AnchorTargetGrad);
