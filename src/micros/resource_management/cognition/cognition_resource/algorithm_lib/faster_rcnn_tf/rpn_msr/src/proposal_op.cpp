//主要作用：proposal_layer就是将预测出的rpn_bbox_pred（框的偏移量）拿过来，经过一系列的操作，生成真正的proposals，
//形状是5列，注意这里是rpn的proposals，只有是否前景之分，没有对应的物体类别，这一层的用处是还原出真正的proposal信息
/*******************************************************************************************************/
#include <stdio.h>
#include <cfloat>
#include <functional>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
#include "generate_anchors.h"
#include <fast_rcnn/config.h>
#include <fast_rcnn/bbox_transform.h>
#include <nms/cpu_nms.h>

using namespace std;
using namespace tensorflow;

REGISTER_OP("Proposal")
    .Attr("cfg_key: int")    
    .Attr("feat_stride: int")
    .Input("rpn_cls_prob_reshape: float")
    .Input("rpn_bbox_pred: float")
    .Input("im_info: float")
    .Output("blob_tf: float")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape);

typedef Eigen::Tensor<float, 1, Eigen::RowMajor> Tensor1f;
typedef Eigen::Tensor<float, 2, Eigen::RowMajor> Tensor2f;
typedef Eigen::Tensor<float, 3, Eigen::RowMajor> Tensor3f;
typedef Eigen::Tensor<float, 4, Eigen::RowMajor> Tensor4f;
typedef Eigen::Tensor<int, 1, Eigen::RowMajor> Tensor1i;

template <typename Device>
class ProposalOp : public OpKernel {
 private:
  int feat_stride_, cfg_key;
 public:
  explicit ProposalOp(OpKernelConstruction* context) : OpKernel(context)
  {
    // Get the feat_stride 
    OP_REQUIRES_OK(context, context->GetAttr("feat_stride", &feat_stride_));
    // Check that feat_stride is non-negative
    OP_REQUIRES(context, feat_stride_ >= 0,
                errors::InvalidArgument("Need feat_stride >= 0, got ", feat_stride_));
    // Get the cfg_key
    OP_REQUIRES_OK(context, context->GetAttr("cfg_key", &cfg_key));
    // Check that cfg_key is either 0 or 1
    OP_REQUIRES(context, cfg_key == 0 || cfg_key == 1,
                errors::InvalidArgument("Need cfg_key == 0 || cfg_key == 1, got ", cfg_key));
  }

  void Compute(OpKernelContext* context) override
  {
    // Grab the input tensor
    const Tensor& rpn_cls_prob_reshape = context->input(0);//(1,h,w,18)
    //因为我们的预测值是anchors与gt的偏移量，这一层的值也就是detlas
    const Tensor& rpn_bbox_pred = context->input(1);//(1,h,w,36),
    const Tensor& im_info = context->input(2);//(1,3)
    // data should have 4 dimensions.
    OP_REQUIRES(context, rpn_cls_prob_reshape.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));
    // data should have 2 dimensions.
    OP_REQUIRES(context, rpn_bbox_pred.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));
    // data should have 2 dimensions.
    OP_REQUIRES(context, im_info.dims() == 2,
                errors::InvalidArgument("data must be 2-dimensional"));

/****************************************************************************/
    vector<vector<float> > text_anchor = anchors_gen().generate_anchors();  
    Tensor2f anchors_(text_anchor.size(),text_anchor[0].size());
    for(int i=0; i<text_anchor.size(); i++)
    {
        for(int j=0; j<text_anchor[0].size(); j++)
        {
            anchors_(i, j) = text_anchor[i][j];
        }
    }
/******************************************/    
    int num_anchors = anchors_.dimension(0);

    auto rpn_cls = rpn_cls_prob_reshape.tensor<float, 4>();
    auto bbox_deltas_t = rpn_bbox_pred.tensor<float, 4>();
    auto im_shape = im_info.tensor<float, 2>();
    int height = rpn_cls_prob_reshape.dim_size(1);
    int width  = rpn_cls_prob_reshape.dim_size(2);

    int pre_nms_topN  = (cfg_key == 1 ? fast_rcnn::cfg.TRAIN.RPN_PRE_NMS_TOP_N : fast_rcnn::cfg.TEST.RPN_PRE_NMS_TOP_N);
    int post_nms_topN = (cfg_key == 1 ? fast_rcnn::cfg.TRAIN.RPN_POST_NMS_TOP_N : fast_rcnn::cfg.TEST.RPN_POST_NMS_TOP_N);
    float nms_thresh  = (cfg_key == 1 ? fast_rcnn::cfg.TRAIN.RPN_NMS_THRESH : fast_rcnn::cfg.TEST.RPN_NMS_THRESH);
    int min_size      = (cfg_key == 1 ? fast_rcnn::cfg.TRAIN.RPN_MIN_SIZE : fast_rcnn::cfg.TEST.RPN_MIN_SIZE);
    // the first set of _num_anchors channels are bg probs
    // the second set are the fg probs, which we want
    Eigen::array<int, 4> offsets({0, 0, 0, num_anchors});
    Eigen::array<int, 4> extents({1, height, width, num_anchors});
    Tensor4f scores_t = rpn_cls.slice(offsets, extents); 
//*******************************************************************//
//和anchor_target_layer一样，也每个位置产生9个anchor，堆叠成anchors, (K×A, 4)， 
//遍历顺序是先遍历完一个位置的所有anchor，然后宽度遍历，最后高度遍历，这种遍历顺序记作(h,w,a)
    Tensor3f shifts_dims(height,width,4);
    for (int j = 0; j < height; j++) {
        for(int i = 0; i < width; i++) {
            shifts_dims(j,i,0) = i * feat_stride_;
            shifts_dims(j,i,1) = j * feat_stride_;
            shifts_dims(j,i,2) = i * feat_stride_;
            shifts_dims(j,i,3) = j * feat_stride_;
        }
    }
    Eigen::array<int, 3> three_dims({1,height*width,4});
    Tensor3f shifts = shifts_dims.reshape(three_dims);
    int A = num_anchors;
    int K = shifts.dimension(1);
//reshape成 (1,A,4)
    Eigen::array<int, 3> three_dim_anchors({1,A,4});
    Tensor3f anchors_re = anchors_.reshape(three_dim_anchors);
//转置操作,shifts_tra(K,1,4)
    Tensor3f shifts_tra = shifts.shuffle(Eigen::array<int, 3>{1,0,2});
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

    Eigen::array<int, 2> two_dims({K*A,4});
    Tensor2f bbox_deltas = bbox_deltas_t.reshape(two_dims);     
    Eigen::array<int, 1> score_dims({K*A});
    Tensor1f scores_r = scores_t.reshape(score_dims);
//Convert anchors (ctr_x,ctr_y,w,h) into proposals via bbox transformations
    Tensor2f proposals_b = fast_rcnn::bbox_transform_inv(all_anchors_re, bbox_deltas);    
//clip predicted boxes to image
    Tensor2f proposals_c = fast_rcnn::clip_boxes(proposals_b, im_shape.chip(0,0)); //应传shape，此处将tensor传入
//Remove all boxes with any side smaller than min_size.
    //std::cout << "image shape: " << std::endl << im_shape.chip(0,0) << std::endl;    
    float ws = 0;
    float hs = 0;
    float min_pro = min_size * im_shape(0,2);
    std::vector<int> indices;
    for(int i = 0; i < K*A; i++) {
        ws = proposals_c(i,2) - proposals_c(i,0) + 1.0f;
        hs = proposals_c(i,3) - proposals_c(i,1) + 1.0f;
        if(ws >= min_pro && hs >= min_pro) {
            indices.push_back(i);
        }
    }
    int index_size = indices.size();
    Tensor2f proposals(index_size,4);
    Tensor1f scores(index_size);
    for(int i = 0; i < index_size; i++) {
        proposals.chip(i,0) = proposals_c.chip(indices[i],0);
        scores(i) = scores_r(indices[i]);
    }

/*****************************************************************/
//order是将scores展开，并由大到小排序的标号，先截取分数最高的pre_nms_topN个框，比如12000个（注意如果少于这个数就是全部），
//然后proposals和scores都按照这个顺序将框排好。这个时候的框已经没有(h,w,a)的遍历顺序了
    Tensor2f proposals_last_p(pre_nms_topN,4);
    Tensor1f scores_last_p(pre_nms_topN);

    vector<float> s;
    vector<vector<float>::iterator> vi;
    int order = 0;
    int indx = 0;
    for(int i = 0; i < index_size; i++)
    {
        s.push_back(scores(i));
    }
    for(vector<float>::iterator it = s.begin(); it != s.end(); it++)
    {
        vi.push_back(it);
    }
    sort(vi.begin(), vi.end(), [](vector<float>::iterator &a,vector<float>::iterator &b) {return *a > *b;});
    for(vector<vector<float>::iterator>::iterator it = vi.begin(); it != vi.end(); it++) {
        indx = *it - s.begin();

        scores_last_p(order) = **it;
        proposals_last_p.chip(order,0) = proposals.chip(indx,0);

        order++;
        if(order >= pre_nms_topN) {
            break;
        }
    }
    Tensor2f proposals_last(order,4);
    Tensor1f scores_last(order);
    Eigen::array<int, 2> off0({0, 0});
    Eigen::array<int, 2> ext0({order, 4});     
    proposals_last = proposals_last_p.slice(off0, ext0);
    for(int i = 0; i < order; i++)
        scores_last(i) = scores_last_p(i);
/***********************************************************************/
//NMS的步骤就是对于分数由高到低排序的框，从分数高的开始，看他和后面每一个没有被扔掉的框的IoU是否大于阈值，
//是的话就将后面的这些框扔掉
    Tensor2f p_s(order,5); //形状由pre_nms_topN修改为order
    Eigen::array<int, 2> off1({0, 0});
    Eigen::array<int, 2> ext1({order, 4});  //形状由pre_nms_topN修改为order
    p_s.slice(off1, ext1) = proposals_last;
    p_s.chip(4,1) = scores_last;

    Tensor1i order_nms = cpu_nms(p_s, nms_thresh);
    //std::cout << "order_nms shape = " << order_nms.dimension(0) << std::endl;
    int len = (order_nms.dimension(0) > post_nms_topN) ? post_nms_topN : order_nms.dimension(0);

    Tensor2f proposals_nms(len,4);
    Tensor1f scores_nms(len);

    for (int i = 0; i < len; i++) {
        scores_nms(i) = scores_last(order_nms(i));
        proposals_nms.chip(i,0) = proposals_last.chip(order_nms(i),0);
    }

    Tensor1f batch_inds(len);
    batch_inds.setZero();

    Tensor2f blob(len, 5);
    Eigen::array<int, 2> off2({0, 1});
    Eigen::array<int, 2> ext2({len, 4});  
    blob.slice(off2, ext2) = proposals_nms;
    blob.chip(0,1) = batch_inds; //代表图片的index，我们只取一张图片，所以是0

    Tensor *blob_tf = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, {len, 5}, &blob_tf));
    blob_tf->tensor<float, 2>() = blob;
    //std::cout << "proposals_layer output: " << blob_tf->shape().DebugString() << endl;
  }
};

Status ProposalGrad(const Scope &scope, const Operation &op, const std::vector<Output> &grad_inputs, std::vector<Output> *grad_outputs)
{
  grad_outputs->push_back(NoGradient());
  return scope.status();
}

#if GOOGLE_CUDA
void ProposalKernelLauncher(OpKernelContext* context, int feat_stride_, int cfg_key, const Eigen::GpuDevice &d);
template <>
class ProposalOp<Eigen::GpuDevice> : public OpKernel {
 private:
  int feat_stride_, cfg_key;
 public:
  explicit ProposalOp(OpKernelConstruction* context) : OpKernel(context)
  {
    // Get the feat_stride
    OP_REQUIRES_OK(context, context->GetAttr("feat_stride", &feat_stride_));
    // Check that feat_stride is non-negative
    OP_REQUIRES(context, feat_stride_ >= 0,
                errors::InvalidArgument("Need feat_stride >= 0, got ", feat_stride_));
    // Get the cfg_key
    OP_REQUIRES_OK(context, context->GetAttr("cfg_key", &cfg_key));
    // Check that cfg_key is either 0 or 1
    OP_REQUIRES(context, cfg_key == 0 || cfg_key == 1,
                errors::InvalidArgument("Need cfg_key == 0 || cfg_key == 1, got ", cfg_key));
  }

  void Compute(OpKernelContext* context) override
  {
    ProposalKernelLauncher(context, feat_stride_, cfg_key, context->eigen_device<Eigen::GpuDevice>());
  }
};
#endif

REGISTER_KERNEL_BUILDER(Name("Proposal").Device(DEVICE_CPU), ProposalOp<Eigen::ThreadPoolDevice>);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Proposal").Device(DEVICE_GPU), ProposalOp<Eigen::GpuDevice>);
#endif
REGISTER_GRADIENT_OP("Proposal", ProposalGrad);
