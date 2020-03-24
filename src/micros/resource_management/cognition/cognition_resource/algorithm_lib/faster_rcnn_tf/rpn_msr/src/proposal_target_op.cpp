#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <ctime>
#include <functional>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
#include "fast_rcnn/config.h"
#include "fast_rcnn/bbox_transform.h"
#include "bbox.h"

using namespace tensorflow;

REGISTER_OP("ProposalTarget")
    .Attr("num_classes: int")    
    .Input("rpn_rois: float")
    .Input("gt_boxes: float")
    .Output("rois: float")
    .Output("labels: float")
    .Output("bbox_targets: float")
    .Output("bbox_inside_weights: float")
    .Output("bbox_outside_weights: float")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape);

static Eigen::Tensor<float, 2, Eigen::RowMajor> operator +(const std::vector<std::vector<float> > &A)
{
  int i, j, m=A.size(), n=A[0].size();
  Eigen::Tensor<float, 2, Eigen::RowMajor> t(m, n);
  for(i=0; i<m; i++)
  {
    for(j=0; j<n; j++)
    {
      t(i,j) = A[i][j];
    }
  }
  return t;
}

static std::vector<std::vector<float> > operator +(const Eigen::Tensor<float, 2, Eigen::RowMajor> &A)
{
  int i, j, m=A.dimension(0), n=A.dimension(1);
  std::vector<std::vector<float> > t(m, std::vector<float>(n));
  for(i=0; i<m; i++)
  {
    for(j=0; j<n; j++)
    {
      t[i][j] = A(i, j);
    }
  }
  return t;
}

class ProposalTargetOp : public OpKernel {
public:
  explicit ProposalTargetOp(OpKernelConstruction* context) : OpKernel(context)
  {
    // Get the feat_stride
    OP_REQUIRES_OK(context, context->GetAttr("num_classes", &num_classes));
    // Check that feat_stride is positive
    OP_REQUIRES(context, num_classes >= 0, errors::InvalidArgument("Need feat_stride >= 0, got ", num_classes));
  }

  void Compute(OpKernelContext* context) override
  {
    // Grab the input tensor
    const Tensor &rpn_rois = context->input(0); //(1,h,w,18)
    const Tensor &gt_boxes = context->input(1);

    typedef Eigen::Tensor<float,1,Eigen::RowMajor> Tensor1f;
    typedef Eigen::Tensor<float,2,Eigen::RowMajor> Tensor2f;
    typedef Eigen::Tensor<float,3,Eigen::RowMajor> Tensor3f;
    typedef Eigen::Tensor<float,4,Eigen::RowMajor> Tensor4f;
    typedef Eigen::Tensor<float,5,Eigen::RowMajor> Tensor5f;    
    typedef Eigen::Tensor<int,1,Eigen::RowMajor> Tensor1i;
    typedef Eigen::Tensor<int,2,Eigen::RowMajor> Tensor2i;
    typedef Eigen::Tensor<int,3,Eigen::RowMajor> Tensor3i;
    typedef Eigen::Tensor<int,4,Eigen::RowMajor> Tensor4i;

    // data should have 2 dimensions.
    OP_REQUIRES(context, rpn_rois.dims() == 2, errors::InvalidArgument("data must be 4-dimensional"));
    // data should have 2 dimensions.
    OP_REQUIRES(context, gt_boxes.dims() == 2, errors::InvalidArgument("data must be 2-dimensional"));
        
/****************************************************************************/
    //Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    Tensor2f all_r = rpn_rois.tensor<float,2>();
    Tensor2f gt_box = gt_boxes.tensor<float,2>();
    int num_gt = gt_boxes.dim_size(0);
    int num_roi = rpn_rois.dim_size(0);
    Tensor2f Zeros(num_gt,1);
    Zeros.setZero();
    Eigen::array<int,2> off0({0,0});
    Eigen::array<int,2> ext0({num_gt,4});
    Tensor2f gt_b = gt_box.slice(off0,ext0);
    Tensor2f gt_zero(num_gt,5);
    Tensor2f all_rois(num_gt + num_roi,5);  

    Eigen::array<int,2> off1({0,0});
    Eigen::array<int,2> ext1({num_gt,1});  
    Eigen::array<int,2> off2({0,1});
    Eigen::array<int,2> ext2({num_gt,4});
    gt_zero.slice(off1,ext1) = Zeros;
    gt_zero.slice(off2,ext2) = gt_b;     

    Eigen::array<int,2> off3({0,0});
    Eigen::array<int,2> ext3({num_roi,5});  
    Eigen::array<int,2> off4({num_roi,0});
    Eigen::array<int,2> ext4({num_gt,5});  
    all_rois.slice(off3,ext3) = all_r;
    all_rois.slice(off4,ext4) = gt_zero; 
    Eigen::array<int,2> off5({0,1});
    Eigen::array<int,2> ext5({num_gt + num_roi,4}); 
    Tensor2f noz_rois = all_rois.slice(off5,ext5);

    int num_images = 1;
    int rois_per_image = fast_rcnn::cfg.TRAIN.BATCH_SIZE / num_images;
    int fg_rois_per_image = round(fast_rcnn::cfg.TRAIN.FG_FRACTION * rois_per_image);
    int num_rois = num_gt + num_roi;

    Tensor1f max_overlaps(num_rois);
    max_overlaps.setZero();
    Tensor1f argmax_overlaps(num_rois);
    argmax_overlaps.setZero();
    Tensor1f labels(num_rois);

    Tensor2f overlaps_matrix(num_rois,gt_box.dimension(0));
    overlaps_matrix = +overlaps(+noz_rois,+gt_b); //gt_box -> gt_b

    for (int r1 = 0; r1 < num_rois; r1++){
      std::vector<float> row1;
      std::vector<std::vector<float>::iterator> vi; 
      for(int c1 = 0; c1 < num_gt; c1++)
      {
        row1.push_back(overlaps_matrix(r1,c1));
      }
      for(std::vector<float>::iterator it = row1.begin(); it != row1.end(); it++)
      {
        vi.push_back(it);
      }
      sort(vi.begin(), vi.end(), [](std::vector<float>::iterator &a, std::vector<float>::iterator &b) {return *a > *b;});
      max_overlaps(r1) = **vi.begin();
      argmax_overlaps(r1) = (*vi.begin() - row1.begin());
      labels(r1) = gt_box(argmax_overlaps(r1),4);
    }

/********************************************************************************/

    std::vector<int> fg_index, fg_inds;
    std::vector<bool> fg_selected;
    for(int i = 0; i < num_rois; i++){
        if(max_overlaps(i) >= fast_rcnn::cfg.TRAIN.FG_THRESH){
            fg_index.push_back(i);
            fg_selected.push_back(true);
        }
    }
    //std::cout << "fg_rois_per_image = " << fg_rois_per_image << std::endl;
    if(fg_index.size() > fg_rois_per_image){
        srand((unsigned)time(0));
        for (int n = 0; n < (fg_index.size()-fg_rois_per_image); n++){
            int t = rand()%(int)fg_index.size();
            if(fg_selected[t] == true)
                fg_selected[t] = false;
            else
                n--;
        }
    }
    for(int n = 0; n < fg_index.size(); n++){
        if(fg_selected[n] == true){
            fg_inds.push_back(fg_index[n]);
        }
    }

/*************************************************************************************/

    std::vector<int> bg_index, bg_inds;
    std::vector<bool> bg_selected;
    for(int i = 0; i < num_rois; i++){
        if((max_overlaps(i) < fast_rcnn::cfg.TRAIN.BG_THRESH_HI) && (max_overlaps(i) >= fast_rcnn::cfg.TRAIN.BG_THRESH_LO)){
            bg_index.push_back(i);
            bg_selected.push_back(true);
        }
    }
    int bg_rois_per_image = rois_per_image - fg_inds.size();
    //std::cout << "bg_rois_per_image = " << bg_rois_per_image << std::endl;
    if(bg_index.size() > bg_rois_per_image){
        srand((unsigned)time(0));
        for (int n = 0; n < (bg_index.size()-bg_rois_per_image); n++){
            int t = rand()%(int)bg_index.size();
            if(bg_selected[t] == true)
                bg_selected[t] = false;
            else
                n--;
        }
    }
    for(int n = 0; n < bg_index.size(); n++){
        if(bg_selected[n] == true){
            bg_inds.push_back(bg_index[n]);
        }
    }

/**************************************************************************************/

    int keep_inds = (int)fg_inds.size()+(int)bg_inds.size();
    Tensor1f labels_last(keep_inds);
    Tensor2f rois(keep_inds,5);
    Tensor1f gt_assignment(keep_inds);
    Tensor2f gt_box_transform(keep_inds,5);
    for (int i = 0; i < (int)fg_inds.size(); i++) {
        labels_last(i) = labels(fg_inds[i]);
        rois.chip(i,0) = all_rois.chip(fg_inds[i],0);
        gt_assignment(i) = argmax_overlaps(fg_inds[i]);
        gt_box_transform.chip(i,0) = gt_box.chip(argmax_overlaps(fg_inds[i]),0);
    }
    for (int i = 0; i < (int)bg_inds.size(); i++) {
        labels_last((int)fg_inds.size()+i) = 0;//Clamp labels for the background RoIs to 0
        rois.chip((int)fg_inds.size()+i,0) = all_rois.chip(bg_inds[i],0);
        gt_assignment((int)fg_inds.size()+i) = argmax_overlaps(bg_inds[i]);
        gt_box_transform.chip((int)fg_inds.size()+i,0) = gt_box.chip(argmax_overlaps(bg_inds[i]),0);
    }

/******************************************************************************************/

    Eigen::array<int,2> off6({0,1});
    Eigen::array<int,2> ext6({keep_inds,4}); 
    Tensor2f rois_f = rois.slice(off6,ext6);
    Eigen::array<int,2> off7({0,0});
    Eigen::array<int,2> ext7({keep_inds,4}); 
    Tensor2f gt_f = gt_box_transform.slice(off7,ext7);   

    Tensor2f targets = fast_rcnn::bbox_transform(rois_f,gt_f);
    int target_columns = targets.dimension(1);
    for(int i = 0; i < target_columns; i++)
    {
        targets.chip(i, 1) = targets.chip(i, 1) / fast_rcnn::cfg.TRAIN.BBOX_NORMALIZE_STDS[i];
    }

    // Eigen::array<float,2> two_dims({keep_inds,1});
    // Tensor2f labels_c = labels_last.reshape(two_dims);
    // Tensor2f bbox_target_data(keep_inds,5);
    // bbox_target_data.chip(0,1) = labels_last;
    // bbox_target_data.slice(off6,ext6) = Targets;

    Tensor2f bbox_targets(keep_inds, 4 * num_classes);
    bbox_targets.setZero();
    Tensor2f bbox_inside_weights(keep_inds, 4 * num_classes);
    bbox_inside_weights.setZero();
    for(int i = 0; i < keep_inds; i++)
    {
        if(labels_last(i) > 0)
        {
            for(int j = 0; j < 4; j++)
            {
                bbox_targets(i, 4 * labels_last(i) + j) = targets(i,j);
                bbox_inside_weights(i, 4 * labels_last(i) + j) = 1.0f;
            }
        }
    }

    Eigen::array<int, 2> two_dim_label({keep_inds,1});
    Tensor2f labels_f = labels_last.reshape(two_dim_label);

    Tensor *output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, {keep_inds, 5}, &output_tensor));
    output_tensor->tensor<float, 2>() = rois;
    OP_REQUIRES_OK(context, context->allocate_output(1, {keep_inds, 1}, &output_tensor));
    output_tensor->tensor<float, 2>() = labels_f;
    OP_REQUIRES_OK(context, context->allocate_output(2, {keep_inds, num_classes*4}, &output_tensor));
    output_tensor->tensor<float, 2>() = bbox_targets;
    OP_REQUIRES_OK(context, context->allocate_output(3, {keep_inds, num_classes*4}, &output_tensor));
    output_tensor->tensor<float, 2>() = bbox_inside_weights;
    OP_REQUIRES_OK(context, context->allocate_output(4, {keep_inds, num_classes*4}, &output_tensor));
    output_tensor->tensor<float, 2>() = bbox_inside_weights;
  }
private:
  int num_classes;
};

Status ProposalTargetGrad(const Scope &scope, const Operation &op, const std::vector<Output> &grad_inputs, std::vector<Output> *grad_outputs)
{
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}

REGISTER_KERNEL_BUILDER(Name("ProposalTarget").Device(DEVICE_CPU), ProposalTargetOp);
REGISTER_GRADIENT_OP("ProposalTarget", ProposalTargetGrad);

//返回值如下：
//rois：128×5，第一列是全0，后面是框的左上角右下角坐标；

//labels_last: 128×1，每个框的物体类别；

//bbox_targets: 128×84，每个框回归的偏差值，经过了normalize

//bbox_inside_weights: 128×84对应类别位置为1.

//这一层就是将proposal_layer提供的roi加上物体类别标签和bbox的回归目标，并计算权重weights。
//注意上面的anchor_target_layer加上的标签和回归目标用于rpn训练，这里的用于目标检测训练
