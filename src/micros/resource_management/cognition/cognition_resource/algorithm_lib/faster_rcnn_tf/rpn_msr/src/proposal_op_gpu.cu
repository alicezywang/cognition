#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <iostream>
#include <cfloat>
#include <cuda.h>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"

#include "fast_rcnn/config.h"
#include "fast_rcnn/bbox_transform.h"
#include <nms/gpu_nms.h>

using namespace std;
using namespace tensorflow;

typedef Eigen::Tensor<float, 1, Eigen::RowMajor> Tensor1f;
typedef Eigen::Tensor<float, 2, Eigen::RowMajor> Tensor2f;
typedef Eigen::Tensor<float, 3, Eigen::RowMajor> Tensor3f;
typedef Eigen::Tensor<float, 4, Eigen::RowMajor> Tensor4f;
typedef Eigen::Tensor<int, 1, Eigen::RowMajor> Tensor1i;

class anchors_gen
{
public:
    anchors_gen();
    ~anchors_gen();
    vector<vector<float> > ratio_enum(vector<float>);
    vector<float> whctrs(vector<float>);
    vector<float> mkanchor(float w,float h,float x_ctr,float y_ctr);
    vector<vector<float> > scale_enum(vector<float>);
    vector<vector<float> > generate_anchors();
private:
    int base_size;
    float ratios[3];
    float scales[3];
};

__global__ void ProposalKernel(const float *boxes_data, const float *deltas_data, int A, int k_height, int k_width, float height, float width, float stride, float min_pro, float *pred_boxes, char *masks)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= k_height * k_width * A) return;
    float box_x1 = boxes_data[i%A*4] + ((i/A)%(k_width)) * stride;
    float box_y1 = boxes_data[i%A*4+1] + ((i/A)/(k_width)) * stride;
    float box_x2 = boxes_data[i%A*4+2] + ((i/A)%(k_width)) * stride;
    float box_y2 = boxes_data[i%A*4+3] + ((i/A)/(k_width)) * stride;
    float width_box  = box_x2 - box_x1 + 1.0f;
    float height_box = box_y2 - box_y1 + 1.0f;
    float ctr_x = box_x1 + width_box * 0.5f;
    float ctr_y = box_y1 + height_box * 0.5f;
    float pred_w = std::exp(deltas_data[i*4+2]) * width_box;
    float pred_h = std::exp(deltas_data[i*4+3]) * height_box;
    float pred_ctr_x = deltas_data[i*4] * width_box + ctr_x;
    float pred_ctr_y = deltas_data[i*4+1] * height_box + ctr_y;
    pred_boxes[i*4] = pred_ctr_x - pred_w * 0.5f;
    if(pred_boxes[i*4] > width - 1)
    {
        pred_boxes[i*4] = width - 1;
    }
    else if(pred_boxes[i*4] < 0)
    {
        pred_boxes[i*4] = 0;
    }
    pred_boxes[i*4+1] = pred_ctr_y - pred_h * 0.5f;
    if(pred_boxes[i*4+1] > height - 1)
    {
        pred_boxes[i*4+1] = height - 1;
    }
    else if(pred_boxes[i*4+1] < 0)
    {
        pred_boxes[i*4+1] = 0;
    }
    pred_boxes[i*4+2] = pred_ctr_x + pred_w * 0.5f;
    if(pred_boxes[i*4+2] > width - 1)
    {
        pred_boxes[i*4+2] = width - 1;
    }
    else if(pred_boxes[i*4+2] < 0)
    {
        pred_boxes[i*4+2] = 0;
    }
    pred_boxes[i*4+3] = pred_ctr_y + pred_h * 0.5f;
    if(pred_boxes[i*4+3] > height - 1)
    {
        pred_boxes[i*4+3] = height - 1;
    }
    else if(pred_boxes[i*4+3] < 0)
    {
        pred_boxes[i*4+3] = 0;
    }
    float ws = pred_boxes[i*4+2] - pred_boxes[i*4] + 1.0f;
    float hs = pred_boxes[i*4+3] - pred_boxes[i*4+1] + 1.0f;
    if(ws >= min_pro && hs >= min_pro)
    {
        masks[i] = 1;
    }
    else
    {
        masks[i] = 0;
    }
}

void ProposalKernelLauncher(OpKernelContext* context, int feat_stride_, int cfg_key, const Eigen::GpuDevice &d)
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

    vector<vector<float> > text_anchor = anchors_gen().generate_anchors();
    const int num_anchors = text_anchor.size();

    const int height = rpn_cls_prob_reshape.dim_size(1);
    const int width  = rpn_cls_prob_reshape.dim_size(2);
    Tensor4f rpn_cls(1, height, width, 18);
    cudaMemcpy(rpn_cls.data(), rpn_cls_prob_reshape.flat<float>().data(), height * width * 18 * sizeof(float), cudaMemcpyHostToHost);
    Tensor4f bbox_deltas_t(1, height, width, 36);
    cudaMemcpy(bbox_deltas_t.data(), rpn_bbox_pred.flat<float>().data(), height * width * 36 * sizeof(float), cudaMemcpyHostToHost);
    Tensor2f im_shape(1,3);
    cudaMemcpy(im_shape.data(), im_info.flat<float>().data(), 3 * sizeof(float), cudaMemcpyHostToHost);

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
    const int A = num_anchors;
    const int K = height * width;

/********************************************************************/

    Eigen::array<int, 1> score_dims({K * A});
    Tensor1f scores_r = scores_t.reshape(score_dims);
    Tensor2f proposals_b(K * A, 4);
    std::vector<char> mask_vector(K * A);
    const int threads_per_block = d.maxCudaThreadsPerBlock();
    float *boxes_data = (float*)d.allocate(K * 4 * sizeof(float));
    float *deltas_data = (float*)d.allocate(K * A * 4 * sizeof(float));
    float *output = (float*)d.allocate(K * A * 4 * sizeof(float));
    char *masks = (char*)d.allocate(K * A);
    for(int i = 0; i < text_anchor.size(); i++)
    {
        d.memcpyHostToDevice(boxes_data + i * text_anchor[i].size(), text_anchor[i].data(), text_anchor[i].size() * sizeof(float));
    }
    d.memcpyHostToDevice(deltas_data, bbox_deltas_t.data(), K * A * 4 * sizeof(float));
    ProposalKernel<<<(K * A + threads_per_block - 1) / threads_per_block, threads_per_block, 0, d.stream()>>>(boxes_data, deltas_data, A, height, width, im_shape(0,0), im_shape(0,1), feat_stride_, min_size * im_shape(0,2), output, masks);
    d.memcpyDeviceToHost(proposals_b.data(), output, K * A * 4 * sizeof(float));
    d.memcpyDeviceToHost(mask_vector.data(), masks, K * A);
    d.synchronize();
    d.deallocate(boxes_data);
    d.deallocate(deltas_data);
    d.deallocate(output);
    d.deallocate(masks);
//Remove all boxes with any side smaller than min_size.
    std::vector<int> indices;
    for(int i = 0; i < K * A; i++)
    {
        if(mask_vector[i] == 1) indices.push_back(i);
    }
    int index_size = indices.size();
    Tensor2f proposals(index_size, 4);
    Tensor1f scores(index_size);
    for(int i = 0; i < index_size; i++)
    {
        proposals.chip(i,0) = proposals_b.chip(indices[i], 0);
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

    Tensor1i order_nms = gpu_nms(p_s, nms_thresh);
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
    cudaMemcpy(blob_tf->flat<float>().data(), blob.data(), len * 5 * sizeof(float), cudaMemcpyHostToHost);
    //std::cout << "proposals_layer output: " << blob_tf->shape().DebugString() << endl;
}

#endif
