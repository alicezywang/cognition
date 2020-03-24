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
#include "generate_anchors.h"
#include "bbox.h"
#include "fast_rcnn/config.h"
#include "fast_rcnn/bbox_transform.h"


using namespace std;
using namespace tensorflow;

typedef Eigen::Tensor<float,1,Eigen::RowMajor> Tensor1f;
typedef Eigen::Tensor<float,2,Eigen::RowMajor> Tensor2f;
typedef Eigen::Tensor<float,3,Eigen::RowMajor> Tensor3f;
typedef Eigen::Tensor<float,4,Eigen::RowMajor> Tensor4f;
typedef Eigen::Tensor<int,1,Eigen::RowMajor> Tensor1i;
typedef Eigen::Tensor<int,2,Eigen::RowMajor> Tensor2i;
typedef Eigen::Tensor<int,3,Eigen::RowMajor> Tensor3i;
typedef Eigen::Tensor<int,4,Eigen::RowMajor> Tensor4i;

static Tensor2f operator +(const vector<vector<float> > &A)
{
  int i, j, m=A.size(), n=A[0].size();
  std::cout<<"m"<<m<<"n"<<n<<endl;
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
vector <vector<float> > operator +(const Tensor2f &A)
{
  int i,j,m=A.dimension(0),n=A.dimension(1);
  std::cout<<"m"<<m<<"n"<<n<<endl;
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

int main()
{
  Tensor gt_boxes(DT_FLOAT,TensorShape({5,5}));
  Tensor rpn_cls_score(DT_FLOAT,TensorShape({2,3,4,5}));
  Tensor im_info(DT_FLOAT,TensorShape({9,7}));
  Tensor data(DT_FLOAT,TensorShape({3,2,4,2}));

  std::cout<<"Run ok"<<std::endl;

  anchors_gen genacs;
  std::cout<<"Run ok 2"<<std::endl;
  Tensor2f anchors_ = +(genacs.generate_anchors());
  std::cout<<"Run ok 3"<<std::endl;
  Tensor2f gt_box = gt_boxes.tensor<float,2>();
  std::cout<<"Run ok 4"<<std::endl;
  int num_anchors = anchors_.dimension(0);
  std::cout<<"anchors_.dimension(0)"<<num_anchors<<std::endl;

  std::cout<<"Run ok 3"<<std::endl;

  int height = rpn_cls_score.dim_size(1);
  int width  = rpn_cls_score.dim_size(2);
  std::cout<<"dim_size(1):"<<height<<std::endl;
  std::cout<<"dim_size(2):"<<width<<std::endl;
  int allowed_border = 0;
  int feat_stride_=1;

  Tensor3f shifts_dims(height,width,4);
  for (int j = 0; j < height; j++){
    for(int i = 0; i < width; i++){
      for(int k = 0; k < 4; k++)
        switch(k)
        {
        case 0:
        {
          shifts_dims(j,i,k) = i * feat_stride_;
          //std::cout<<"shifts_dims(j,i,k):"<<shifts_dims(j,i,k)<<std::endl;
          break;
        }
        case 1:
        {
          shifts_dims(j,i,k) = j * feat_stride_;
          //std::cout<<"shifts_dims(j,i,k):"<<shifts_dims(j,i,k)<<std::endl;
          break;
        }
        case 2:
        {
          shifts_dims(j,i,k) = i * feat_stride_;
          //std::cout<<"shifts_dims(j,i,k):"<<shifts_dims(j,i,k)<<std::endl;
          break;
        }
        case 3:
        {
          shifts_dims(j,i,k) = j * feat_stride_;
          //std::cout<<"shifts_dims(j,i,k):"<<shifts_dims(j,i,k)<<std::endl;
          break;
        }
        default:
          break;
        }
    }
  }
  ///test 03
  Eigen::array<float, 3> three_dims({1,height*width,4});
  Tensor3f shifts = shifts_dims.reshape(three_dims);

  int A = num_anchors;
  int K = shifts.dimension(1);
    std::cout<<"k:"<<K<<std::endl;
    std::cout<<"A:"<<A<<std::endl;
  //reshape成（1,A,4）
  Eigen::array<float, 3> three_dim_anchors({1,A,4});
  Tensor3f anchors_re = anchors_.reshape(three_dim_anchors);
  //转置操作,shifts_tra(K,1,4)
  Tensor3f shifts_tra(K,1,4);
  for(int m = 0; m < K; m++){
    for(int n = 0; n < 4; n++){
      shifts_tra(m,0,n) = shifts(0,m,n);
      std::cout<<"shifts_tra(m,0,n):"<<shifts_tra(m,0,n)<<std::endl;
    }
  }
  //通过broadcast机制，将标准anchors与偏移量shifts相加，得到原图上的A×K个anchors的坐标
  Eigen::array<int, 3> aa({1,A,1});
  Eigen::array<int, 3> bb({K,1,1});
  auto cc = shifts_tra.broadcast(aa);
  auto dd = anchors_re.broadcast(bb);
  Tensor3f all_anchors = cc + dd;
  Eigen::array<int, 2> ee({K*A,4});
  Tensor2f all_anchors_re = all_anchors.reshape(ee);

  //通过边界参数进行初步筛选，这里参数为0,即不能超出图片边界
  int total_anchors = K*A;

  int in = 0;
  for(int a = 0; a < total_anchors; a++) {
    std::cout<<"all_anchors_re(a,0):"<<all_anchors_re(a,0)<<std::endl
            <<"all_anchors_re(a,1)"<<all_anchors_re(a,1)<<std::endl
           <<"all_anchors_re(a,2)"<<all_anchors_re(a,2)<<std::endl
          <<"all_anchors_re(a,3)"<<all_anchors_re(a,3)<<std::endl
         <<std::endl;


    bool inside_expression = all_anchors_re(a,0) >= allowed_border
        && all_anchors_re(a,1) >= allowed_border
        && all_anchors_re(a,2) < im_info.tensor<float,2>()(0,1)
        && all_anchors_re(a,3) < im_info.tensor<float,2>()(0,0);
    if(inside_expression) in++;
  }
  //default in=10 test
    in =10;
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
  std::cout<<"test3 ok "<<std::endl;
  ///test 04
  ///
  ///
  Tensor1i labels(in);
  labels.setConstant(-1);
  vector<vector<float> > temp=+(inside_anchors);
  vector<vector<float> > temp_gt=+(gt_boxes.tensor<float,2>());

//  vector<vector<float> > temp={{1,1},{1,1}};
//  vector<vector<float> > temp_gt={{1,1},{1,1}};
     std::cout<<"--------here1--------"<<std::endl;
  Tensor2f overlaps_ = +(overlaps(temp,temp_gt));
     std::cout<<"--------here overlaps--------"<<std::endl;
  int num_gt = gt_boxes.dim_size(0);
  Tensor1f max_overlaps(in);
  max_overlaps.setZero();
  Tensor1f gt_max_overlaps(num_gt);
  gt_max_overlaps.setZero();
  Tensor1f argmax_overlaps(in);
  argmax_overlaps.setZero();
  Tensor1f gt_argmax_overlaps(num_gt);
  gt_argmax_overlaps.setZero();

  std::cout<<"--------here set zaro--------"<<std::endl;
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
    //  row1.clear();
    //  vi.clear();
    //当最大值小于LOU < 0.3，将anchors所在序号标签置为0，当最大LOU值 >= 0.7 将anchors所在序号标签置为1

    //fast_rcnn::train train;

    if(max_overlaps(r1) < fast_rcnn::cfg.TRAIN.RPN_NEGATIVE_OVERLAP)
      labels(r1) = 0;
    else if(max_overlaps(r1) >= fast_rcnn::cfg.TRAIN.RPN_POSITIVE_OVERLAP)
      labels(r1) = 1;
  }
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
    gt_argmax_overlaps(c1) = (*vi_c.begin() - col1.begin());
    labels((int)gt_argmax_overlaps(c1)) = 1;  //选出gt_boxes与所有anchors的LOU最大值，标记该anchors的索引
  }

  /************************************************************************/
  int num_fg = fast_rcnn::cfg.TRAIN.RPN_BATCHSIZE * fast_rcnn::cfg.TRAIN.RPN_FG_FRACTION;
  int num_bg = 0;
  int count = 0;
  int count_p = 0;
  int count_n = 0;
  //此时正向样本数
  vector<int> one_label;
  for (int n = 0; n < in; n++){
    if (labels(n) == 1){
      one_label.push_back(n);
    }
  }
  //如果正向样本数大于128,随机抽取其差值个样本，将其置为-1
  if(one_label.size() > num_fg){
    srand((unsigned)time(0));
    for (int sub = 0; sub < (one_label.size()-num_fg); sub++){
      int t;
      t= rand()%one_label.size();   //此处修改为在0-one_label.size()处取值（labels为1的数组的下标），节省大量时间
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
  //如果负样本数大于（256-正样本数），随机抽取其差值个样本，将其置为-1
  num_bg = fast_rcnn::cfg.TRAIN.RPN_BATCHSIZE - one_indx.size();
  if(count_n > num_bg){
    srand((unsigned)time(0));
    for (int sub = 0; sub < (zero_label.size() - num_bg); sub++){
      int t;
      t = rand()%zero_label.size();
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
    std::cout<<"--------zero_indx size:"<<zero_indx.size()<<"--------"<<std::endl;

  /*********************************************************************/
  //计算anchor和他对应的最大的LOU的gtbox之间的偏移量
  Eigen::array<int,2> off1({0,0});
  Eigen::array<int,2> ext1({num_gt,4});
  Tensor2f gt_bbox = gt_box.slice(off1,ext1);
  Tensor2f bbox_targets(in, 4);
  bbox_targets.setZero();
  std::cout<<"--------bbox_trasform --------"<<std::endl;
  //bbox_targets = fast_rcnn::bbox_transform(inside_anchors, gt_bbox);//此处需将gt_boxes取（N,4）
  Tensor2f bbox_inside_weights(in, 4);
  bbox_inside_weights.setZero();
  Tensor1f onesf(4);
    std::cout<<"----------------------------"<<std::endl;
  // onesf.setValues(const typename internal::Initializer<Derived, NumDimensions>::InitList& vals)
  onesf.setValues({fast_rcnn::cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0],fast_rcnn::cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[1],\
                   fast_rcnn::cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[2],fast_rcnn::cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[3]});

  for(int n = 0; n < one_indx.size(); n++)
    bbox_inside_weights.chip(one_indx[n],0) = onesf;

  Tensor2f bbox_outside_weights(in, 4);
  bbox_outside_weights.setZero();
  Tensor1f ones(4);
  ones.setConstant(1.0f);
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
  else if((fast_rcnn::cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) & (fast_rcnn::cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1)){

    float f=fast_rcnn::cfg.TRAIN.RPN_POSITIVE_WEIGHT;

    positive_weights = ones * f / num_ones;
    negative_weights = ones * f / (num_examples - num_ones);
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
  for (int i = 0; i < ins.size(); in++){
    labels_t(ins[i].second) = labels(ins[i].first);
    bbox_target_t.chip(ins[i].second,0) = bbox_targets.chip(ins[i].first,0);
    bbox_inside_weights_t.chip(ins[i].second,0) = bbox_inside_weights.chip(ins[i].first,0);
    bbox_outside_weights_t.chip(ins[i].second,0) = bbox_outside_weights.chip(ins[i].first,0);
  }

  //labels
  Eigen::array<float, 4> four_dims_f({1,height,width,A*4});
  Eigen::array<int, 4> four_dims({1,height,width,A});
  Tensor4i labels_r = labels_t.reshape(four_dims);
  Tensor4f bbox_target_r = bbox_target_t.reshape(four_dims_f);
  Tensor4f bbox_inside_weights_r = bbox_inside_weights_t.reshape(four_dims_f);
  Tensor4f bbox_outside_weights_r = bbox_outside_weights_t.reshape(four_dims_f);

  Tensor4i rpn_labels_r(1,A,height,width);
  Tensor4f rpn_bbox_targets(1,A*4,height,width) ;
  Tensor4f rpn_bbox_inside_weights(1,A*4,height,width) ;
  Tensor4f rpn_bbox_outside_weights(1,A*4,height,width) ;

  std::cout<<"--------labels ok--------"<<std::endl;
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      for(int k = 0; k < A; k++){
        rpn_labels_r(0,i,j,k) = labels_r(0,k,i,j);
        rpn_bbox_targets(0,i,j,k) = bbox_target_r(0,k,i,j);
        rpn_bbox_inside_weights(0,i,j,k) = bbox_inside_weights_r(0,k,i,j);
        rpn_bbox_outside_weights(0,i,j,k) = bbox_outside_weights_r(0,k,i,j);
      }
    }
  }
  Eigen::array<int, 4> four_dims_l({1,1,A*height,width});
  Tensor4i rpn_labels = rpn_labels_r.reshape(four_dims_l);

  return 0;
}
