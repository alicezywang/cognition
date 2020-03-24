#include "fast_rcnn/train.h"
#include "utils/myTimer.h"
#include "time_op.h"

#define TIMER_DEBUG

tensorflow::SessionOptions session_options;

namespace fast_rcnn {

static Tensor operator+(const std::vector<float> &array)
{
  int i, n = array.size();
  Tensor result(DT_FLOAT, {n});
  for(i=0; i<n; i++) result.vec<float>()(i) = array[i];
  return result;
}

SolverWrapper::SolverWrapper(const Scope &scope, string output_dir): scope(scope), session(scope, session_options), net(scope), output_dir(output_dir), data_layer(rdl::get_data_layer(bbox_means, bbox_stds))
{

}

SolverWrapper::~SolverWrapper()
{

}

void SolverWrapper::snapshot(int iter)
{
  Tensor orig_0, orig_1;

  if(cfg.TRAIN.BBOX_REG && net.output_exists("bbox_pred"))
  {
    Output weights = net.get_weight("bbox_pred");
    Output biases = net.get_bias("bbox_pred");

    vector<Tensor> outputs;
    TF_CHECK_OK(session.Run({weights, biases}, &outputs));
    orig_0 = outputs[0];
    orig_1 = outputs[1];

    TF_CHECK_OK(session.Run({Multiply(scope, orig_0, +bbox_stds), Add(scope, Multiply(scope, orig_1, +bbox_stds), +bbox_means)}, &outputs));
    TF_CHECK_OK(session.Run({{net.bbox_weights, outputs[0]}, {net.bbox_biases, outputs[1]}}, {net.bbox_weights_assign, net.bbox_biases_assign}, NULL));
  }

  //获取 CKPT 文件名
  const string filename = output_dir + cfg.TRAIN.SNAPSHOT_PREFIX + "_iter_" + to_string(iter + 1) + ".ckpt";

  //获取变量名列表
  Tensor arr_slice_tensor(DT_STRING, {((int)net.get_weight_list().size())<<1});
  auto temp_arr_slice = arr_slice_tensor.vec<string>();
  vector<string> weights_list= net.get_weight_list();
  vector<string> biases_list = net.get_bias_list();

  //获取张量值列表
  vector<Output> tensor_list;
  for(int i=0; i<weights_list.size(); i++)
  {
    tensor_list.push_back(net.get_weight(weights_list[i]));
    tensor_list.push_back(net.get_bias(biases_list[i]));
  }
  for(int i=0; i<weights_list.size(); i++)
  {
    temp_arr_slice(i<<1) = weights_list[i]+"/weights";
    temp_arr_slice((i<<1)+1) = biases_list[i]+"/biases";
  }

  //将变量名和张量值列表存入指定的 CKPT 文件路径中
  TF_CHECK_OK(session.Run(ClientSession::FeedType(), {}, {Save(scope, filename, arr_slice_tensor, tensor_list)}, NULL));
  cout << "Wrote snapshot to: " << filename << endl;

  if(cfg.TRAIN.BBOX_REG && net.output_exists("bbox_pred"))
  {
    TF_CHECK_OK(session.Run({{net.bbox_weights, orig_0}, {net.bbox_biases, orig_1}}, {net.bbox_weights_assign, net.bbox_biases_assign}, NULL));
  }
}

Output SolverWrapper::_modified_smooth_l1(float sigma, Input bbox_pred, Input bbox_targets, Input bbox_inside_weights, Input bbox_outside_weights)
{
  float sigma2 = sigma*sigma;
  auto inside_mul = Multiply(scope,bbox_inside_weights,Subtract(scope,bbox_pred,bbox_targets));
  auto smooth_l1_sign = Cast(scope,Less(scope,Abs(scope,inside_mul),1.0f/sigma2),DT_FLOAT);
  auto smooth_l1_option1 = Multiply(scope,Multiply(scope,inside_mul,inside_mul),0.5f/sigma2);
  auto smooth_l1_option2 = Subtract(scope,Abs(scope,inside_mul),0.5f/sigma2);
  auto smooth_l1_result = Add(scope,Multiply(scope,smooth_l1_option1,smooth_l1_sign),Multiply(scope,smooth_l1_option2,Abs(scope,Subtract(scope,smooth_l1_sign,1.0f))));
  return Multiply(scope,bbox_outside_weights,smooth_l1_result);
}

void SolverWrapper::train_model(int max_iters, int iter)
{
  vector<Tensor> outputs;
  string pretrained_model = output_dir + cfg.TRAIN.SNAPSHOT_PREFIX + "_iter_" + to_string(iter) + ".ckpt";
  cout << "Loading pretrained model weights from " << pretrained_model << endl;
  net.load(session, pretrained_model); //加载预训练模型 net.load

  //RPN
  //classification loss
  auto rpn_cls_score_old = Reshape(scope,net.get_output("rpn_cls_score_reshape").front(),{-1,2});
  auto rpn_label_old = Reshape(scope,net.get_output("rpn-data").front(),{-1});
  auto rpn_cls_score = Reshape(scope,GatherNd(scope,rpn_cls_score_old,Cast(scope,Where(scope,NotEqual(scope,rpn_label_old,-1)),DT_INT32)),{-1,2});
  auto rpn_label = Reshape(scope,GatherNd(scope,rpn_label_old,Cast(scope,Where(scope,NotEqual(scope,rpn_label_old,-1)),DT_INT32)),{-1});
  auto rpn_cross_entropy = ReduceMean(scope,SparseSoftmaxCrossEntropyWithLogits(scope,rpn_cls_score,rpn_label).loss,{0});

  auto rpn_bbox_pred = net.get_output("rpn_bbox_pred").front();
  auto rpn_bbox_targets = Transpose(scope,(net.get_output("rpn-data"))[1],{0,2,3,1});
  auto rpn_bbox_inside_weights = Transpose(scope,(net.get_output("rpn-data"))[2],{0,2,3,1});
  auto rpn_bbox_outside_weights = Transpose(scope,(net.get_output("rpn-data"))[3],{0,2,3,1});
  auto rpn_smooth_li = _modified_smooth_l1(3.0f,rpn_bbox_pred,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights);
  auto rpn_loss_box = Mean(scope,Sum(scope,rpn_smooth_li,{1,2,3}),{0});

  //R-CNN
  //得到最后一个score分支fc层的输出
  auto cls_score = net.get_output("cls_score").front();
  auto label = Reshape(scope,net.get_output("roi-data")[1],{-1});
      //cout << "~~~~~~~~~~~~~~~~~~~~~~~~~0~~~~~~~~~~~~~~~~~~~~~~" << endl;
      //TF_CHECK_OK(session.Run({{net.data, blobs.data}, {net.im_info, blobs.im_info}, {net.gt_boxes, blobs.gt_boxes}}, {label}, &outputs));
      //cout << "~~~~~~~~~~~~~~~~~~~~~~~~~1~~~~~~~~~~~~~~~~~~~~~~" << endl;
      //TF_CHECK_OK(session.Run({{net.data, blobs.data}, {net.im_info, blobs.im_info}, {net.gt_boxes, blobs.gt_boxes}}, {cls_score}, &outputs));
      //cout << "~~~~~~~~~~~~~~~~~~~~~~~~~2~~~~~~~~~~~~~~~~~~~~~~" << endl;
  auto cross_entropy = ReduceMean(scope,SparseSoftmaxCrossEntropyWithLogits(scope,cls_score,label).loss,{0});
      //TF_CHECK_OK(session.Run({{net.data, blobs.data}, {net.im_info, blobs.im_info}, {net.gt_boxes, blobs.gt_boxes}}, {cross_entropy}, &outputs));
      //cout << "~~~~~~~~~~~~~~~~~~~~~~~~~3~~~~~~~~~~~~~~~~~~~~~~" << endl;
  auto bbox_pred = net.get_output("bbox_pred").front();
  auto bbox_targets = (net.get_output("roi-data"))[2];
  auto bbox_inside_weights = (net.get_output("roi-data"))[3];
  auto bbox_outside_weights = (net.get_output("roi-data"))[4];
      //TF_CHECK_OK(session.Run({{net.data, blobs.data}, {net.im_info, blobs.im_info}, {net.gt_boxes, blobs.gt_boxes}}, {bbox_targets}, &outputs));
      //cout << outputs[0].DebugString() << endl;
  auto smooth_li = _modified_smooth_l1(1.0f,bbox_pred,bbox_targets,bbox_inside_weights,bbox_outside_weights);
  auto loss_box = ReduceMean(scope,ReduceSum(scope,smooth_li,{1}),{0}); //smooth_l1计算出的为一个向量，现在要合成loss形式
  auto loss = Add(scope,Add(scope,cross_entropy,loss_box),Add(scope,rpn_cross_entropy,rpn_loss_box)); //final loss

  //momentum 梯度下降
  vector<Output> grad;
  vector<Output> accum_var_list; //var累加中间值accum
  vector<Output> assign_op_list; //累加变量accum初始化
  vector<Operation> train_op_list;  //var梯度更新op
  vector<Output> variables = net.get_trainable_variables();
  vector<Output> all_variables = variables;
  vector<Output> non_trainable_variables = net.get_non_trainable_variables();
  vector<TensorShape> variable_shapes = net.get_trainable_variable_shapes();
  auto learning_rate = Placeholder(scope, DT_FLOAT, Placeholder::Shape({}));
  const float momentum = cfg.TRAIN.MOMENTUM; //动态系数为0.9的梯度下降法
  all_variables.insert(all_variables.end(), non_trainable_variables.begin(), non_trainable_variables.end());
  TF_CHECK_OK(AddSymbolicGradients(scope, {loss}, all_variables, &grad));
  cout << "~~~~~~~~~~~~~~~~~~~~~~~~AddSymbolicGradients Ends~~~~~~~~~~~~~~~~~~~~~~" << endl;
  for(int i=0; i<variables.size(); i++)
  {
    accum_var_list.push_back(Variable(scope, variable_shapes[i], DT_FLOAT));
    Tensor zero_tensor(DT_FLOAT, variable_shapes[i]);
    zero_tensor.flat<float>().setZero();
    assign_op_list.push_back(Assign(scope, accum_var_list[i], zero_tensor));
    train_op_list.push_back(ApplyMomentum(scope, variables[i], accum_var_list[i], learning_rate, grad[i], momentum).out.op());
  }
  auto global_step = Variable(scope, TensorShape(), DT_FLOAT);
  TF_CHECK_OK(session.Run(assign_op_list, NULL));
  TF_CHECK_OK(session.Run({Assign(scope, global_step, 0.0f)}, NULL));
  Scope sub_scope = scope.WithControlDependencies(train_op_list);
  auto no_op = NoOp(sub_scope);
  auto train_op = AssignAdd(sub_scope.ColocateWith(no_op), global_step, 1.0f);

  //run_op_list
  vector<Output> run_ops;
  #ifdef TIMER_DEBUG
      net.timers.push_back(Time(scope.WithOpName("loss_timer"), loss).time);
      net.timers.push_back(Time(scope.WithOpName("grad_back_timer"), train_op_list.back().output(0)).time);
      net.timers.push_back(Time(scope.WithOpName("grad_front_timer"), train_op_list.front().output(0)).time);
      run_ops.insert(run_ops.end(), net.timers.begin(), net.timers.end());
  #endif
  run_ops.push_back(train_op);

  //训练部分
  int last_snapshot_iter = iter - 1;
  #ifdef TIMER_DEBUG
      FILE *file = fopen("train_timers.csv", "w");
      for(int i=0; i<16; ++i){
          fprintf(file, "%s,", ((std::string)net.timers[i].name()).c_str());
      }
      fprintf(file, "data_prepare,total_time,iter_num\n");
  #endif
  while(iter < max_iters)
  {
    myTimer timer,timer1;
    timer.tic();
    timer1.tic();
    rdl::Blobs blobs = data_layer.forward();
    float decayed_rate = cfg.TRAIN.LEARNING_RATE*exp((iter/cfg.TRAIN.STEPSIZE)*log(0.1));  // 更新 learning rate ，使用 exponential decay
    timer1.toc();
        // TF_CHECK_OK(session.Run({{net.data, blobs.data}, {net.im_info, blobs.im_info}, {net.gt_boxes, blobs.gt_boxes}, {learning_rate, decayed_rate}},
        //                         {net.get_output("conv5_3")[0], rpn_loss_box}, &outputs));
        // int n = outputs[0].shape().num_elements(); ofstream ofs;
        // auto test_tensor_data = outputs[0].flat<float>().data();
        // ofs.open("test_tensor_conv5_3.txt");
        // for(int i=0; i<n; i++)
        // {
        //    ofs << fixed << setprecision(4) << ((float*)test_tensor_data)[i] << endl;
        // }
        // ofs.close();
        // std::cout << outputs[1].DebugString() << std::endl;
    TF_CHECK_OK(session.Run({{net.data, blobs.data}, {net.im_info, blobs.im_info}, {net.gt_boxes, blobs.gt_boxes}, {learning_rate, decayed_rate}}, run_ops, &outputs));
    timer.toc();
    printf("~~~~~~~~~~~~~~~~~~~~~~~~iter %d cost time: %0.6f ms~~~~~~~~~~~~~~~~~~~~~~\n", iter+1, timer.total_time);        
    #ifdef TIMER_DEBUG
        //save csv
        for(int i=1; i<16; ++i){
            fprintf(file, ",%0.6f", outputs[i].scalar<double>()(0)-outputs[i-1].scalar<double>()(0));
        }
        fprintf(file, ",%0.6f,%0.6f,%d\n",timer1.total_time,timer.total_time, iter+1);
    #endif
    //model save
    if((iter+1)%cfg.TRAIN.DISPLAY==0)
    {
      printf("iter: %d / %d\n", iter+1, max_iters);
    }
    if((iter + 1) % cfg.TRAIN.SNAPSHOT_ITERS == 0)
    {
      last_snapshot_iter = iter;
      snapshot(iter);
    }
    iter++;
  }
  #ifdef TIMER_DEBUG
      fclose(file);
  #endif
  if(last_snapshot_iter != iter - 1)
  {
    snapshot(iter - 1);
  }
}

void train_net(const Scope &scope, const string &output_dir, const int iter, const int max_iters)
{
    SolverWrapper solver(scope, output_dir);
    cout << "Solving..." << endl;
    solver.train_model(max_iters, iter);
    cout << "Done solving" << endl;
}

}
