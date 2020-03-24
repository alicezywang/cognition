#include "fast_rcnn/train.h"
#include "fast_rcnn/config.h"


int main()
{
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();


  tensorflow::Output weights = tensorflow::ops::Variable(scope.WithOpName("weights"),{2},tensorflow::DT_FLOAT);
  tensorflow::Output biases = tensorflow::ops::Variable(scope.WithOpName("biases"),{2},tensorflow::DT_FLOAT);

  auto assign_W = Assign(scope, weights, RandomNormal(scope, {2}, DT_FLOAT));
  auto assign_b = Assign(scope, biases, RandomNormal(scope, {2}, DT_FLOAT));


  std::string filename = "./src/faster_rcnn_tf/data/saveV2.ckpt";
  std::string filename_save = "./src/faster_rcnn_tf/data/save.ckpt";
  //savev2
  tensorflow::OutputList tensorlist;
  tensorlist.push_back(weights);
  tensorlist.push_back(biases);

  std::initializer_list<tensorflow::string> arr_slice;
  std::initializer_list<tensorflow::string> arr_slice_str;
  arr_slice={"conv1_1/weights","conv1_1/biases"};
  arr_slice_str={"",""};

  auto saveV2 = tensorflow::ops::SaveV2(scope.WithOpName("savev2"),filename,Input(arr_slice),Input(arr_slice_str),tensorlist);

  auto save = tensorflow::ops::Save(scope.WithOpName("save"),filename_save,Input(arr_slice),tensorlist);

  GraphDef graph_def;
  scope.ToGraphDef(&graph_def);

  //  TF_CHECK_OK(sess->Create(graph_def));
  tensorflow::ClientSession sess(scope);
  std::vector<Tensor> outputs;
  TF_CHECK_OK(sess.Run({assign_W,assign_b},nullptr));
  // TF_CHECK_OK(sess->Run({},{"weights"},{"biases"},&outputs));
  TF_CHECK_OK(sess.Run({weights,biases},&outputs));

  //saveV2
  TF_CHECK_OK(sess.Run(ClientSession::FeedType(),{},{saveV2},NULL));
  //save
  TF_CHECK_OK(sess.Run(ClientSession::FeedType(),{},{save},NULL));

  std::cout<<"success"<<std::endl;
}
