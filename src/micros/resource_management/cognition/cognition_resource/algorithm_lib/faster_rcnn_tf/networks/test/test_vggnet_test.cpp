#include <tensorflow/core/public/session.h>
#include "vggnet_test.h"
#include "fast_rcnn/config.h"

int main()
{
    Scope scope = Scope::NewRootScope();
    vggnet_test test(scope);
    std::vector<Tensor> outputs;
    const std::vector<std::string> weight_list = test.get_weight_list();
    const std::vector<std::string> bias_list = test.get_bias_list();
    const int count = weight_list.size();
    assert(weight_list == bias_list);

    // 写入PB
    // std::cout << std::string("Writing data structure GraphDef to PB file: ") + fast_rcnn::cfg.DATA_DIR + std::string("vggnet_test.pb") << std::endl;
    // TF_CHECK_OK(WriteBinaryProto(tensorflow::Env::Default(), fast_rcnn::cfg.DATA_DIR + "vggnet_test.pb", test.get_graph_def()));

    // Restore 权重恢复及验证
    for(int i=0; i<count; i++)
    {
        Assign(scope.WithOpName(weight_list[i] + "/assign_weights"), test.get_weight(weight_list[i]), Restore(scope, fast_rcnn::cfg.DATA_DIR + "VGGnet_fast_rcnn_iter_70000.ckpt", weight_list[i] + "/weights", DT_FLOAT));
    }
    for(int i=0; i<count; i++)
    {
        Assign(scope.WithOpName(bias_list[i] + "/assign_biases"), test.get_bias(bias_list[i]), Restore(scope, fast_rcnn::cfg.DATA_DIR + "VGGnet_fast_rcnn_iter_70000.ckpt", bias_list[i] + "/biases", DT_FLOAT));
    }

    GraphDef graph_def;
    TF_CHECK_OK(scope.ToGraphDef(&graph_def));
    tensorflow::Session *session = tensorflow::NewSession(tensorflow::SessionOptions());
    session->Create(graph_def);
    for(int i=0; i<count; i++)
    {
        TF_CHECK_OK(session->Run({}, {}, {weight_list[i] + "/assign_weights"}, NULL));
        TF_CHECK_OK(session->Run({}, {weight_list[i] + "/weights"}, {}, &outputs));
        std::cout << weight_list[i] << std::endl;
        std::cout << outputs[0].shape().DebugString() << std::endl;
    }
    for(int i=0; i<count; i++)
    {
        TF_CHECK_OK(session->Run({}, {}, {bias_list[i] + "/assign_biases"}, NULL));
        TF_CHECK_OK(session->Run({}, {bias_list[i] + "/biases"}, {}, &outputs));
        std::cout << bias_list[i] << std::endl;
        std::cout << outputs[0].shape().DebugString() << std::endl;
    }
    session->Close();

    // RestoreV2 权重恢复及验证
    /*ClientSession session(scope);
    for(int i=0; i<count; i++)
    {
        auto restore = RestoreV2(scope, fast_rcnn::cfg.DATA_DIR + "VGGnet_fast_rcnn_iter_70000.ckpt", {weight_list[i] + "/weights", bias_list[i] + "/biases"}, {std::string(), std::string()}, tensorflow::DataTypeSlice({DT_FLOAT, DT_FLOAT}));
        TF_CHECK_OK(session.Run({Assign(scope, test.get_weight(weight_list[i]), restore[0])}, NULL));
        TF_CHECK_OK(session.Run({Assign(scope, test.get_bias(bias_list[i]), restore[1])}, NULL));
        TF_CHECK_OK(session.Run({Shape(scope, test.get_weight(weight_list[i]))}, &outputs));
        std::cout << weight_list[i] << std::endl;
        std::cout << outputs[0].tensor<int, 1>() << std::endl;
        TF_CHECK_OK(session.Run({Shape(scope, test.get_bias(bias_list[i]))}, &outputs));
        std::cout << bias_list[i] << std::endl;
        std::cout << outputs[0].tensor<int, 1>() << std::endl;
    }*/
    return 0;
}
