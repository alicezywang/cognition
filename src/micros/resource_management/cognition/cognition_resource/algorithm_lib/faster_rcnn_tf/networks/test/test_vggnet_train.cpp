#include "vggnet_train.h"

int main()
{
    std::vector<Tensor> outputs;
    Scope scope = Scope::NewRootScope();
    ClientSession session(scope);
    vggnet_train train(scope);
    train.load(session, "src/faster_rcnn_tf/data/pretrain_model/vgg_imagenet.ckpt");
    TF_CHECK_OK(session.Run(train.feed_map, {train.get_output("rpn_cls_score_reshape").front()}, &outputs));
    std::cout << outputs[0].DebugString() << std::endl;
    return 0;
}
