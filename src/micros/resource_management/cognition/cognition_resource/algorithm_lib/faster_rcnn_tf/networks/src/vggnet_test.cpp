#include "vggnet_test.h"

#define CLASSES 21

static std::vector<int> _feat_stride({16});
static std::vector<int> anchor_scales({8,16,32});

vggnet_test::vggnet_test(const Scope &scope): Network(scope)
{
    std::vector<Tensor> data_list;
    Tensor im_info_tensor(DT_FLOAT, {1, 3});
    data = Placeholder(scope.WithOpName("data"), DT_FLOAT, Placeholder::Shape({-1, -1, -1, 3}));
    im_info = Placeholder(scope.WithOpName("im_info"), DT_FLOAT, Placeholder::Shape({-1, 3}));
    TF_CHECK_OK(session.Run({RandomNormal(scope, {1, 60, 80, 3}, DT_FLOAT)}, &data_list));
    im_info_tensor.tensor<float, 2>()(0, 0) = 60.0f;
    im_info_tensor.tensor<float, 2>()(0, 1) = 80.0f;
    im_info_tensor.tensor<float, 2>()(0, 2) = 1.0f;
    layers.insert(std::map<std::string, std::vector<Output> >::value_type("data", {data}));
    layers.insert(std::map<std::string, std::vector<Output> >::value_type("im_info", {im_info}));
    feed_map = FeedType({{data, data_list[0]}, {im_info, im_info_tensor}});
    setup();
}
vggnet_test::~vggnet_test()
{

}
void vggnet_test::setup()
{
    this->feed({"data"})                                             // 输出张量形状： N, H, W, 3
        ->conv(3, 3, 64, 1, 1, "conv1_1", true, "SAME", 1, false)    // N, H, W, 64
        ->conv(3, 3, 64, 1, 1, "conv1_2", true, "SAME", 1, false)    // N, H, W, 64
        ->max_pool(2, 2, 2, 2, "pool1", "VALID")                     // N, H/2, W/2, 64
        ->conv(3, 3, 128, 1, 1, "conv2_1", true, "SAME", 1, false)   // N, H/2, W/2, 128
        ->conv(3, 3, 128, 1, 1, "conv2_2", true, "SAME", 1, false)   // N, H/2, W/2, 128
        ->max_pool(2, 2, 2, 2, "pool2", "VALID")                     // N, H/4, W/4, 128
        ->conv(3, 3, 256, 1, 1, "conv3_1")                           // N, H/4, W/4, 256
        ->conv(3, 3, 256, 1, 1, "conv3_2")                           // N, H/4, W/4, 256
        ->conv(3, 3, 256, 1, 1, "conv3_3")                           // N, H/4, W/4, 256
        ->max_pool(2, 2, 2, 2, "pool3", "VALID")                     // N, H/8, W/8, 256
        ->conv(3, 3, 512, 1, 1, "conv4_1")                           // N, H/8, W/8, 512
        ->conv(3, 3, 512, 1, 1, "conv4_2")                           // N, H/8, W/8, 512
        ->conv(3, 3, 512, 1, 1, "conv4_3")                           // N, H/8, W/8, 512
        ->max_pool(2, 2, 2, 2, "pool4", "VALID")                     // N, H/16, W/16, 512
        ->conv(3, 3, 512, 1, 1, "conv5_1")                           // N, H/16, W/16, 512
        ->conv(3, 3, 512, 1, 1, "conv5_2")                           // N, H/16, W/16, 512
        ->conv(3, 3, 512, 1, 1, "conv5_3");                          // N, H/16, W/16, 512

    this->feed({"conv5_3"})                                          // N, H/16, W/16, 512
        ->conv(3, 3, 512, 1, 1, "rpn_conv/3x3")                      // N, H/16, W/16, 512
        ->conv(1, 1, anchor_scales.size()*3*2, 1, 1, "rpn_cls_score", false, "VALID"); // N, H/16, W/16, 18

    this->feed({"rpn_conv/3x3"})                                                       // N, H/16, W/16, 512
        ->conv(1, 1, anchor_scales.size()*3*4, 1, 1, "rpn_bbox_pred", false, "VALID"); // N, H/16, W/16, 36

    this->feed({"rpn_cls_score"})                                                      // N, H/16, W/16, 18
        ->reshape_layer(2, "rpn_cls_score_reshape")                                    // N, 9*H/16, W/16, 2
        ->softmax("rpn_cls_prob");                                                     // N, 9*H/16, W/16, 2

    this->feed({"rpn_cls_prob"})
        ->reshape_layer(anchor_scales.size()*3*2, "rpn_cls_prob_reshape"); // N, H/16, W/16, 18

    this->feed({"rpn_cls_prob_reshape", "rpn_bbox_pred", "im_info"})
        ->proposal_layer(_feat_stride, anchor_scales, "TEST", "rois");    // proposals_num, 5

    this->feed({"conv5_3", "rois"})
        ->roi_pool(7, 7, 1.0/16, "pool_5")  // proposals_num, 7, 7, 512
        ->fc(4096, "fc6")                   // proposals_num, 4096
        ->fc(4096, "fc7")                   // proposals_num, 4096
        ->fc(CLASSES, "cls_score", false)   // proposals_num, 21
        ->softmax("cls_prob");              // proposals_num, 21

    this->feed({"fc7"})
        ->fc(CLASSES*4, "bbox_pred", false);// proposals_num, 84

    this->feed({"data"})->timer_leaf("data_timer");          // min
    this->feed({"conv5_3"})->timer_leaf("conv5_3_timer");    // rpn_msr:start
    this->feed({"rois"})->timer_leaf("rois_timer");          // rpn_msr:end + roi_pool:start
    this->feed({"pool_5"})->timer_leaf("pool_5_timer");      // roi_pool:end + drop7:start
    this->feed({"fc7"})->timer_leaf("fc7_timer");            // drop7:end
    this->feed({"cls_prob"})->timer_leaf("cls_prob_timer");  // max:1/2
    this->feed({"bbox_pred"})->timer_leaf("bbox_pred_timer");// max:1/2

    this->feed({"rpn_cls_prob_reshape"})->timer_leaf("rpn_cls_prob_reshape_timer");// proposal:start
    this->timers.push_back(this->timers[2]);                                       // proposal:end

}
