#include "vggnet_train.h"

#define CLASSES 21

static std::vector<int> _feat_stride({16});
static std::vector<int> anchor_scales({8,16,32});

vggnet_train::vggnet_train(const Scope &scope): Network(scope)
{
    std::vector<Tensor> data_list, gt_boxes_list;
    Tensor im_info_tensor(DT_FLOAT, {1, 3});
    im_info_tensor.tensor<float, 2>()(0, 0) = 60.0f;
    im_info_tensor.tensor<float, 2>()(0, 1) = 80.0f;
    im_info_tensor.tensor<float, 2>()(0, 2) = 1.0f;
    data = Placeholder(scope.WithOpName("data"), DT_FLOAT, Placeholder::Shape({-1, -1, -1, 3}));
    im_info = Placeholder(scope.WithOpName("im_info"), DT_FLOAT, Placeholder::Shape({-1, 3}));
    gt_boxes = Placeholder(scope.WithOpName("gt_boxes"), DT_FLOAT, Placeholder::Shape({-1, 5}));
    layers.insert(std::map<std::string, std::vector<Output> >::value_type("data", {data}));
    layers.insert(std::map<std::string, std::vector<Output> >::value_type("im_info", {im_info}));
    layers.insert(std::map<std::string, std::vector<Output> >::value_type("gt_boxes", {gt_boxes}));
    TF_CHECK_OK(session.Run({RandomNormal(scope, {1, 60, 80, 3}, DT_FLOAT)}, &data_list));
    TF_CHECK_OK(session.Run({RandomNormal(scope, {3, 5}, DT_FLOAT)}, &gt_boxes_list));
    feed_map = FeedType({{data, data_list[0]}, {im_info, im_info_tensor}, {gt_boxes, gt_boxes_list[0]}});
    setup();
    Output weights = get_weight("bbox_pred");
    Output biases = get_bias("bbox_pred");
    bbox_weights = Placeholder(scope, DT_FLOAT, Placeholder::Shape((*(weight_shapes.find("bbox_pred"))).second));
    bbox_biases = Placeholder(scope, DT_FLOAT, Placeholder::Shape((*(bias_shapes.find("bbox_pred"))).second));
    bbox_weights_assign = Assign(scope, weights, bbox_weights);
    bbox_biases_assign = Assign(scope, biases, bbox_biases);
}
vggnet_train::~vggnet_train()
{

}
void vggnet_train::setup()
{
    std::vector<Tensor> output_list;
    this->feed({"data"})
        ->conv(3, 3, 64, 1, 1, "conv1_1", true, "SAME", 1, false)
        ->conv(3, 3, 64, 1, 1, "conv1_2", true, "SAME", 1, false)
        ->max_pool(2, 2, 2, 2, "pool1", "VALID")
        ->conv(3, 3, 128, 1, 1, "conv2_1", true, "SAME", 1, false)
        ->conv(3, 3, 128, 1, 1, "conv2_2", true, "SAME", 1, false)
        ->max_pool(2, 2, 2, 2, "pool2", "VALID")
        ->conv(3, 3, 256, 1, 1, "conv3_1")
        ->conv(3, 3, 256, 1, 1, "conv3_2")
        ->conv(3, 3, 256, 1, 1, "conv3_3")
        ->max_pool(2, 2, 2, 2, "pool3", "VALID")
        ->conv(3, 3, 512, 1, 1, "conv4_1")
        ->conv(3, 3, 512, 1, 1, "conv4_2")
        ->conv(3, 3, 512, 1, 1, "conv4_3")
        ->max_pool(2, 2, 2, 2, "pool4", "VALID")
        ->conv(3, 3, 512, 1, 1, "conv5_1")
        ->conv(3, 3, 512, 1, 1, "conv5_2")
        ->conv(3, 3, 512, 1, 1, "conv5_3");

    // ============ RPN ============
    this->feed({"conv5_3"})                           // N, H/16, W/16, 512
        ->conv(3, 3, 512, 1, 1, "rpn_conv/3x3")       // N, H/16, W/16, 512
        ->conv(1, 1, anchor_scales.size()*3*2, 1, 1, "rpn_cls_score", false, "VALID");   // N, H/16, W/16, 18

    this->feed({"rpn_cls_score", "gt_boxes", "im_info", "data"})         
        ->anchor_target_layer(_feat_stride, anchor_scales, "rpn-data"); 

    // Loss of rpn_cls & rpn_boxes
    this->feed({"rpn_conv/3x3"})  // N, H/16, W/16, 36
        ->conv(1, 1, anchor_scales.size()*3*4, 1, 1, "rpn_bbox_pred", false, "VALID");  // N, H/16, W/16, 36

    // ============ RoI Proposal ============
    this->feed({"rpn_cls_score"})   // N, H/16, W/16, 18
        ->reshape_layer(2, "rpn_cls_score_reshape")   // N, 9*H/16, W/16, 2
        ->softmax("rpn_cls_prob"); //N, 9*H/16, W/16, 2

    this->feed({"rpn_cls_prob"})  //N, 9*H/16, W/16, 2
        ->reshape_layer(anchor_scales.size()*3*2, "rpn_cls_prob_reshape");//N, H/16, W/16, 18

    this->feed({"rpn_cls_prob_reshape", "rpn_bbox_pred", "im_info"})
        ->proposal_layer(_feat_stride, anchor_scales, "TRAIN", "rpn_rois");//TRAIN.RPN_POST_NMS_TOP_N,5

    this->feed({"rpn_rois", "gt_boxes"})
        ->proposal_target_layer(CLASSES, "roi-data");
    //rois：128×5，
    //labels_last: 128x1,
    //bbox_targets: 128×84，
    //bbox_inside_weights, bbox_outside_weights: 128×84,

    // ============ RCNN ============
    this->feed({"conv5_3", "roi-data"})
        ->roi_pool(7, 7, 1.0/16, "pool_5")
        ->fc(4096, "fc6")
        ->dropout(0.5, "drop6")
        ->fc(4096, "fc7")
        ->dropout(0.5, "drop7")
        ->fc(CLASSES, "cls_score", false)
        ->softmax("cls_prob");

    this->feed({"drop7"})
        ->fc(CLASSES*4, "bbox_pred", false);

    this->feed({"data"})->timer_leaf("data_timer");          // min
    this->feed({"conv5_3"})->timer_leaf("conv5_3_timer");    // rpn_msr:start
    this->feed({"roi-data"})->timer_leaf("roi-data_timer");  // rpn_msr:end + roi_pool:start
    this->feed({"pool_5"})->timer_leaf("pool_5_timer");      // roi_pool:end + drop7:start
    this->feed({"drop7"})->timer_leaf("drop7_timer");        // drop7:end
    this->feed({"cls_prob"})->timer_leaf("cls_prob_timer");  // max:1/2
    this->feed({"bbox_pred"})->timer_leaf("bbox_pred_timer");// max:1/2

    this->feed({"rpn_cls_score"})->timer_leaf("rpn_cls_score_timer");              // anchor_target:start
    this->feed({"rpn-data"})->timer_leaf("rpn-data_timer");                        // anchor_target:end
    this->feed({"rpn_cls_prob_reshape"})->timer_leaf("rpn_cls_prob_reshape_timer");// proposal:start
    this->feed({"rpn_rois"})->timer_leaf("rpn_rois_timer");                        // proposal:end + proposal_target:start
    this->timers.push_back(this->timers[2]);                                       // proposal_target:end
    this->timers.push_back(this->timers[6]);                                       // net:end = max:1/2

}
