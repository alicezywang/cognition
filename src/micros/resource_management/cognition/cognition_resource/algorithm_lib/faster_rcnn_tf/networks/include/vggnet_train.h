#ifndef VGGNET_TRAIN_H
#define VGGNET_TRAIN_H

#include "network.h"

class vggnet_train: public Network
{
public:
    Output bbox_weights_assign, bbox_biases_assign;
    Output bbox_weights, bbox_biases, data, im_info, gt_boxes;
    vggnet_train(const Scope &scope);
    ~vggnet_train();
    virtual void setup();
};

#endif
