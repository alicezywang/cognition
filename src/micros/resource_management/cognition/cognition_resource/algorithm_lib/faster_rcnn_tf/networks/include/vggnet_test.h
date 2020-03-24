#ifndef VGGNET_TEST_H
#define VGGNET_TEST_H

#include "network.h"

class vggnet_test: public Network
{
public:
    Output data, im_info;
    vggnet_test(const Scope &scope);
    ~vggnet_test();
    virtual void setup();
};

#endif
