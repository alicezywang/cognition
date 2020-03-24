#ifndef _GENERATE_ANCHORS_H_
#define _GENERATE_ANCHORS_H_

#include <vector>
using namespace std;

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
#endif
