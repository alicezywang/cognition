#ifndef _BBOX_H_
#define _BBOX_H_

#include <vector>
using namespace std;
/*计算anchors与ft_box区域交集与并集的比值，-> LOU*/
vector<vector<float> > overlaps(vector<vector<float> > boxes, vector<vector<float> > query_boxes);

#endif
