#include <algorithm>
#include <math.h>
#include <vector>
#include "generate_anchors.h"

using namespace std;

anchors_gen::anchors_gen(): base_size(16), ratios{0.5,1.0,2.0}, scales{8.0,16.0,32.0} {}

anchors_gen::~anchors_gen(){}

//Return width, height, x center, and y center for an anchor (window).
//(x1,y1,x2,y2) -> (w,h,ctrx,ctry)
vector<float> anchors_gen::whctrs(vector<float> anchor)
{
		vector<float> result;
		result.push_back(anchor[2] - anchor[0] + 1); //w, (x2-x1) + 1 ?
		result.push_back(anchor[3] - anchor[1] + 1); //h
		result.push_back((anchor[2] + anchor[0]) / 2); //ctrx
		result.push_back((anchor[3] + anchor[1]) / 2); //ctry
		return result;
}

/*
Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
实际上是whctrs的逆操作，即根据（w,h,ctrx,ctry）-> (x1,y1,x2,y2)
求解方程：
            x2 - x1 +1 = w
            y2 - y1 +1 = h
            (x1 + x2)/2 = ctrx
            (y1 + y2)/2 = ctry
*/
vector<float> anchors_gen::mkanchor(float w, float h, float x_ctr, float y_ctr)
{
		vector<float> tmp;
		tmp.push_back(x_ctr - 0.5*(w - 1));
		tmp.push_back(y_ctr - 0.5*(h - 1));
		tmp.push_back(x_ctr + 0.5*(w - 1));
		tmp.push_back(y_ctr + 0.5*(h - 1));
		return tmp;
}

/*
Enumerate a set of anchors for each aspect ratio wrt an anchor.
求解方程：
        h/w = ratios
        w*h = s
    (x1,y1,x2,y2) ->（w,h,ctrx,ctry）-> (ws, hs, x_ctr, y_ctr) -> 
    (x1',y1',x2',y2') -> {(x1',y1',x2',y2'),... }根据3种高宽比生成3个anchor
*/
vector<vector<float> > anchors_gen::ratio_enum(vector<float> anchor)
{
		vector<vector<float> > result;
		vector<float> reform_anchor = whctrs(anchor);
		float x_ctr = reform_anchor[2];
		float y_ctr = reform_anchor[3];
		float size = reform_anchor[0] * reform_anchor[1];//area
		for (int i = 0; i < 3; ++i)
		{
        float size_ratios = size / ratios[i];
        float ws = round(sqrt(size_ratios));
        float hs = round(ws*ratios[i]);
        vector<float> tmp = mkanchor(ws, hs, x_ctr, y_ctr);
        result.push_back(tmp);
		}
		return result;
}

//Enumerate a set of anchors for each scale wrt an anchor.
//Similar to ratio_enum
vector<vector<float> > anchors_gen::scale_enum(vector<float> anchor)
{
		vector<vector<float> > result;
		vector<float> reform_anchor = whctrs(anchor);
		float x_ctr = reform_anchor[2];
		float y_ctr = reform_anchor[3];
		float w = reform_anchor[0];
		float h = reform_anchor[1];
		for (int i = 0; i < 3; ++i)
		{
        float ws = w * scales[i];
        float hs = h *  scales[i];
        vector<float> tmp = mkanchor(ws, hs, x_ctr, y_ctr);
        result.push_back(tmp);
		}
		return result;
}

//return 9 个 anchors
vector<vector<float> > anchors_gen::generate_anchors()
{
    vector<vector<float> > result;
    //generate base anchor (0,0,15,15), 表示左上角跟右下角坐标
		vector<float> base_anchor;
		base_anchor.push_back(0);
		base_anchor.push_back(0);
		base_anchor.push_back(base_size - 1);
		base_anchor.push_back(base_size - 1);
		//enum ratio anchors
		vector<vector<float> >ratio_anchors = ratio_enum(base_anchor);
		for (int i = 0; i < ratio_anchors.size(); ++i)
		{
        vector<vector<float> > tmp = scale_enum(ratio_anchors[i]); //给每一个比例windows做尺寸变化
        result.insert(result.end(), tmp.begin(), tmp.end());
		}
    return result;
}
