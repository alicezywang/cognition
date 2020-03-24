#include <iostream>
#include <vector>
#include <algorithm>
#include <unsupported/Eigen/CXX11/Tensor>
#include "nms/cpu_nms.h"

static bool cmp(const std::pair<int, float> &x, const std::pair<int, float> &y)
{
    return x.second > y.second;
}

//nms.pyx 中的 nms() 改写
Tensor1i cpu_nms_old(Tensor2f dets, float thresh)
{
    std::vector<float> x1,y1,x2,y2,areas;
    std::vector<std::pair<int, float> > scores;
    std::vector<int> keepv;
    float xx1,yy1,xx2,yy2,w,h,inter,ovr;
    for(int i = 0; i < dets.dimension(0); i++)
    {
        x1.push_back(dets(i,0));
        y1.push_back(dets(i,1));
        x2.push_back(dets(i,2));
        y2.push_back(dets(i,3));
        scores.push_back(std::make_pair(i,dets(i,4)));
        areas.push_back((x2[i] - x1[i] +1) * (y2[i] - y1[i] +1));
    }
    //从大到小返回元素索引
    sort(scores.begin(), scores.end(), cmp);
    //把索引存到order里
    while(scores.size() > 0)
    {
        keepv.push_back(scores[0].first);

        for(std::vector<std::pair<int, float> >::iterator i = scores.begin()+1; i != scores.end(); )
        {
            xx1 = (x1[scores[0].first] > x1[i->first]? x1[scores[0].first] : x1[i->first]);
            yy1 = (y1[scores[0].first] > y1[i->first]? y1[scores[0].first] : y1[i->first]);
            xx2 = (x2[scores[0].first] < x2[i->first]? x2[scores[0].first] : x2[i->first]);
            yy2 = (y2[scores[0].first] < y2[i->first]? y2[scores[0].first] : y2[i->first]);

            w = 0.0 > (xx2 - xx1 + 1) ? 0.0 : (xx2 - xx1 + 1);
            h = 0.0 > (yy2 - yy1 + 1) ? 0.0 : (yy2 - yy1 + 1);
            inter = w * h;
            ovr = inter / (areas[scores[0].first] + areas[i->first] - inter);
            //剔除不满足条件的元素
            if(ovr >= thresh)
                scores.erase(i);
            else   
                i++;
            // ovr = inter / (areas[scores[0].first] + areas[i->first] - inter);
            // ovr1 = inter / areas[scores[0].first];
            // ovr2 = inter / areas[i->first];
            // if(ovr < thresh && ovr1 <= 0.95 && ovr2 <= 0.95)
        }
        scores.erase(scores.begin());
    }
    int len = keepv.size();
    Tensor1i keep(len);
    for(int i = 0; i < len; i++)
    {
        keep(i) = keepv[i];
    }

    //返回值是一个 Eigen::Tensor<int, 1, Eigen::RowMajor>, 对应Python版中的list
    return keep;

}

//nms.pyx 中的 nms_new() 改写
Tensor1i cpu_nms(Tensor2f dets, float thresh)
{
    std::vector<float> x1,y1,x2,y2,areas;
    std::vector<std::pair<int, float> > scores;
    std::vector<int> order;
    std::vector<int> tmp;
    std::vector<int> keepv;
    float xx1,yy1,xx2,yy2,w,h,inter,ovr;
    std::vector<float> inds;
    for(int i = 0; i < dets.dimension(0); i++)
    {
        x1.push_back(dets(i,0));
        y1.push_back(dets(i,1));
        x2.push_back(dets(i,2));
        y2.push_back(dets(i,3));
        scores.push_back(std::make_pair(i,dets(i,4)));
        areas.push_back((x2[i] - x1[i] +1) * (y2[i] - y1[i] +1));
    }
    //从大到小返回元素索引
    sort(scores.begin(), scores.end(), cmp);
    //把索引存到order里
    for(std::vector<std::pair<int, float> >::size_type i = 0; i < scores.size(); i++)
    {
        order.push_back(scores[i].first);
    }

    while(order.size() > 0)
    {
        keepv.push_back(order[0]);
        int len1 = order.size();
        for(int i = 1; i < len1; i++)
        {
            xx1 = (x1[order[0]] > x1[order[i]]? x1[order[0]] : x1[order[i]]);
            yy1 = (y1[order[0]] > y1[order[i]]? y1[order[0]] : y1[order[i]]);
            xx2 = (x2[order[0]] < x2[order[i]]? x2[order[0]] : x2[order[i]]);
            yy2 = (y2[order[0]] < y2[order[i]]? y2[order[0]] : y2[order[i]]);

            w = 0.0 > (xx2 - xx1 + 1) ? 0.0 : (xx2 - xx1 + 1);
            h = 0.0 > (yy2 - yy1 + 1) ? 0.0 : (yy2 - yy1 + 1);
            inter = w * h;
            ovr = inter / (areas[order[0]] + areas[order[i]] - inter);
            if(ovr < thresh)
                inds.push_back(order[i]);
                //python版中inds是获取ovr中元素小于thresh的下标，然后根据这些下标重新输出order
                //这里直接把小于thresh的下标对应的order值存到inds中
        }
        order.clear();
        int len2 = inds.size();
        for(int i = 0; i < len2; i++)
            order.push_back(inds[i]);
        inds.clear();
    }

    int len = keepv.size();
    Tensor1i keep(len);
    for(int i = 0; i < len; i++)
    {
        keep(i) = keepv[i];
    }

    //返回值是一个 Eigen::Tensor<int, 1, Eigen::RowMajor>, 对应Python版中的list
    return keep;
}
