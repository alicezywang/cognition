#include <vector>
#include "bbox.h"
using namespace std;

float min(float a, float b){
    return (a<b?a:b);
}

float max(float a, float b){
    return (a>=b?a:b);
}
/**返回交并比（IOU）**/
vector<vector<float> > overlaps(vector<vector<float> > boxes, vector<vector<float> > query_boxes){
    int N = boxes.size();   
    int K = query_boxes.size();  
    vector<vector<float> > result(N,vector<float>(K,0));
    float iw,ih,box_area;
    float ua;
    for(int i = 0; i < K; ++i){
        box_area = (query_boxes[i][2] - query_boxes[i][0] + 1) *
                   (query_boxes[i][3] - query_boxes[i][1] + 1);
        for (int j = 0; j < N; ++j){
            iw =  min(boxes[j][2], query_boxes[i][2]) -         
                  max(boxes[j][0], query_boxes[i][0]) + 1;

            if(iw > 0){      
                ih =  min(boxes[j][3], query_boxes[i][3]) -     
                      max(boxes[j][1], query_boxes[i][1]) + 1;     
                if(ih > 0){  
                    ua = (boxes[j][2] - boxes[j][0] + 1) *      
                         (boxes[j][3] - boxes[j][1] + 1) +
                         box_area - iw * ih;
                    result[j][i] = iw * ih / ua;                
                }             
            }
            
        }
    }
    return result;
}
