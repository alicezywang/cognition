#include <vector>
#include <iostream>
#include "bbox.h"
using namespace std;

int main(){
    vector<vector<float> > Pre_box = {{0,0,2,2},{0,0,4,4}};
    vector<vector<float> > Ft_box = {{1,1,3,3},{2,2,6,6}}; 
    vector<vector<float> > result = overlaps(Pre_box, Ft_box);
		for (int i = 0; i < result.size(); i++){
			for (int j = 0; j < result[0].size(); j++){
				cout << result[i][j] << "  ";
			}
			cout << endl;
		}
    	return 0;
    
}