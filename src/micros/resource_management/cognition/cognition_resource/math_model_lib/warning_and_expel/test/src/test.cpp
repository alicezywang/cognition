#include <warning_and_expel/warning_and_expel.h>
#include <nearby_reconnaissance_model_lib/nearby_reconnaissance_model.h>
#include <vector>
#include <iostream>
#include <iomanip>
using namespace std;
int main(int argc,char **argv){

	// std::vector< std::vector<double> > points;
	// std::vector<double> temp;
	// temp.push_back(84.4847);
	// temp.push_back(59.3366);
	// // temp.push_back(1);
	// // temp.push_back(1);
	// points.push_back(temp);
	// temp[0] = 62.0051;
	// temp[1] = 34.5763;
	// points.push_back(temp);
	// double a,b;
	// // warning_expel_model::LeastSquare(points,a,b);
	// std::cout<<a<<"  "<<b<<std::endl;



//	std::vector< std::vector<double> > points;
//	std::vector< std::vector<double> > in_out;
//	std::vector<double> temp;
//	temp.push_back(113.008274);
//	temp.push_back(28.230941);
//	points.push_back(temp);
//	temp[0] = 113.00805;
//	temp[1] = 28.23071;
//// 113.007398,28.231522

//	points.push_back(temp);

//	// 1
//	temp[0] = 113.008377;
//	temp[1] = 28.231561;
//	points.push_back(temp);

//	//2
//	temp[0] = 113.007398;
//	temp[1] = 28.231522;
//	points.push_back(temp);

//	//3
//	temp[0] = 113.007425;
//	temp[1] = 28.230376;
//	points.push_back(temp);

//	//4
//	temp[0] = 113.008409;
//	temp[1] = 28.230412;
//	points.push_back(temp);

//	temp[0]= 113.007232;
//	temp[1] = 28.231279;
//	points.push_back(temp);
//	temp[0] = 113.007232;
//	temp[1] = 28.230798;
//	points.push_back(temp);

//	warning_expel_model::lowAltitudeSuppression(points ,in_out);
//  std::cout <<"=====================低空掠过======================"<<std::endl;
//	std::cout<<std::fixed << std::setprecision(6)<<in_out[0][0] << " , "<< in_out[0][1] << " ; "<< in_out[1][0] << " , "<< in_out[1][1]<<std::endl;
//  std::cout <<"==================================================="<<std::endl;
//  std::cout <<std::endl;



//std::cout <<"=====================效果评估======================"<<std::endl;
//  double A=35.0;
//  double B=10.0;

//std::vector<double> info;
//info.push_back(A);
//info.push_back(B);
//info.push_back(2.0);

// std::vector<std::vector<double> > in;
//  std::vector<double> tmp;


//  tmp.push_back(113.00741);
//  tmp.push_back(28.231127);
//  in.push_back(tmp);
//  tmp[0]=113.008394;
//  tmp[1]=28.231119;
//  in.push_back(tmp);
//  tmp[0]=113.007902;
//  tmp[1]=28.231544;
//  in.push_back(tmp);
//  tmp[0]=113.008391;
//  tmp[1]=28.231166;
//  in.push_back(tmp);
  
//   tmp[0]=113.007693;
//  tmp[1]=28.23154;
//  in.push_back(tmp);
//  tmp[0]=113.008394;
//  tmp[1]=28.230808;
//  in.push_back(tmp);

//  std::vector< std::vector<double> > point;
//  tmp[0]=113.007599;
//  tmp[1]=28.230796;
//  point.push_back(tmp);
//   tmp[0]=113.007650;
//  tmp[1]=28.231470;
//  point.push_back(tmp);
//   tmp[0]=113.007875;
//  tmp[1]=28.231381;
//  point.push_back(tmp);
//   tmp[0]=113.008144;
//  tmp[1]=28.231023;
//  point.push_back(tmp);
//   tmp[0]=113.008216;
//  tmp[1]=28.23123;
//  point.push_back(tmp);
//   tmp[0]=113.007397;
//  tmp[1]=28.231521;
//  point.push_back(tmp);
//  tmp[0]=113.008376;
//  tmp[1]=28.23156;
//  point.push_back(tmp);
//  tmp[0]=113.008376;
//  tmp[1]=28.231552;
//  point.push_back(tmp);
//  tmp[0]=113.008421;
//  tmp[1]=28.230423;
//  point.push_back(tmp);
//  tmp[0]=113.007437;
//  tmp[1]=28.230383;
//  point.push_back(tmp);
//  tmp[0]=113.007356;
//  tmp[1]=28.23154;
//  point.push_back(tmp);
//  tmp[0]=113.007365;
//  tmp[1]=28.230375;
//  point.push_back(tmp);
  
// std::vector <double> xx(warning_expel_model::groupDeterrence(A,B,3,in,point));
//   for(int i=0;i<xx.size();i++){
//  cout<<xx[i]<<endl;
//  }
//  std::vector <double> xxx(warning_expel_model::singleDeterrence(A,B,in,point));
//  for(int i=0;i<xxx.size();i++){
//  cout<<xxx[i]<<endl;
//  }


    std::cout<<"======= nearby recon ======"<<std::endl;
    double _w,_l;
    std::cout<<"heightest =  "<<nearby_reconnaissance_model::furthestHight(_l,_w)<<std::endl;
    std::cout<<"l_max: "<<_l<<"  w_max: "<<_w<<std::endl;
    nearby_reconnaissance_model::nearRconUnitState lt,rb;
    //std::cout<<nearby_reconnaissance_model::cameraRectangle(0,0,95,lt,rb)<<std::endl;
    //std::cout<<"left top : "<<lt.x<<"  rigth bottom : "<<rb.y<<std::endl;

    std::vector< std::vector<double> > points;
    std::vector< std::vector<double> > points_frame;
    std::vector<double> temp;
    temp.push_back(116.41349);
    temp.push_back(39.924761);
    points.push_back(temp);
    temp[0] = 116.417442;
    temp[1] = 39.924795;
    points.push_back(temp);
    //nearby_reconnaissance_model::theBestPositionsInLine(points,points_frame);
	return 0;
	
	
}
