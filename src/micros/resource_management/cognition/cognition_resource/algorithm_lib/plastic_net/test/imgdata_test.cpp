#include <iostream>
#include <ros/package.h>
#include <plastic_net/imgdata.h>
#include <plastic_net/random.h>

int main()
{
  ImgData imgdata(ros::package::getPath("ml_datasets") + "/train_datasets/miniimagenet/train/");
  std::multimap<std::string, std::vector<std::string> > ClassInfoImg;

  ClassInfoImg = imgdata.GetClassinfo();
  std::cout<<imgdata.ImgPathToClassInfo.size()<<std::endl;

  std::multimap<std::string, std::vector<std::string> >::iterator it;
  for(it=imgdata.ImgPathToClassInfo.begin();it!=imgdata.ImgPathToClassInfo.end();it++)
  /*for(auto it:imgdata.ImgPathToClassInfo)
  {
    cout<<"-------"<<it->second.size()<<endl
        <<it->second[0]<<endl//className
        <<it->second[1]<<endl//language
        <<it->second[2]<<endl//character
        <<it->second[3]<<endl;//imgName
        cout<<it->first<<endl;
  }*/
  std::cout<<imgdata.ImgClassinfo.size()<<std::endl;

  std::pair<int, int> range;
  range.first = 0;
  range.second = imgdata.ImgPathToClassInfo.size()-1;

  std::cout << tools::RandomIntTool(range) << std::endl;
  std::cout<<"size:"<<imgdata.filesnumber<<std::endl;
  std::cout<<"size:"<<imgdata.fixedDate.size()<<std::endl;

  for(int i=0;i<imgdata.fixedDate.size();i++)
  {
    std::cout<<"path:"<<imgdata.fixedDate[i].size()<<std::endl;
  }
  return 0;
}
