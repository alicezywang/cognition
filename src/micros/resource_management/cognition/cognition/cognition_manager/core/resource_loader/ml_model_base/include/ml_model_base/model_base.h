#ifndef MODEL_BASE_H
#define MODEL_BASE_H

#include <opencv2/core/core.hpp>
#include <vector>

namespace machine_learning
{
/**
 * @brief The ResultType struct is for the result of machine learning on image.
 */
struct ResultType
{
  std::vector<std::string> vec_classes;                    //class names for one image
  std::vector<float> vec_scores;                           //confidences of each class for one image
  std::vector<std::vector<float> > vec_bboxes;             //bounding boxes of each class for one image
  std::vector<std::vector<std::vector<bool> > > vec_masks; //mask of each class for one image
};
/**
 * @brief The ModelBase class
 */
class ModelBase
{
public:

  virtual void train(int start, int end) = 0;
  virtual ResultType evaluate(cv::Mat &test_image) = 0;
  virtual void batch_evaluate(int size) = 0;
};

}

#endif
