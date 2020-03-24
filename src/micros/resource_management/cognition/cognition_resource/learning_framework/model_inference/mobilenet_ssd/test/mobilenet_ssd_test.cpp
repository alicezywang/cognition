#include <mobilenet_ssd/mobilenet_ssd_model.h>
#include <mobilenet_ssd/mobilenet_ssd_dnn_model.h>

#define MOBILE_NET_SSD_DNN 1

using namespace machine_learning;

int main()
{
  const std::string image_path = "result-Img/source.jpg";
  cv::Mat test_image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
#if MOBILE_NET_SSD_DNN
  MobileNetSSDDNNModel model;
#else
  MobileNetSSDModel model;
#endif
  machine_learning::ResultType result = model.evaluate(test_image);
  for(int i = 0; i < result.vec_classes.size(); i++)
  {
    float confidence = result.vec_scores[i];
    int xLeftBottom = static_cast<int>(result.vec_bboxes[i][0]);
    int yLeftBottom = static_cast<int>(result.vec_bboxes[i][1]);
    int xRightTop = static_cast<int>(result.vec_bboxes[i][2]);
    int yRightTop = static_cast<int>(result.vec_bboxes[i][3]);
    std::ostringstream ss;
    ss << confidence;
    cv::String conf(ss.str());
    Rect object((int)xLeftBottom, (int)yLeftBottom, (int)(xRightTop - xLeftBottom), (int)(yRightTop - yLeftBottom));
    rectangle(test_image, object, Scalar(0, 255, 0), 2);
    String label = String(result.vec_classes[i]) + ": " + conf;
    std::cout << label << std::endl;
    int baseLine = 0;
    cv::Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    rectangle(test_image, cv::Rect(Point(xLeftBottom, yLeftBottom - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), Scalar(0, 255, 0), CV_FILLED);
    putText(test_image, label, cv::Point(xLeftBottom, yLeftBottom), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
  }
  cv::imshow("image", test_image);
  cv::waitKey(0);
  return 0;
}
