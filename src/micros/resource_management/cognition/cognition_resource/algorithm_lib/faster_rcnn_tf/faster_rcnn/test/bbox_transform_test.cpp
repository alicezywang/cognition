#include <fast_rcnn/bbox_transform.h>
#ifdef EIGEN_USE_THREADS
#include "ThreadPool"
#endif

using namespace fast_rcnn;
using namespace std;


int main(int argc, char **argv)
{
  //sleep(3);
  Tensor2f ex_rois(4,4);
  Tensor2f gt_rois(4,4);
  Tensor2f boxes(4,4);
  Tensor2f deltas(4,4);

  ex_rois.setRandom();
  gt_rois.setRandom();
  boxes.setRandom();
  deltas.setRandom();

  Tensor1f im_shape(2);
  im_shape(0) = 4;
  im_shape(1) = 4;

  Tensor2f bbox_trans = bbox_transform(ex_rois, gt_rois);
  cout << "bbox_trans:" << endl << bbox_trans <<endl;

  Tensor2f bbox_trans_inv = bbox_transform_inv(boxes, deltas);
  cout << "bbox_trans_inv:" << endl << bbox_trans_inv <<endl;

  Tensor2f cliped_boxes = clip_boxes(boxes, im_shape);
  cout << "cliped_boxes:" << endl << cliped_boxes <<endl;
  return 0;
}
