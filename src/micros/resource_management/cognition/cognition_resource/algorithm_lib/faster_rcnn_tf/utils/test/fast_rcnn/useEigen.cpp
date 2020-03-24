#include <vector>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;

void eigen_test(){
//详细介绍: https://github.com/PaddlePaddle/Paddle/wiki/A-Survey-and-Taxonomy-of-Eigen
//中文介绍: https://blog.csdn.net/hjimce/article/details/71710893
    vector<int> vec_int{ 1,2,3,4,5,6,7,8,9 };
    Eigen::Array<int, 3, 3> arr_3(vec_int.data());
    cout << arr_3 << endl;


//example 1 :创建，赋值，访问　<<<< 运算, 切分
//1.创建  Eigen::Tensor会分配内存
    Eigen::Tensor<double,4> T(1,2,3,4); //创建一个4维数组,形状为(1,2,3,4)
    int size = T.dimension(0);
    int rows = T.dimension(1);
    int cols = T.dimension(2);
    int channels = T.dimension(3);
//2.创建固定大小tensor: Create a 4 x 3 tensor of floats.
    Eigen::TensorFixedSize<float, Eigen::Sizes<4, 3>> t_4x3;

//3.赋值/初始化  Eigen::TensorMap不分配内存
    Eigen::TensorMap<Eigen::Tensor<int, 4>> t_4d(vec_int.data(), 3, 3, 1, 1);
        //该构造函数并没有在内存中另外拷贝一份data中的数据，而仅仅是数据指针映射，来创建Eigen::Tensor;
        //所以，一旦构造，该tensor矩阵是大小不可变的;
        //默认为列存储，增加flag(Eigen::RowMajor),变为行存储;
        //列存储运算速度更快;
    cout << t_4d << endl;

//4.元素访问
    t_4d(2,2,0,0) = 99; //()方法访问
    cout << t_4d << endl;
    cout << t_4d(1,1,0,0) << endl;

//5.auto自动类型特殊功能：auto只能用于非数值访问表达式，延迟计算，这个类似于深度学习常用库中的符号编程
    Eigen::Tensor<float, 3> t1(1,2,3);
    Eigen::Tensor<float, 3> t2(1,2,3);
    Eigen::Tensor<float, 3> t3(t1+t2);//t3 = t1 + t2
    auto t4 = t1 + t2;
    cout << "延时计算:" << t3(0, 0, 0) << endl; // OK prints the value of t1(0, 0, 0) + t2(0, 0, 0)

    //cout << t4(0, 0, 0);  // Compilation error!
        Eigen::Tensor<float, 3> result = t4;  // Could also be: result(t4);这样就能获取t4中的数值了
        cout << "延时计算:" <<  result(0, 0, 0);
    //所以如果希望矩阵经过一些列的计算后，到最后才获取具体的结果，可以采用auto;
    // Another way, exactly as efficient as the previous one:
    Eigen::Tensor<float, 3> result1 = ((t1 + t2) * 0.2f).exp(); //同样可以延迟计算的效果

//6.auto符号表达式效率：
    //Assignment to a Tensor, TensorFixedSize, or TensorMap. 将 auto 对象制定给 x,x,or x
    //Use of the eval() method. 计算得到中间值,避免重复计算
    //Assignment to a TensorRef.

//7.减少不必要的计算，采用符号编程，计算指定元素：TensorRef。
    //有的时候，我们并不需要计算一整个输出矩阵，可能我们仅仅想要计算矩阵某个元素的数值而已，
    //如果一整个矩阵计算，然后再拿出具体元素，无疑浪费不必要的计算。
    // Create a TensorRef for the expression.  The expression is not evaluated yet.
    Eigen::TensorRef<Eigen::Tensor<float, 3> > ref = ((t1 + t2) * 0.2f).exp();

    // Use "ref" to access individual elements.  The expression is evaluated on the fly.
    float at_0 = ref(0, 0, 0); //不计算整个矩阵,只进行局部计算
    cout << ref(0, 1, 0) <<endl;
    //这个类似于稀疏矩阵，如果你要获取一整个矩阵，建议不要用这个，效率反而更低。

//8.硬件、多线程、指令集等加速设置devices：在默认情况下，是采用cpu单线程，比如下面的代码：
    Eigen::Tensor<float, 2> a(3, 4);
    Eigen::Tensor<float, 2> b(3, 4);
    Eigen::Tensor<float, 2> c = a + b;

    //此时C的计算，默认是cpu 单线程。可以通过设置device，指定运行设备：
    Eigen::DefaultDevice my_device;
    c.device(my_device) = a + b;
    //device可选参数：DefaultDevice, ThreadPoolDevice 、GpuDevice三个类的对象。设置device，必须知道c的大小(shape)。
    cout << "compute with device:" <<endl<< c <<endl;

    //采用多线程，线程池： 需要安装时的编译选项支持
    // Create the Eigen ThreadPoolDevice.
    //Eigen::ThreadPool tp(4);
    //Eigen::ThreadPoolDevice my_device1(&tp, tp.NumThreads() /* number of threads to use */);
    //Eigen::Tensor<float, 2> c1(30, 50);
    // Now just use the device when evaluating expressions.
    //c1.device(my_device1) = a.contract(b, dot_product_dims); //contract是Eigen的一个方法，表示矩阵相乘

//9.一些常用的函数API：
    // Create 2 matrices using tensors of rank 2
    Eigen::Tensor<int, 2> aa(2, 3);
    aa.setValues({{1, 2, 3}, {6, 5, 4}});
    Eigen::Tensor<int, 2> bb(3, 2);
    bb.setValues({{1, 2}, {4, 5}, {5, 6}});

    // Compute the traditional matrix product
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
    Eigen::Tensor<int, 2> AB = aa.contract(bb, product_dims); //Contraction axes must be same size
    cout << "AB:" << endl << AB <<endl;
    // Compute the product of the transpose of the matrices
    Eigen::array<Eigen::IndexPair<int>, 1> transposed_product_dims = { Eigen::IndexPair<int>(0, 1) };
    Eigen::Tensor<int, 2> AtBt = aa.contract(bb, transposed_product_dims);
    cout << "AtBt:" << endl << AtBt <<endl;

    //维度相关
    Eigen::Tensor<double, 3> epsilon(3,3,3);
    cout << "Dims: " << epsilon.NumDimensions <<endl; //矩阵的维度
    epsilon.setZero();
    epsilon(0,1,2) = 1;
    epsilon(1,2,0) = 1;
    epsilon(2,0,1) = 1;
    epsilon(1,0,2) = -1;
    epsilon(2,1,0) = -1;
    epsilon(0,2,1) = -1;
    // dimensionalities
    assert(epsilon.dimension(0) == 3);//获取指定维度大小
    assert(epsilon.dimension(1) == 3);
    assert(epsilon.dimension(2) == 3);
    auto dims = epsilon.dimensions(); //矩阵形状
    assert(dims[0] == 3);
    assert(dims[1] == 3);
    assert(dims[2] == 3);
    cout << "Size: " << epsilon.size()<<endl; //获取矩阵元素总个数

//10.矩阵初始化API
    //1、所有元素初始化：setConstant(const Scalar& val)，用于把一个矩阵的所有元素设置成一个指定的常数。
    Eigen::Tensor<float, 2> a_float(2, 3);
    a_float.setConstant(12.3f);
    cout << "Constant: " << endl << a_float << endl << endl;
    Eigen::Tensor<string, 2> a_string(2, 3);
    a_string.setConstant("yolo");
    cout << "String tensor: " << endl << a_string << endl << endl;

    //2、全部置零：setZero()

    //3、从列表、数据初始化：setValues({..initializer_list})
    a_float.setValues({{0.0f, 1.0f, 2.0f}, {3.0f, 4.0f, 5.0f}});
    cout << "a" << endl << a_float << endl << endl;

    //如果给定的数组数据，少于矩阵元素的个数，那么后面不足的元素其值不变：
    Eigen::Tensor<int, 2> a_int(2, 3);
    a_int.setConstant(1000);
    a_int.setValues({{10, 20, 30}});
    cout << "a" << endl << a_int << endl << endl;

    //4、随机初始化：setRandom()
    a_float.setRandom();
    cout << "Random: " << endl << a_float << endl << endl;
    //当然也可以设置指定的随机生成器，类似于python 的 random seed。也可以选择初始化方法：
      //UniformRandomGenerator
      //NormalRandomGenerator

    //5、数据指针：Scalar* data() and const Scalar* data() const，一般用于与其它库、类型数据转换的时候使用，比如opencv mat类型等
    float* a_data = a_float.data(); //用a_data 代表 a_float的数据指针
    a_data[0] = 123.45f;
    cout << "a(0, 0) = 123.45f : " << endl << a_float << endl<< endl; //=> a(0, 0): 123.45

    //6、其他
    Eigen::Tensor<float, 2> a_const(2, 3);
    a_const.setConstant(1.0f);
    Eigen::Tensor<float, 2> b_const = a_const + a_const.constant(2.0f); ///.constant():不改变a_const
    Eigen::Tensor<float, 2> c_const = b_const * b_const.constant(0.2f);
    cout << "a_const" << endl << a_const << endl << endl;
    cout << "b_const" << endl << b_const << endl << endl;
    cout << "c_const" << endl << c_const << endl << endl;

    Eigen::Tensor<float, 2> a_random(2, 3);
    a_random.setConstant(1.0f);
    Eigen::Tensor<float, 2> b_random = a_random + a_random.random();///.random():不改变a_random
    cout << "b_random" << endl << b_random << endl << endl;

//11.其他重要函数
    //数据存储方式交换：swap_layout()
    Eigen::Tensor<float, 2, Eigen::ColMajor> col_major(2, 4);
    Eigen::Tensor<float, 2, Eigen::RowMajor> row_major(2, 4);
    cout <<"col_major.dimensions: "<<endl;
    cout << col_major.dimension(0) << endl ;
    cout << col_major.dimension(1) << endl << endl;
    cout <<"row_major.dimensions: "<<endl;
    cout << row_major.dimension(0) << endl ;//2
    cout << row_major.dimension(1) << endl << endl;//4
    row_major = col_major.swap_layout();  ///存储方式交换,改变了原维度的大小
    cout <<"swap_layout(): "<<endl;
    cout << row_major.dimension(0) << endl ; //4
    cout << row_major.dimension(1) << endl << endl;//2

    ///维度缩减: maximum(); sum(); mean();
    // Create a tensor of 2 dimensions
    Eigen::Tensor<int, 2> a_maximum(2, 3);  //2行3列, 默认列存储
    a_maximum.setValues({{5, 2, 3}, {1, 5, 4}});
    // Reduce it along the second dimension (1)...
    Eigen::array<int, 1> dims_max({0 /* dimension to reduce */}); ///0:代表行比较,得到列值最大; 1:代表列比较,取行最大
    cout << a_maximum.dimension(0) << endl;
    // ...using the "maximum" operator.
    // The result is a tensor with one dimension.  The size of
    // that dimension is the same as the first (non-reduced) dimension of a.
    Eigen::Tensor<int, 1> b_maximum = a_maximum.maximum(dims_max);
    cout << "a: " << endl << a_maximum << endl << endl;
    cout << "a_maximum: " << endl << b_maximum << endl << endl;

    ///切分操作: slice(); chip();
    Eigen::Tensor<int, 2> a_slice(4, 3);
    a_slice.setValues({{0, 100, 200}, {300, 400, 500},
                 {600, 700, 800}, {900, 1000, 1100}});
    Eigen::array<int, 2> offsets = {1, 0};
    Eigen::array<int, 2> extents = {2, 2};//(2,2)代表切分后的大小为矩阵
    Eigen::array<int, 2> extents_chip = {2, 1}; //(2,1)代表切分后的shape为向量
    Eigen::Tensor<int, 2> slice = a_slice.slice(offsets, extents); ///切分后的数据大小不能搞错,否则会报错
    Eigen::Tensor<int, 2> slice_chip = a_slice.slice(offsets, extents_chip); ///切分后的数据大小不能搞错,否则会报错
    cout << "a:" << endl << a_slice << endl << endl;
    cout << "slice:" << endl << slice << endl << endl;
    cout << "slice_chip:" << endl << slice_chip << endl << endl;

    Eigen::Tensor<int, 2> a_chip(4, 3);
    a_chip.setValues({{0, 100, 200}, {300, 400, 500},
                 {600, 700, 800}, {900, 1000, 1100}});
    Eigen::Tensor<int, 1> row_3 = a_chip.chip(2, 0);  ///第二个参数:维度index
    Eigen::Tensor<int, 1> col_3 = a_chip.chip(2, 1);  ///第一个参数:维度上哪个点的值
    cout << "a:" << endl << a_chip << endl << endl;
    cout << "row_3:" << endl << row_3 << endl << endl;
    cout << "col_3:" << endl << col_3 << endl << endl;

    //It is possible to assign values to a tensor chip since the chip operation is a lvalue.

    ///广播函数broadcast():用于张量在特定维度的复制
    Eigen::Tensor<int, 2> a_broadcast(2, 4);
    a_broadcast.setValues({{1, 2, 3, 4}, {5,6,7,8}});
    Eigen::Tensor<int, 2> b_broadcast(2, 1);
    b_broadcast.setValues({{1},{2}});

    Eigen::array<int, 2> bcast({1, 4});
    auto c_broadcast = b_broadcast.broadcast(bcast);
    auto d_broadcast = a_broadcast * c_broadcast; //*只能用于同型矩阵相乘，不同型要用contract
    cout << "c_broadcast = " <<endl;
    cout << c_broadcast <<endl;
    cout << "同型矩阵相乘(2,4)*(2,4) = " <<endl;
    cout << d_broadcast <<endl;
}

int main(int argc, char **argv)
{
//    ros::init(argc, argv, "myclass");

    eigen_test();
    return 0;
}
