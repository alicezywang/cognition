# 编译与运行
1. pre-install
  - 安装cuda-9.0和对应版本cudnn
  - 安装认知域dependency(tensorrt,tensorflow等)
    - //PC端
      - $ cd third_party
      - $ git clone http://192.168.8.22/machinelearning/amd64.git
      - $ ./setup
    - //TX2端
      - $ cd third_party
      - $ git clone http://192.168.8.22/machinelearning/arm64.git
      - $ ./setup

2. 编译运行
  - $ cd ..
  - $ catkin_make
  - $ rosrun cognition_dev test_cognition_dev



# cognition-chang-log
## 2019-12-02
1. feat(third_party): add runtime-lib setup shell;
2. feat(micros): update new cognition code dir;

## 2019-11-26
1. feat(cognition_dev): add new BusType and new API;

## 2019-11-25
1. feat(cognition_dev): add cognition_softbus API;
2. feat(cognition_bus): add async_call()/call() in cognition_bus_base;
3. feat(bus/implements): update the bus implement according new cognition_bus_base;
4. feat(cognition_resource): add model_recommend package in algorithm_lib;
5. feat(cognition_resource): add ml_database package;

