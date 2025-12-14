# body_animation



## Getting started
location: This folder should be put under 3dsmc-env/files/body_animation

### Check your folder structure
it should looks like this

/workspace/Exercises/
├── Libs/
│   ├── cnpy/             <-- 为这个项目新下载的
│   │   ├── cnpy.cpp
│   │   ├── cnpy.h
│   │   └── CMakeLists.txt 
│   ├── Eigen/            <-- 作业自带的
│   └── ...
├── Data/    
    ├── star_1_1/         <-- 放 smpl_model.npz
└── src/                  <-- 你的工作区 (改成其他名字也行)
    ├── build/
    ├── CMakeLists.txt    
    ├── main.cpp          <-- 入口
    ├── smpl_model.h     
    └── smpl_model.cpp    