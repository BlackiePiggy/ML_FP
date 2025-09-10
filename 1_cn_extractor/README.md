使用方法：

1. 将所有待处理的.o文件放在一个文件夹下；
2. 修改config.ini中的输入文件路径和输出路径；
3. 直接编译运行整个工程即可

terminal执行：
```
# 进入项目根目录
cd 1_cn_extractor

rd /s /q build_mingw # 删除之前的build cache

cmake -S . -B build_mingw -G Ninja ^
  -DCMAKE_C_COMPILER="D:/softwares/CLion/bin/mingw/bin/gcc.exe" ^
  -DCMAKE_CXX_COMPILER="D:/softwares/CLion/bin/mingw/bin/g++.exe"
  
cmake --build build_mingw

# 运行程序（可能需要配置文件 config.ini 在同一目录）
.\build_mingw\snr_extract.exe
```