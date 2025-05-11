# 编译器设置
CXX := g++
# CXXFLAGS := -Wall -Wextra -std=c++11 -O3
CXXFLAGS := -Wall -std=c++11 -O3
# CXXFLAGS := -O3 -Wall -shared -std=c++11 -fPIC

# 获取所有.cpp文件并生成对应的可执行文件名
SRCS := $(wildcard *.cpp)
EXECS := $(patsubst %.cpp,%,$(SRCS))

# 默认目标：编译所有可执行文件
all: $(EXECS)

# 模式规则：将每个.cpp文件编译为同名可执行文件
%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# 清理生成的可执行文件
clean:
	rm -f $(EXECS)

# 声明伪目标，防止与同名文件冲突
.PHONY: all clean
