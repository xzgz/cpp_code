CXX := g++
PYTHON_INCLUDE := $(shell python3 -m pybind11 --includes)
CXXFLAGS := -O3 -Wall -shared -std=c++11 -fPIC

TARGET := custom_converter$(shell python3-config --extension-suffix)

all: $(TARGET)

$(TARGET): custom_converter.cpp
	$(CXX) $(CXXFLAGS) $(PYTHON_INCLUDE) $< -o $@

clean:
	rm -f *.so *.o
