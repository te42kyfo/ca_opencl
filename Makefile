CXX 	:= g++
CCFLAGS := -Wall -O3 -std=c++11
LDFLAGS := -lOpenCL

test: main.cpp Makefile
	$(CXX) $< -o $@ $(CCFLAGS) $(LDFLAGS)
