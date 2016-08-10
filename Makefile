CXX 	:= g++
CCFLAGS := -Wall -O3 -std=c++11
LDFLGAS := -lOpenCL

test: main.cpp Makefile
	$(CXX) $< -o $@ $(CCFLAGS) $(LDFLAGS)
