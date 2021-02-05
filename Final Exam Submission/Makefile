CXX := g++
CXXFLAGS := -std=c++11 -Wall -Wextra -Wpedantic -Wconversion
TARGET := main
CPP := main.cpp 
HPP := symmetric.hpp transpose.hpp reshape.hpp intersection.hpp MinStack.hpp

main: $(CPP) $(HPP)
	$(CXX) -O3 $(CXXFLAGS) -o $(TARGET) $(CPP)

.PHONY: clean
clean: 
	rm *.o *~ $(TARGET)

