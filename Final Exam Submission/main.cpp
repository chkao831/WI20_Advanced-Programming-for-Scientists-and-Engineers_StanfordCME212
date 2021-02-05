#include "MinStack.hpp"
#include "symmetric.hpp"
#include "transpose.hpp"
#include "reshape.hpp"
#include "intersection.hpp"

#include <cassert>
#include <iostream>
#include <vector>

int main() {

    //test default constructor
    MinStack<int> s;
    s.push(5);
    s.push(2);
    s.push(100);
    assert(s.size() == 3);
    assert(s.min() == 2);
    assert(s.top() == 100);
    s.pop();
    assert(s.size() == 2);
    assert(s.top() == 2);
    s.pop();
    assert(s.size() == 1);
    assert(s.min() == 5);

    std::vector<std::vector<int>> sym{{1, 2, 3},{2, 4, 6},{3, 6, 9}};
    std::cout << std::boolalpha << "Symmetric? " << isSymmetric(sym) << std::endl;


    std::cout << "==TRANSPOSE SECTION==" << std::endl;
    std::vector<std::vector<int>> trans{{1, 2, 3},{4, 5, 6},{7, 8, 9}};
    std::vector<std::vector<int>> returned_trans = transpose(trans);
    for (const auto& vec : returned_trans){
        for (auto i : vec){
            std::cout << i << " ";
        } std::cout << " " << std::endl;
    } std::cout << " " << std::endl;

    std::cout << "==RESHAPE SECTION==" << std::endl;
    std::vector<std::vector<int>> res{{1, 2},{2, 4}};
    std::vector<std::vector<int>> reshaped = matrixReshape(res,1,4);
    for (const auto& vec : reshaped){
        for (auto i : vec){
            std::cout << i << " ";
        } std::cout << " " << std::endl;
    } std::cout << " " << std::endl;

    std::cout << "==INTERSECTION SECTION==" << std::endl;
    std::vector<int> first{19,7,11,13,3,5,17};
    std::vector<int> second{11,15,17,19,1,3,5,7,9};
    std::vector<int> third{1,4,8,5,7};
    std::vector<int> inter = intersection(first, second, third);
    for(const auto ele : inter) std::cout << ele << " "; //should be 7 5
    std::cout << " " << std::endl;

}
