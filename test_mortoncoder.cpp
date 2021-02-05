#include "MortonCoder.hpp"
#include <bitset>

/** @brief A test script for MortonCode.hpp. */
int main(){
    uint32_t test0 = detail::spread_bits(0);
    assert(test0 == 0);
    uint32_t test1 = detail::spread_bits(1); //1
    assert(test1 == 1); //001
    uint32_t test2 = detail::spread_bits(2); //10
    assert(test2 == 8); //1000
    uint32_t test8 = detail::spread_bits(8); //1000
    assert(test8 == 512); //1000000000
    uint32_t test9 = detail::spread_bits(9); //1001
    assert(test9 == 513); //1000000001
    uint32_t test10 = detail::spread_bits(10); //1010
    assert(test10 == 520); //1000001000
    uint32_t test99 = detail::spread_bits(99); //1100011
    assert(test99 == 294921); //1001000000000001001 
    uint32_t test1023 = detail::spread_bits(1023);
    assert(test1023 == 153391689);
//    std::bitset<32> x {test1023};
//    std::cout << x << std::endl; //made sure it's 00001001001001001001001001001001
    uint32_t test1024 = detail::spread_bits(1024);
    assert(test1024 == 0);
//    std::bitset<32> y {test1024};
//    std::cout << y << std::endl; //made sure it's 00000000000000000000000000000000

//    uint32_t test1024compact = detail::compact_bits(test1024);
//    assert(test1024compact == 0);
//    std::bitset<32> z {test1024compact};
//    std::cout << z << std::endl; //made sure it's 00000000000000000000000000000000
    
    for(unsigned int k = 0; k <= 1023; k++){
        bool correct = detail::compact_bits(detail::spread_bits(k)) == k;
        assert(correct);
    }

    std::cout << "All tests passed." << std::endl;

    return 0;
}

