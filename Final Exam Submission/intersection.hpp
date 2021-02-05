#include <algorithm>
#include <vector> 

/**
 * Given three vector<T>s,  this function computes their intersection,
 * returning a vector<T> that contains elements that are common to all three input vectors.
 * reference: https://stackoverflow.com/questions/19483663/vector-intersection-in-c
 */
template<typename T>
std::vector<T> intersection(const std::vector<T> &a,
                            const std::vector<T> &b,
                            const std::vector<T> &c){
    //stable_sort sorts in-place, so deep copy each vector then sort
    std::vector<T> A = a;
    std::stable_sort(A.begin(), A.end());

    std::vector<T> B = b;
    std::stable_sort(B.begin(), B.end());

    std::vector<T> C = c;
    std::stable_sort(C.begin(), C.end());

    //a b intersection
    std::vector<T> AB;
    std::set_intersection(A.begin(),A.end(),
                          B.begin(),B.end(),
                          back_inserter(AB));
    //b c intersection
    std::vector<T> BC;
    std::set_intersection(B.begin(),B.end(),
                          C.begin(),C.end(),
                          back_inserter(BC));
    
    //ab bc intersection is abc intersection
    std::vector<T> result;
    std::set_intersection(AB.begin(),AB.end(),
                          BC.begin(),BC.end(),
                          back_inserter(result));
    
    //the output should be sorted in descending order
    std::reverse(result.begin(),result.end());

    return result;
}
