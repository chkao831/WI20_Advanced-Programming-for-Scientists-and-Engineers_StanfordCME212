#include <vector>

/**
 * Given a valid matrix (but not neccessarily square), represented by a
 * vector<vector<T>>, as well as two positive integers r and c which
 * respectively represent the row number and column number reshaped matrix,
 * perform the reshape operation by row raversing and return the new
 * reshaped matrix.
 */
template <typename T>
std::vector<std::vector<T>> matrixReshape(const std::vector<std::vector<T>>& mat,
                                          std::size_t r,
                                          std::size_t c){
    // If the reshape parameters r and c do not conform to a valid reshape operation,
    // should return the original matrix
    if (r * c != (mat.size() * mat[0].size())){
        return mat;
    }
    
    // possible to reshape
    // declare result matrix
    std::vector<std::vector<T>> result(r, std::vector<T>());

    //row traversing
    for (std::size_t i = 0; i < r; i++){
        for (std::size_t j = 0; j < c; j++){
            result[i].push_back(mat[(i*c + j)/mat[0].size()][(i*c + j)%mat[0].size()]);
        }
    }
    
    //make sure reshaped properly
    assert(result.size() == r);
    assert(result[0].size() == c);
    
    return result;
}

