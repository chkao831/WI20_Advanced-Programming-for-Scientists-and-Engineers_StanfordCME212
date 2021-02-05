#include <cassert>
#include <vector>

/* Given a valid matrix (but not neccessarily square),
 * which is represented by a vector<vector<T>>.
 * This function return its transpose.
 */
template <typename T>
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& mat){
    
    unsigned row = static_cast<unsigned>(mat.size());
    unsigned col = static_cast<unsigned>(mat[0].size());
    assert(row == col);
    
    //initialize transpose matrix with same size
    std::vector<std::vector<T>> trans(row, std::vector<T>(row,0));
    for(unsigned r = 0; r < row; r++){
        for(unsigned c = 0; c < col; c++){
            trans[c][r] = mat[r][c];
        }
    }
    return trans;
}

