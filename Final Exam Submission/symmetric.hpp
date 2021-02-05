#include <cassert>
#include <vector>

/* Given a valid matrix (but not neccessarily square),
 * which is represented by a vector<vector<T>>.
 * This function checks if it is symmetric.
 */
template <typename T>
bool isSymmetric(const std::vector<std::vector<T>>& matrix){
    unsigned row = static_cast<unsigned>(matrix.size());
    unsigned col = static_cast<unsigned>(matrix[0].size());
    assert(row == col);

    //initialize transpose matrix
    std::vector<std::vector<T>> trans(row,std::vector<T>(row,0));
    //add elements
    for(unsigned r = 0; r < row; r++){
        for(unsigned c = 0; c < col; c++){
            trans[c][r] = matrix[r][c];
        }
    }
    //see if matrix = matrix_transpose
    for(unsigned r = 0; r < row; r++){
        for(unsigned c = 0; c < col; c++){
            if(trans[r][c] != matrix[r][c]) return false;
        }
    }
    return true;
}
