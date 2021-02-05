#ifndef MinStack_hpp
#define MinStack_hpp

#include <iostream>

/**
 * @class MinStack
 * @brief Without using the standard library (or any third-party libraries),
 * MinStack is backed by a built-in array that grows dynamically based on the needs
 * of the user of the class. This MinStack requires no more than O(n) storage and
 * supports a min operation which returns the minimum element contained within
 * the stack in O(1) time.
 * @ref https://www.geeksforgeeks.org/design-and-implement-special-stack-data-structure/
 */
template <typename T>
class MinStack {

    public:

        int size_of_stack;
        int current_stack_limit; //current max limit of this stack
        T* ordinary_stack; //this stack
        T* min_stack; //an auxiliary stack to store minimum values


        /* Default Constructor */
        MinStack(){
            this->size_of_stack = 0;
            this->current_stack_limit = 1; //by default, limit is 1
            this->ordinary_stack = new T[current_stack_limit];
            this->min_stack = new T[current_stack_limit];
        }

        /* Copy Constructor */
        MinStack(MinStack& copiedMS){
            this->size_of_stack = copiedMS.size_of_stack;
            this->current_stack_limit = copiedMS.current_stack_limit;

            this->ordinary_stack = new T[current_stack_limit];
            this->min_stack = new T[current_stack_limit];
            for(int k = 0; k < size_of_stack; k++){
                this->ordinary_stack[k] = copiedMS.ordinary_stack[k];
                this->min_stack[k] = copiedMS.min_stack[k];
            }
        }

        /* Copy Assignment */
        MinStack& operator=(MinStack copiedMS){
            MinStack copied_item{copiedMS}; //using copy constructor to make copy
            std::swap(copied_item, *this); //Swap copied_item’s representation into *this’s.
            return *this;
        }

        /* Move Constructor */
        MinStack(MinStack&& fromMS)
            : size_of_stack(fromMS.size_of_stack),
            current_stack_limit(fromMS.current_stack_limit),
            ordinary_stack(fromMS.ordinary_stack),
            min_stack(fromMS.min_stack){
                fromMS.ordinary_stack = nullptr;
                fromMS.min_stack = nullptr;
            }

        /* Move Assignment */
        MinStack& operator=(MinStack&& fromMS){
            //if the input minStack is exactly this, do nothing and return the stack
            if(this == &fromMS){
                return *this;
            }

            //otherwise, it's different, swap and delete
            swap(size_of_stack, fromMS.size_of_stack);
            swap(current_stack_limit, fromMS.current_stack_limit);

            delete[] ordinary_stack;
            ordinary_stack = fromMS.ordinary_stack;
            fromMS.ordinary_stack = nullptr;

            delete[] min_stack;
            min_stack = fromMS.min_stack;
            fromMS.min_stack = nullptr;

            return *this;
        }

        /**
         * @brief Add an element to the stack in O(1).
         * @param[in] ele the element to be added to the stack.
         * @post new size_of_stack == old size_of_stack + 1
         * @post size_of_stack <= current_stack_limit
         * @post min_stack.top() == ordinary_stack.min()
         * @post ordinary_stack.top() == ele
         *
         * Complexity: O(1) amortized.
         */
        void push(T ele){

            //doubling strategy
            if(size_of_stack == current_stack_limit){
                //double capacity
                current_stack_limit *= 2;
                T* new_ordinary_stack = new T[current_stack_limit]();
                T* new_min_stack = new T[current_stack_limit]();

                for(int k = 0; k < size_of_stack; k++){
                    std::swap(new_ordinary_stack[k], ordinary_stack[k]);
                    std::swap(new_min_stack[k], min_stack[k]);
                }
                delete[] ordinary_stack;
                ordinary_stack = new_ordinary_stack;
                new_ordinary_stack = nullptr;

                delete[] min_stack;
                min_stack = new_min_stack;
                new_min_stack = nullptr;

            }//end doubling

            //If the stack is empty, add the item directly
            if(size_of_stack == 0){
                ordinary_stack[0] = ele;
                min_stack[0] = ele;
            } else {
                //otherwise, not empty and not reaching limit (by doubling strategy)
                //add the element and update sizes
                T current_min = min_stack[size_of_stack - 1];
                ordinary_stack[size_of_stack] = ele;
                if(ele < current_min){
                    min_stack[size_of_stack] = ele;
                } else {
                    min_stack[size_of_stack] = current_min;
                }
            }
            size_of_stack += 1;
            return;
        }//end push

        /**
         * @brief Remove the most recently inserted element of stack.
         * @pre size_of_stack > 0
         * @post new size_of_stack == old size_of_stack - 1
         * @post size_of_stack <= current_stack_limit
         * @post min_stack.top() == ordinary_stack.min()
         *
         * Complexity: O(1) amortized.
         */
        void pop() {
            assert(size_of_stack > 0);
            size_of_stack -= 1;
        }

        /**
         * @brief Return the value of the most recently inserted element onto the stack.
         * @pre size_of_stack > 0
         *
         * Complexity: O(1) amortized.
         */
        T top() {
            assert(size_of_stack > 0);
            return ordinary_stack[size_of_stack - 1];
        }

        /**
         * @brief Return the current number of elements held in the container.
         *
         * Complexity: O(1) amortized.
         */
        int size(){
            return size_of_stack;
        }

        /**
         * @brief Return the minimum value held by the stack, if it is not empty, in O(1) time.
         * @pre size_of_stack > 0
         *
         * Complexity: O(1) amortized.
         */
        T min(){
            assert(size_of_stack > 0);
            return min_stack[size_of_stack - 1];
        }

}; //end class MinStack

#endif
