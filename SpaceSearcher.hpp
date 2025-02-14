#pragma once
/** @file SpaceSearcher.hpp
 * @brief Define the SpaceSearcher class for making efficient spatial searches.
 */

#include "thrust/execution_policy.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include "thrust/tuple.h"

#include "CME212/Util.hpp"
#include "CME212/Point.hpp"
#include "CME212/BoundingBox.hpp"
#include "MortonCoder.hpp"

/** @class SpaceSearcher
 * @brief Class for making spatial searches, which uses the MortonCoder
 *        class as a backend.
 *
 * Given a range of data items and a mapping between these
 * data items and Points, the SpaceSearcher class can be used to quickly
 * iterate over data items which are contained (or almost conatined) inside
 * any given BoundingBox.
 *
 * See "space_search_test.cpp" for a usage example.
 */
template <typename T, int L = 7>
class SpaceSearcher
{
    private:
        // Implementation types

        /** The number of levels in the MortonCoder. This controls the "accuracy" of
         * the searching iterators (10 -- high accuracy, 1 -- low accuracy), but
         * can also impose more expensive searching.
         */
        static constexpr int NumLevels = L;
        /** Type of MortonCoder. */
        using MortonCoderType = MortonCoder<NumLevels>;
        /** Type of the Morton codes. */
        using code_type = typename MortonCoderType::code_type;
        /** Helper struct: (code_type,T) pair */
        struct morton_pair;

    public:

        ////////////////////////////////////
        // TYPE DEFINITIONS AND CONSTANTS //
        ////////////////////////////////////

        /** The type of values that are stored and returned in the spacial search */
        using value_type = T;

        /** Type of iterators, which iterate over items inside a BoundingBox. */
        struct NeighborhoodIterator;

        /** Synonym for NeighborhoodIterator */
        using iterator       = NeighborhoodIterator;
        using const_iterator = NeighborhoodIterator;

    public:

        /**
         * @brief A helper functor that maps a MortonCodeType to
         * a unary function that maps points to Morton codes.
         */
        struct Point2MC {
            
            /** constructor */
            Point2MC(MortonCoderType mc)
                : mc_(mc) {}
            
            code_type operator()(Point p) {
                // MortonCoderType.code(Point) returns code_type
                return mc_.code(p);
            }

         private:
          MortonCoderType mc_; // MortonCoder instance
        };//end Point2MC

        
        /////////////////
        // CONSTRUCTOR //
        /////////////////

        /** @brief SpaceSearcher Constructor.
         *
         * For a range of data items of type @a T given by [@a first, @a last)
         * and a function object @a t2p that maps between data items and @a Points, we
         * arrange the data along a space filling curve which allows all of the data
         * contained withing a given bounding box to be iterated over in less than
         * linear time.
         *
         * @param[in] bb      The "parent" bounding box which this SpaceSearcher
         *                      functions within. All data and queries should
         *                      be contained within.
         * @param[in] t_begin Iterator to first data item of type @a T.
         * @param[in] t_end   Iterator to one past the last data item.
         * @param[in] t2p     A functor that maps data items to @a Points.
         *                      Provides an interface equivalent to
         *                        Point t2p(const T& t) const
         *
         * @pre For all i in [@a first,@a last), @a bb.contains(@a t2p(*i)).
         *
         * This is a delegating constructor, delegating to the SpaceSearcher constructor below.
         */
        template <typename TIter, typename T2Point>
            SpaceSearcher(const Box3D& bb,
                    TIter first, TIter last, T2Point t2p)
            : SpaceSearcher(bb,
                            first,
                            last,
                            thrust::make_transform_iterator(first,t2p),
                            thrust::make_transform_iterator(last,t2p)){
            }


        /** @brief SpaceSearcher Constructor.
         *
         * For a range of data items of type @a T given by [@a tfirst, @a tlast)
         * and a corresponding range of @a Points given by [@a pfirst, @a plast),
         * we arrange the data along a space filling curve which allows all of the
         * data contained withing a given bounding box to be iterated over in less
         * than linear time.
         *
         * @param[in] bb      The "parent" bounding box which this SpaceSearcher
         *                      functions within. All data and queries should
         *                      be contained within.
         * @param[in] tfirst  Iterator to first data item of type T.
         * @param[in] tlast   Iterator to one past the last data item.
         * @param[in] pfirst  Iterator to first Point corresponding to the position
         *                      of the first data item, *tfirst.
         * @param[in] tlast   Iterator to one past the last @a Point.
         *
         * @pre std::distance(tfirst,tlast) == std::distance(pfirst,plast).
         * @pre For all i in [@a pfirst,@a plast), bb.contains(*i).
         */
        template <typename TIter, typename PointIter>
            SpaceSearcher(const Box3D& bb,
                    TIter tfirst, TIter tlast,
                    PointIter pfirst, PointIter plast)
                : mc_(bb) {

                    // transform_iterator<functor,iterator,value_type>
                    using MCIter = thrust::transform_iterator<Point2MC,PointIter,code_type>;
                    // specifying MCIter ranges
                    MCIter mc_begin = MCIter(pfirst,Point2MC(mc_));
                    MCIter mc_end = MCIter(plast,Point2MC(mc_));
                
                    // zip_iterator is an iterator which represents a pointer into a range of tuples
                    // whose elements are themselves taken from a tuple of input iterators
                    using IteratorTuple = thrust::tuple<MCIter,TIter>;
                    using ZipIter = thrust::zip_iterator<IteratorTuple>;
                    // use the make_zip_iterator function with the make_tuple function
                    // to avoid explicitly specifying the type of the zip_iterator
                    ZipIter first = thrust::make_zip_iterator(thrust::make_tuple(mc_begin,tfirst));
                    ZipIter last = thrust::make_zip_iterator(thrust::make_tuple(mc_end,tlast));
                    
                    // use range constructor of std::vector to initialize the data
                    z_data_ = std::vector<morton_pair>(first, last);
                    
                    // Use (optionally) parallel algorithms to sort the data by morton codes
                    thrust::sort(thrust::omp::par,
                                 z_data_.begin(),
                                 z_data_.end(),
                                 [](morton_pair mp1, morton_pair mp2){ return mp1.code_ < mp2.code_; });
                }//end constructor

        ///////////////
        // Accessors //
        ///////////////

        /** The bounding box this SpaceSearcher functions within. */
        Box3D bounding_box() const {
            return mc_.bounding_box();
        }

        //////////////
        // Iterator //
        //////////////

        /** @class SpaceSearcher::NeighborhoodIterator
         * @brief NeighborhoodIterator class for data items. A forward iterator.
         *
         * Iterates over data items of type @a T contained
         * within epsilon of a given bounding box.
         */
        struct NeighborhoodIterator {
            using value_type        = T;
            using pointer           = T*;
            using reference         = T&;
            using difference_type   = std::ptrdiff_t;
            using iterator_category = std::forward_iterator_tag;

            // Default constructor
            NeighborhoodIterator() = default;

            // Iterator operators
            const value_type& operator*() const {
                return (*i_).value_;
            }
            NeighborhoodIterator& operator++() {
                ++i_; fix();
                return *this;
            }
            bool operator==(const NeighborhoodIterator& other) const {
                return i_ == other.i_;
            }
            bool operator!=(const NeighborhoodIterator& other) const {
                return !(*this == other);
            }

            private:
            friend SpaceSearcher;
            using MortonIter = typename std::vector<morton_pair>::const_iterator;
            // RI: i_ == end_ || MortonCoderType::is_in_box(*i_, min_, max_)
            MortonIter i_, end_;
            code_type min_, max_;
            NeighborhoodIterator(MortonIter i, MortonIter end,
                    code_type min, code_type max)
                : i_(i), end_(end), min_(min), max_(max) {
                    fix();
                }
            // @post RI
            void fix() {
                while (i_ < end_) {
                    code_type c = MortonCoderType::advance_to_box(*i_, min_, max_);
                    if (c == *i_) break;
                    i_ = std::lower_bound(i_, end_, c);
                }
            }
        };

        /** Iterator to the first item contained
         *   within some epsilon of a bounding box.
         * @param bb The bounding box to iterate over.
         * @pre bounding_box.contains(bb)
         */
        const_iterator begin(const Box3D& bb) const {
            assert(bounding_box().contains(bb));
            code_type morton_min = mc_.code(bb.min());
            code_type morton_max = mc_.code(bb.max());
            auto mit_end = std::lower_bound(z_data_.begin(), z_data_.end(), morton_max);
            return NeighborhoodIterator(z_data_.begin(), mit_end, morton_min, morton_max);
        }

        /** Iterator to one-past-the-last item contained
         *   within some epsilon of a bounding box.
         * @param bb The bounding box to iterate over.
         * @pre bounding_box.contains(bb)
         */
        const_iterator end(const Box3D& bb) const {
            assert(bounding_box().contains(bb));
            code_type morton_min = mc_.code(bb.min());
            code_type morton_max = mc_.code(bb.max());
            auto mit_end = std::lower_bound(z_data_.begin(), z_data_.end(), morton_max);
            return NeighborhoodIterator(mit_end, mit_end, morton_min, morton_max);
        }

    private:

        // MortonCoder instance associated with this SpaceSearcher.
        MortonCoderType mc_;

        // A (code_type,value_type) pair that can be used as a MortonCode
        struct morton_pair {
            code_type code_;
            value_type value_;
            // Cast operator to treat a morton_pair as a code_type in std::algorithms
            operator const code_type&() const {
                return code_;
            }
            
            morton_pair(thrust::tuple<code_type, T> pair)
                : code_(thrust::get<0>(pair)),
                  value_(thrust::get<1>(pair)) {}
        }; // end morton_pair struct

        // Pairs of Morton codes and data items of type T.
        // RI: std::is_sorted(z_data_.begin(), z_data_.end())
        std::vector<morton_pair> z_data_;
};
