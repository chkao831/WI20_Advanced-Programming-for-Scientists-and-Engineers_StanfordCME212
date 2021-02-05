#ifndef CME212_GRAPH_HPP
#define CME212_GRAPH_HPP

/**
 * @file Graph.hpp
 *
 * @author Chih-Hsuan (Carolyn) Kao
 * Contact: chkao831@stanford.edu
 * Date: Mar 13, 2020
 *
 * @brief This file, Graph.hpp, contains an undirected graph type, class Graph.
 */
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"

#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <vector>

#include "CME212/Util.hpp"
#include "CME212/Point.hpp"
/**
 * @class Graph
 * @brief A template for 3D undirected graphs.
 *
 * Users can add and retrieve nodes and edges. Edges are unique (there is at
 * most one edge between any pair of distinct nodes).
 */
template <typename V, typename E>
class Graph {

    /** Declaration of Internal Structs */
    private:
        struct NodeInfo;
        struct EdgeInfo;

    public:

        /** PUBLIC TYPE DEFINITIONS */

        /** Type of this graph. */
        using graph_type = Graph;

        /** Predeclaration of Node type. */
        class Node;
        /** Synonym for Node (following STL conventions). */
        using node_type = Node;
        /** Synonym for Node Value Type as Graph is a template */
        using node_value_type = V;

        /** Predeclaration of Edge type. */
        class Edge;
        /** Synonym for Edge (following STL conventions). */
        using edge_type = Edge;
        /** Synonym for Edge Value Type as Graph is a template */
        using edge_value_type = E;

        /** Type of node iterators, which iterate over all graph nodes. */
        struct NodeIterator;
        /** Synonym for NodeIterator */
        using node_iterator = NodeIterator;

        /** Type of edge iterators, which iterate over all graph edges. */
        class EdgeIterator;
        /** Synonym for EdgeIterator */
        using edge_iterator = EdgeIterator;

        /** Type of incident iterators, which iterate incident edges to a node. */
        class IncidentIterator;
        /** Synonym for IncidentIterator */
        using incident_iterator = IncidentIterator;

        /** Type of indexes and sizes.
          Return type of Graph::Node::index(), Graph::num_nodes(),
          Graph::num_edges(), and argument type of Graph::node(size_type) */
        using size_type = unsigned;
        using length_type = double;

        /** Invalid node index value */
        static size_type constexpr invalid_index = size_type(-1);

        /** CONSTRUCTORS AND DESTRUCTOR of Graph */

        /** Construct an empty graph. */
        Graph()
            : vec_nodeinfo_(),
            vec_activenodes_(),
            vec_edgeinfo_(),
            vec_activeedges_(),
            map_edge2nodes_(),
            map_rootnode2adjnodes_(),
            map_small2bignode_eid_(){}

        /** Default destructor */
        ~Graph() = default;

        //
        // NODE
        //
        /**
         * @class Graph::Node
         * @brief Class representing the graph's nodes.
         *
         * Node objects are used to access information about the Graph's nodes.
         */
        class Node : private totally_ordered<Node>{

            public:
                /*
                 *
                 * Valid nodes are obtained from the Graph class, but it
                 * is occasionally useful to declare an @i invalid node, and assign a
                 * valid node to it later. For example:
                 *
                 * @code
                 * Graph::node_type x;
                 * if (...should pick the first node...)
                 *   x = graph.node(0);
                 * else
                 *   x = some other node using a complicated calculation
                 * do_something(x);
                 * @endcode
                 */
                /** Construct an invalid node. */
                Node()
                    : ptr_to_graph_(nullptr),
                    node_uid_(invalid_index){
                    }

                /**
                 * @brief Return this node's position.
                 *
                 * The node's position is returned as a reference that CAN be updated.
                 */
                Point& position() {
                    return ptr_to_graph_->vec_nodeinfo_[node_uid_].position_;
                }

                /**
                 * @brief Return this node's position that cannot be updated.
                 */
                const Point& position() const {
                    return ptr_to_graph_->vec_nodeinfo_[node_uid_].position_;
                }

                /**
                 * @brief Return this node's index, a number in the range [0, graph_size).
                 */
                size_type index() const {
                    return ptr_to_graph_->vec_nodeinfo_[node_uid_].node_idx_;
                }

                /**
                 * @brief This is a getter to retrieve the node's private attribute @a node_uid_
                 */
                size_type get_node_uid() const {
                    return node_uid_;

                }

                /**
                 * @brief Return ref of (or set) this node's value
                 * @return the value of this node
                 */
                node_value_type& value() {
                    return ptr_to_graph_->vec_nodeinfo_[node_uid_].node_val_;
                }

                /**
                 * @brief Return ref of this node's value; value is unchangeable
                 * @return the value of this node
                 */
                const node_value_type& value() const {
                    return ptr_to_graph_->vec_nodeinfo_[node_uid_].node_val_;
                }

                /**
                 * @brief Return the num of edges incident to this node
                 * @return degree of this node (connecting edges)
                 */
                size_type degree() const {
                    return ptr_to_graph_-> map_rootnode2adjnodes_.at(node_uid_).size();
                }

                /**
                 * @brief For this node's edges, return first @a incident_iterator
                 * @return @a incident_iterator corresponding to first edge of this node
                 */
                IncidentIterator edge_begin() const {
                    return IncidentIterator(ptr_to_graph_, node_uid_);
                }

                /**
                 * @brief For this node's edges, return last @a incident_iterator
                 * @return @a incident_iterator corresponding to pass-the end edge of this node
                 */
                IncidentIterator edge_end() const{
                    return IncidentIterator(ptr_to_graph_, node_uid_, degree());
                }

                /**
                 * @brief Test whether this node and @a n are equal.
                 * @param[in] @a n  Node to test
                 * Equal nodes have the same graph and the same index.
                 */
                bool operator==(const Node& n) const {
                    return (n.ptr_to_graph_ == this->ptr_to_graph_) \
                                             && (n.node_uid_ == this->node_uid_);
                }

                /**
                 * @brief Test whether this node is less than @a n in a global order.
                 * @param[in] @a n  Node to test
                 * This ordering function is useful for STL containers such as
                 * std::map<>. It need not have any geometric meaning.
                 *
                 * The node ordering relation must obey trichotomy: For any two nodes x
                 * and y, exactly one of x == y, x < y, and y < x is true.
                 */
                bool operator<(const Node& n) const {
                    //pointing to different graph but happened to have the same unique ID
                    if(this->node_uid_ == n.node_uid_ && this->ptr_to_graph_ != n.ptr_to_graph_){
                        //compare graph pointer values with std::operator<
                        return std::less<Graph*>{}(this->ptr_to_graph_, n.ptr_to_graph_);
                    } else {
                        //otherwise, simply compare unique IDs
                        return this->node_uid_ < n.node_uid_;
                    }
                }

            private:
                // Allow Graph to access Node's private member data and functions.
                friend class Graph;

                /** private attributes of Node class*/
                // pointer to Graph
                Graph* ptr_to_graph_;
                //the index of this node
                size_type node_uid_;

                /** Node's private constructor */
                Node(const Graph* graph, size_type nid)
                    : ptr_to_graph_(const_cast<Graph*>(graph)),
                    node_uid_(nid){}
        }; //end Node class

        /** public methods of Graph related to Node */
        /**
         * @brief Return the number of nodes in the graph.
         * Complexity: O(1).
         */
        size_type size() const {
            return vec_activenodes_.size();
        }

        /** Synonym for size(). */
        size_type num_nodes() const {
            return size();
        }

        /**
         * @brief Add a node to the graph, returning the added node.
         * @param[in] position The new node's position
         * @param[in] value  Value of the adding node
         * @post new num_nodes() == old num_nodes() + 1
         * @post result_node.index() == old num_nodes()
         *
         * Complexity: O(1) amortized operations.
         */
        Node add_node(const Point& position,
                const node_value_type& value = node_value_type()) {

            //capture the index of the newly-added node, using current active nodes
            size_type new_node_index = num_nodes();
            //new node uid is the current total number of nodes of all time
            size_type new_node_uid = map_rootnode2adjnodes_.size();

            //matching the conversion map
            vec_activenodes_.push_back(new_node_uid);

            //construct internal struct object
            NodeInfo newNodeInternal = NodeInfo(position,new_node_index,value);
            vec_nodeinfo_.push_back(newNodeInternal);

            //insert the new unique node id into conversion map and push back an empty vector
            //such that we can later add neighbor nodes [KEY: rootnode uid; VALUE: neighbors uid]
            map_rootnode2adjnodes_.insert(std::pair<size_type,std::vector<size_type>>(new_node_uid,
                        std::vector<size_type>()));

            return Node(this,new_node_uid);
        }

        /**
         * @brief Determine if a Node belongs to this Graph
         * @param[in] @a n  Node to test
         * @return True if @a n is currently a Node of this Graph
         *
         * Complexity: O(1).
         */
        bool has_node(const Node& n) const {
            return n.ptr_to_graph_== this;
        }


        /**
         * @brief Return the node with index @a i.
         * @param[in] @a i  Index of the node
         * @pre 0 <= @a i < num_nodes()
         * @post result_node.index() == i
         *
         * Complexity: O(1).
         */
        Node node(size_type i) const {
            //vec_activenodes_[i] gives us node's uid
            return Node(this,vec_activenodes_[i]);
        }

        //
        // EDGE
        //
        /**
         * @class Graph::Edge
         * @brief Class representing the graph's edges.
         *
         * Edges are order-insensitive pairs of nodes. Two Edges with the same nodes
         * are considered equal if they connect the same nodes, in either order.
         */
        class Edge : private totally_ordered<Edge>{

            public:
                /** Construct an invalid Edge. */
                Edge()
                    : ptr_to_graph_(nullptr),
                    edge_uid_(invalid_index),
                    node1_uid_(invalid_index),
                    node2_uid_(invalid_index){}

                /** Return a node of this Edge */
                Node node1() const {
                    return Node(ptr_to_graph_,node1_uid_);
                }

                /** Return the other node of this Edge */
                Node node2() const {
                    return Node(ptr_to_graph_,node2_uid_);
                }

                /**
                 * @brief Test whether this edge and @a e are equal.
                 * @param[in] @a e  Edge to test
                 * Equal edges represent the same undirected edge between two nodes.
                 */
                bool operator==(const Edge& e) const {
                    bool same_eid = (edge_uid_ == e.edge_uid_);
                    bool same_nodes = (e.node1_uid_ == node1_uid_ && e.node2_uid_ == node2_uid_);
                    bool same_graphs = e.ptr_to_graph_ == ptr_to_graph_;
                    //return true if connecting same nodes and pointing to same graph
                    return (same_eid && same_nodes && same_graphs);
                }

                /**
                 * @brief Test whether this edge is less than @a e in a global order.
                 * @param[in] @a e  Edge to test
                 * This ordering function is useful for STL containers such as
                 * std::map<>. It need not have any interpretive meaning.
                 */
                bool operator<(const Edge& e) const {
                    //happend to have same edge unique id but pointing to different graph
                    if (edge_uid_ == e.edge_uid_ && this->ptr_to_graph_ != e.ptr_to_graph_) {
                        //compare graph pointer values with std::operator<
                        return std::less<Graph*>{}(this->ptr_to_graph_, e.ptr_to_graph_);
                    } else {
                        //otherwise, simply compare different edge id's
                        return this->edge_uid_ < e.edge_uid_;
                    }
                }

                /**
                 * @brief Return the length of this edge
                 */
                length_type length(){
                    return (norm(node1().position() - node2().position()));
                }

                /**
                 * @brief Return the edge's value as a reference that CANNOT be changed.
                 */
                const edge_value_type& value() const {
                    return ptr_to_graph_ -> vec_edgeinfo_[edge_uid_].edge_val_;
                }

                /**
                 * @brief Return the edge's value as a reference that CAN be changed.
                 */
                edge_value_type& value() {
                    return ptr_to_graph_ -> vec_edgeinfo_[edge_uid_].edge_val_;
                }

            private:
                // Allow Graph to access Edge's private member data and functions.
                friend class Graph;

                /** private attributes for Edge */
                //pointer to graph
                Graph* ptr_to_graph_;
                //edge index
                size_type edge_uid_;
                //first node (smaller) index
                size_type node1_uid_;
                //second node (bigger) index
                size_type node2_uid_;

                /** private constructor for Edge */
                Edge(const Graph* graph,
                        size_type eid,
                        size_type nid_1,
                        size_type nid_2)
                    : ptr_to_graph_(const_cast<Graph*>(graph)),
                    edge_uid_(eid),
                    node1_uid_(nid_1),
                    node2_uid_(nid_2){}
        };//end Edge

        /** public methods of Graph related to Edge */
        /**
         * @brief Return the total number of edges in the graph.
         *
         * Complexity: No more than O(num_nodes() + num_edges()), hopefully less
         */
        size_type num_edges() const {
            return vec_activeedges_.size();
        }

        /**
         * @brief Return the edge with index @a i.
         * @param[in] @a i  Index of this edge
         * @pre 0 <= @a i < num_edges()
         *
         * Complexity: No more than O(num_nodes() + num_edges()), hopefully less
         */
        Edge edge(size_type i) const {
            //precondition: passed-in index within range
            assert((i>=0) && (i<num_edges()));
            //do index -> unique edge id conversion
            size_type eid = vec_activeedges_[i];
            size_type index1 = map_edge2nodes_.at(eid)[0];
            size_type index2 = map_edge2nodes_.at(eid)[1];

            if(index1 < index2){
                return Edge(this,eid,index1,index2);
            } else {
                return Edge(this,eid,index2,index1);
            }
        }

        /**
         * @brief Test whether two nodes are connected by an edge.
         * @param[in] @a a  First node of this edge
         * @param[in] @a a  Second node of this edge
         * @pre @a a and @a b are valid nodes of this graph
         * @return True if for some @a i, edge(@a i) connects @a a and @a b.
         *
         * In my design, the nodes are checked in ascending order (smaller to bigger)
         *
         * Complexity: No more than O(num_nodes() + num_edges()), hopefully less
         */
        bool has_edge(const Node& a, const Node& b) const {
            //precondition: nodes are valid
            assert(has_node(a));
            assert(has_node(b));

            //sort node uid from small to big
            std::tuple<size_type, size_type> pair = sort_node_indices(a.get_node_uid(),b.get_node_uid());
            size_type smallnode = std::get<0>(pair);
            size_type bignode = std::get<1>(pair);

            //search inside nested map and find eid
            if(map_small2bignode_eid_.count(smallnode)){
                if(map_small2bignode_eid_.at(smallnode).count(bignode)){
                    size_type eid = map_small2bignode_eid_.at(smallnode).at(bignode);
                    size_type idx = vec_edgeinfo_[eid].edge_idx_;
                    //see if the edge index is valid
                    if(idx != invalid_index){
                        return true;
                    }
                }

            }
            return false;
        }

        /**
         * @brief Add an edge to the graph, or return the current edge if it already exists.
         * @param[in] @a a  First node of this edge
         * @param[in] @a a  Second node of this edge
         * @pre @a a and @a b are distinct valid nodes of this graph
         * @return an Edge object e with e.node1() == @a a and e.node2() == @a b
         * @post has_edge(@a a, @a b) == true
         * @post If old has_edge(@a a, @a b), new num_edges() == old num_edges().
         *       Else,                        new num_edges() == old num_edges() + 1.
         *
         * Can invalidate edge indexes -- in other words, old edge(@a i) might not
         * equal new edge(@a i). Must not invalidate outstanding Edge objects.
         *
         * In my design, the passed-in nodes are added to structures in ascending order.
         *
         * Complexity: No more than O(num_nodes() + num_edges()), hopefully less
         */
        Edge add_edge(const Node& a, const Node& b, const edge_value_type& val = edge_value_type()) {

            //precondition: nodes are valid
            assert(has_node(a));
            assert(has_node(b));
            //precondition: nodes are distinct
            assert(!(a==b));

            //sort the node uid from small to big
            std::tuple<size_type, size_type> pair = sort_node_indices(a.get_node_uid(),b.get_node_uid());
            size_type smallnode = std::get<0>(pair);
            size_type bignode = std::get<1>(pair);

            if(!has_edge(a,b)){

                size_type new_edge_uid = vec_edgeinfo_.size();
                size_type new_edge_idx = num_edges();

                //add to map (key is edge; val is pair (vec) of nodes)
                map_edge2nodes_[new_edge_uid] = {smallnode,bignode};
                //add node neighbors (add to map of adjacency in both direction for two nodes)
                map_rootnode2adjnodes_[smallnode].push_back(bignode);
                map_rootnode2adjnodes_[bignode].push_back(smallnode);
                //add to map of map
                map_small2bignode_eid_[smallnode][bignode] = new_edge_uid;
                //matching edge index and id (conversion)
                vec_activeedges_.push_back(new_edge_uid);
                //construct internal struct object and push back to vec_edgeinfo_
                EdgeInfo newEdgeInternal = EdgeInfo(new_edge_idx,new_edge_uid,val);
                vec_edgeinfo_.push_back(newEdgeInternal);

            }
            return Edge(this, map_small2bignode_eid_.at(smallnode).at(bignode),smallnode,bignode);

        }

        /**
         * @brief Remove all nodes and edges from this graph.
         * @post num_nodes() == 0 && num_edges() == 0
         *
         * Invalidates all outstanding Node and Edge objects.
         */
        void clear() {
            vec_nodeinfo_.clear();
            vec_activenodes_.clear();
            vec_edgeinfo_.clear();
            vec_activeedges_.clear();
            map_edge2nodes_.clear();
            map_rootnode2adjnodes_.clear();
            map_small2bignode_eid_.clear();
        }

        //
        // Node Iterator
        //
        /**
         * Sequential Note Iterator (Unparallelized Version)
         * Commented out for HW4 Random Access Iterator
         * @class Graph::NodeIterator
         * @brief Iterator class for nodes. A forward iterator.
         */
//        class NodeIterator: private totally_ordered<NodeIterator>{
//
//            public:
//                // These type definitions let us use STL's iterator_traits.
//                using value_type        = Node;                     // Element type
//                using pointer           = Node*;                    // Pointers to elements
//                using reference         = Node&;                    // Reference to elements
//                using difference_type   = std::ptrdiff_t;           // Signed difference
//                using iterator_category = std::forward_iterator_tag;  // Weak Category, Proxy
//
//                /** Construct an invalid NodeIterator. */
//                NodeIterator()
//                    : ptr_to_graph_(nullptr),
//                    nodeiter_currentidx_(invalid_index){
//                    }
//
//                /**
//                 * @brief A dereferencing operator for @a node_iterator
//                 * @return the value of underlying node of this iterator
//                 */
//                Node operator*() const {
//                    //using node function under Graph (returning a node)
//                    return ptr_to_graph_->node(nodeiter_currentidx_);
//                }
//
//                /**
//                 * @brief An incrementing operator for @node_iterator to the next position
//                 * @return this iterator that wraps to the next underlying element
//                 */
//                NodeIterator& operator++() {
//                    nodeiter_currentidx_++;
//                    return *this;
//                }
//
//                /**
//                 * @brief An equality operator for @a node_iterator to test if two iters are equal
//                 * @param @a nodeiter node_iterator to be compared with this
//                 * @return true if two are pointing to the same graph and have same current index
//                 */
//                bool operator==(const NodeIterator& nodeiter) const {
//                    bool same_graph = (nodeiter.ptr_to_graph_ == ptr_to_graph_);
//                    bool same_currentindex = (nodeiter.nodeiter_currentidx_ == nodeiter_currentidx_);
//                    return (same_graph && same_currentindex);
//                }
//
//            private:
//                friend class Graph;
//
//                Graph* ptr_to_graph_;
//                size_type nodeiter_currentidx_;
//
//                /** Nodeiterator  private constructor */
//                NodeIterator(const Graph* graph, const size_type nodeiterator_idx = 0)
//                    : ptr_to_graph_(const_cast<Graph*>(graph)),
//                    nodeiter_currentidx_(nodeiterator_idx) {
//                    }
//        }; //end Node Iterator (Unparallelized ver.)
    
        /** @brief A functor that maps a @a node_idx to its  node.
         *
         * @param[in] ptr_to_graph Pointer to the graph with the node with index @a node_idx
         *
         * This is a helper functor for new NodeIterator as a random access iterator.
         */
        struct Idx2Node {

          Idx2Node(const Graph* ptr_to_graph)
            : ptr_to_graph_(const_cast<Graph*>(ptr_to_graph)) {}

          Node operator() (size_type node_idx){
              return ptr_to_graph_->node(node_idx);
          }

         private:
          Graph* ptr_to_graph_;
        };//end Idx2Node
        
        /**
         * @brief A random access iterator that enables parallelized access of nodes in graph.
         *
         * thrust::transform_iterator is an iterator which represents a pointer into a range of values
         * after transformation by a function, useful for creating a range filled with the result of
         * applying an operation to another range without either explicitly storing it in memory.
         *
         * thrust::counting_iterator is an iterator which represents a pointer into a range of sequentially
         * changing values, useful for creating a range filled with a sequence without explicitly storing it in memory.
         */
        struct NodeIterator: thrust::transform_iterator<Idx2Node,thrust::counting_iterator<size_type>,node_type> {

          using super_t = thrust::transform_iterator<Idx2Node,
            thrust::counting_iterator<size_type>,
            node_type>;
            
          // NodeIterator invalid default constructor
          NodeIterator(){}

          // KLUDGE conversion constructor
          NodeIterator(const super_t& ti) : super_t{ti} {}

         private:
          friend class Graph;

          // Valid NodeIterator constructor
          NodeIterator(const Graph* graph, const size_type nodeiter_idx = 0)
            : super_t{thrust::counting_iterator<size_type>(nodeiter_idx),
                      Idx2Node(graph)} {
          }
        };//end NodeIterator struct (parallelized ver.)
    
        /** public methods of Graph related to node iterator */
        /** @brief Return the node iterator at the first starting position */
        NodeIterator node_begin() const {
            return NodeIterator(this);
        }

        /** @brief Return the node iterator at the pass-the-end position */
        NodeIterator node_end() const {
            return NodeIterator(this, num_nodes());
        }

        //
        // Incident Iterator
        //
        /**
         * @class Graph::IncidentIterator
         * @brief Iterator class for edges incident to a node. A forward iterator.
         */
        class IncidentIterator: private totally_ordered<IncidentIterator>{

            public:
                // These type definitions let us use STL's iterator_traits.
                using value_type        = Edge;                     // Element type
                using pointer           = Edge*;                    // Pointers to elements
                using reference         = Edge&;                    // Reference to elements
                using difference_type   = std::ptrdiff_t;           // Signed difference
                using iterator_category = std::input_iterator_tag;  // Weak Category, Proxy

                /** Construct an invalid IncidentIterator. */
                IncidentIterator()
                    : ptr_to_graph_(nullptr),
                    root_node_idx_(invalid_index),
                    num_visited_edges_(invalid_index){
                    }

                /**
                 * @brief A dereferencing operator for @a incident_iterator
                 * @return the value of underlying edge of this iterator
                 */
                Edge operator*() const {
                    //firstly capture the adjacent node relative to the root node
                    size_type adjnode_idx = ptr_to_graph_->map_rootnode2adjnodes_.at(root_node_idx_)[num_visited_edges_];
                    //declare a variable that captures the connecting edge index of root & adjacent node
                    size_type connecting_edge_id;

                    //sort the node uid from small to big
                    std::tuple<size_type, size_type> sorted = ptr_to_graph_->sort_node_indices(root_node_idx_,adjnode_idx);
                    //get the connecting edge uid for these two nodes
                    connecting_edge_id = ptr_to_graph_->map_small2bignode_eid_.at(std::get<0>(sorted)).at(std::get<1>(sorted));

                    return Edge(ptr_to_graph_, connecting_edge_id, root_node_idx_, adjnode_idx);
                }

                /**
                 * @brief An incrementing operator for @incident_iterator to the next neighbor node
                 * @return this iterator that wraps to the next underlying element
                 */
                IncidentIterator& operator++() {
                    num_visited_edges_++;
                    return *this;
                }

                /**
                 * @brief An equality operator for @a incident_iterator to test if two iters are equal
                 * @param @a inc_iter incident_iterator to be compared with this
                 * @return true if two are pointing to the same graph, have same root note, and visited same neighbors
                 */
                bool operator==(const IncidentIterator& inc_iter) const {
                    bool same_graph = inc_iter.ptr_to_graph_ == ptr_to_graph_;
                    bool same_rootnode = inc_iter.root_node_idx_ == root_node_idx_;
                    bool same_visited_edges = inc_iter.num_visited_edges_ == num_visited_edges_;
                    return (same_graph && same_rootnode && same_visited_edges);
                }

            private:
                friend class Graph;

                Graph* ptr_to_graph_;
                size_type root_node_idx_;
                size_type num_visited_edges_;

                /** Incident Iterator private constructor */
                IncidentIterator(const Graph* graph,
                        const size_type root_node_id,
                        const size_type num_visited_edges = 0)
                    : ptr_to_graph_(const_cast<Graph*>(graph)),
                    root_node_idx_(root_node_id),
                    num_visited_edges_(num_visited_edges) {}
        };//end IncidentIterator class

        //
        // Edge Iterator
        //
        /**
         * @class Graph::EdgeIterator
         * @brief Iterator class for edges. A forward iterator.
         */
        class EdgeIterator : private totally_ordered<EdgeIterator>{
            public:
                // These type definitions let us use STL's iterator_traits.
                using value_type        = Edge;                     // Element type
                using pointer           = Edge*;                    // Pointers to elements
                using reference         = Edge&;                    // Reference to elements
                using difference_type   = std::ptrdiff_t;           // Signed difference
                using iterator_category = std::input_iterator_tag;  // Weak Category, Proxy

                /** @brief Construct an invalid EdgeIterator. */
                EdgeIterator() {
                }

                /** @brief Return the edge currently indexed by EdgeIterator. */
                Edge operator*() const {
                    // get the node uid
                    size_type node_uid = ptr_to_graph_->vec_activenodes_[current_root_idx_];
                    // get node uid of the current neighbor of the root node
                    size_type neigh_uid = ptr_to_graph_->map_rootnode2adjnodes_.at(node_uid)[num_leaves_visited_];
                    // declare edge uid connecting the root node and neighbor node
                    size_type connecting_edge_uid;
                    // sort the node uid from small to big
                    std::tuple<size_type, size_type> sorted = ptr_to_graph_->sort_node_indices(node_uid,neigh_uid);
                    // capture the edge uid connecting the root node and neighbor node
                    connecting_edge_uid = ptr_to_graph_ -> map_small2bignode_eid_.at(std::get<0>(sorted)).at(std::get<1>(sorted));

                    return Edge(ptr_to_graph_, connecting_edge_uid, node_uid, neigh_uid);

                }

                /** @brief Return the EdgeIterator pointing to the next edge in the graph OR
                 * to the EdgeIterator object with @a current_root_id_ = num_nodes() and
                 * @a num_leaves_visited_ = 0 i.e. @a this.edge_end().
                 *
                 * @pre 0 <= @a current_root_id_ < @a num_nodes()
                 * @pre 0 <= @a num_leaves_visited < degree(current root node)
                 * @pre Node IDs are sequentially numbered from 0 to @a num_nodes() - 1
                 *      without any gaps. That is, letting k == @a num_nodes(), for all
                 *      the nodes n1,...,nk in G, {n1.node_id_,...,nk.node_id_} = {0,1,..
                 *      ..., @a num_nodes() - 1}.
                 * @post new @a num_leaves_visited_ == old @a num_leaves_visited_ + 1 OR
                 *                                     0
                 * @post 0 <= old @current_root_id_ <= new @a current_root_id_
                 *                                  <= @a num_nodes()
                 */
                EdgeIterator& operator++() {

                    num_leaves_visited_++;

                    // Begin at the current root node, moving to next root node
                    for( ; current_root_idx_ != ptr_to_graph_->num_nodes(); ++current_root_idx_) {

                        //get node uid of this root node
                        size_type root_node_uid = ptr_to_graph_->vec_activenodes_[current_root_idx_];
                        //get the degree of this root node
                        size_type root_degree = ptr_to_graph_->node(current_root_idx_).degree();

                        // Iterate over incident edges of current root node
                        for ( ; num_leaves_visited_ != root_degree; ++num_leaves_visited_) {
                            size_type leaf_uid = ptr_to_graph_->map_rootnode2adjnodes_.at(root_node_uid)[num_leaves_visited_];

                            // To avoid double-counting, traverse edge (e1, e2) only if e1.node_uid < e2.node_uid
                            if (root_node_uid < leaf_uid){
                                // Dereference EdgeIterator and return the element
                                return *this;
                            }
                        }
                        // Move to the next node and reset the number of leaves visited
                        num_leaves_visited_ = 0;
                    }
                    num_leaves_visited_ = 0;
                    // Dereference EdgeIterator and return the element
                    return *this;
                }

                /** @brief Check if this EdgeIterator and another @a e_itr are equal.
                 *
                 * @param[in] @a e_itr  EdgeIterator to compare against
                 * @return              True if and only if @a e_itr and this
                 *                      EdgeIterator: (1) have the same graph, (2) have
                 *                      the same root node, and (3) have traversed the same
                 *                      number of edges so far
                 */
                bool operator==(const EdgeIterator& e_itr) const {
                    return ( (ptr_to_graph_ == e_itr.ptr_to_graph_)
                            && (current_root_idx_ == e_itr.current_root_idx_)
                            && (num_leaves_visited_ == e_itr.num_leaves_visited_) );
                }


            private:
                friend class Graph;

                Graph* ptr_to_graph_;
                size_type current_root_idx_;
                size_type num_leaves_visited_;

                /** @brief Valid constructor for EdgeIterator.
                 *
                 * @param[in] graph               Graph pointer for EdgeIterator
                 * @param[in] current_root_id     ID of current root node (default = 0)
                 * @param[in] num_leaves_visited  Position of the iterator relative to
                 *                                the current root node (default = 0)
                 */
                EdgeIterator(const Graph* graph, const size_type current_root_idx = 0,
                        const size_type leaves_visited = 0)
                    : ptr_to_graph_(const_cast<Graph*>(graph)),
                    current_root_idx_(current_root_idx),
                    num_leaves_visited_(leaves_visited) {
                    }

        };

        /** @brief Return the iterator for where to start visiting (unique) edges. */
        EdgeIterator edge_begin() const {
            // Start iterating with current_root_id_ == num_leaves_visited_ == 0
            return EdgeIterator(this, 0, 0);
        }

        /** @brief EdgeIterator indicating when to stop attempting to visit (unique)
         * edges. */
        EdgeIterator edge_end() const {
            // Stop iterating when current_root_id_ == num_nodes() and
            // num_leaves_visited_ == 0
            return EdgeIterator(this, num_nodes(), 0);
        }

        /**
         * @brief Given two nodes, remove the connecting edge between them from a graph
         * @param[in,out] a   A node of the edge to be deleted
         * @param[in,out] b   Another node of the edge to be deleted
         * @returns return 0 if not deleted; 1 if successfully deleted
         *
         * @post If (has_edge(@a a, @a b) == true) {old @a num_edges_ == new @a num_edges_ + 1}
         *  else { old @a num_edges_ == new @a num_edges_ }
         *
         * Complexity: O(d) overall runtime
         *
         */
        size_type remove_edge(const Node& a, const Node& b) {

            //precondition: nodes are valid
            assert(has_node(a));
            assert(has_node(b));
            //precondition: nodes are distinct
            assert(!(a==b));

            //sor the node uid from small to big
            std::tuple<size_type, size_type> pair = sort_node_indices(a.get_node_uid(),b.get_node_uid());
            size_type smallnode = std::get<0>(pair);
            size_type bignode = std::get<1>(pair);

            if(!has_edge(a,b)) {
                return (size_type) false;
            }

            //last edge id of the current last active edge
            size_type last_eid = vec_activeedges_.back();
            size_type index; //deleted edge's original idx
            //edge id of the edge to be deleted
            size_type target_eid = map_small2bignode_eid_.at(smallnode).at(bignode);

            /** vec_activeedges_ section */
            //Edge uid's are obtained. Then loop through conversion map and perform swap and pop
            for(auto iter = vec_activeedges_.begin(); iter != vec_activeedges_.end();) {
                if((*iter) == target_eid) {
                    //get target eid idx among activeedges
                    index = std::distance(vec_activeedges_.begin(), iter);
                    //swap with the back
                    std::swap(*iter, vec_activeedges_.back());
                    //erase the element
                    vec_activeedges_.pop_back();
                    break;
                } else {
                    ++iter;
                }
            }

            /** vec_edgeinfo_ section */
            //Now change Edge Internal Struct by accessing vec_edgeinfo_
            for(auto it = vec_edgeinfo_.begin(); it != vec_edgeinfo_.end(); ++it){
                size_type eid_count = std::distance(vec_edgeinfo_.begin(), it);
                //let the swapped last one have the deleted edge's index
                if(eid_count == last_eid){
                    vec_edgeinfo_[eid_count].edge_idx_ = index;
                }
                //nullify the deleted edge's index
                if(eid_count == target_eid){
                    vec_edgeinfo_[eid_count].edge_idx_ = invalid_index;
                }
            }

            /** map_edge2nodes_ section */
            map_edge2nodes_.erase(target_eid); //erase map value by key

            /** map_rootnode2adjnodes_ section */
            //delete smallnode's neighbor bignode
            std::vector<size_type>& small = map_rootnode2adjnodes_.at(smallnode);
            for(auto it = small.begin(); it != small.end(); ++it){
                if(bignode == (*it)){
                    small.erase(it);
                    break;
                }
            }
            //delete bignode's neighbor smallnode
            std::vector<size_type>& big = map_rootnode2adjnodes_.at(bignode);
            for(auto it = big.begin(); it != big.end(); ++it){
                if(smallnode == (*it)){
                    big.erase(it);
                    break;
                }
            }

            //one edge was removed
            return size_type(1);

        }//end size_type remove_edge


        /**
         * @brief Given an edge, remove this edge from the graph.
         * @param[in,out] e   A reference to an edge to be deleted
         * @post If (has_edge(@a a, @a b) == true) {old @a num_edges_ == new @a num_edges_ + 1}
         *  else { old @a num_edges_ == new @a num_edges_ }
         * @returns return 0 if not deleted; 1 if successfully deleted
         *
         * Complexity: O(d)
         */
        size_type remove_edge(const Edge& e){
            return remove_edge(e.node1(),e.node2());
        }

        /**
         * @brief Given an edge iterator, remove the edge to which iterator points.
         * @param[in,out] e_it   The edge iterator
         * @post If (has_edge(@a a, @a b) == true) {old @a num_edges_ == new @a num_edges_ + 1}
         *  else { old @a num_edges_ == new @a num_edges_ }
         * @returns EdgeIterator object to the new @a graph_.edge_end()
         *
         * Complexity: O(d)
         */
        EdgeIterator remove_edge(edge_iterator e_it) {

            // If this edge is the last one
            if(e_it.current_root_idx_  == num_nodes() - 1 &&
                    e_it.num_leaves_visited_ == node(e_it.current_root_idx_).degree() - 1){
                // Decrement the iterator to return points to the graph.edge_end()
                --e_it;
            }
            // Otherwise, removing an edge takes iterator to the next edge in the graph
            remove_edge(*e_it);
            // Return the iterator
            return e_it;
        }

        /**
         * @brief Given a node @a a, remove it from the graph
         * @param[in]  a Node to be deleted
         * @returns return 0 if not deleted; reutrn the @a node_uid_ if successfully deleted
         *
         * Complexity: O(num_nodes() + num_edges_), hopefully less.
         */
        size_type remove_node(const Node& a){

            if(!has_node(a)){
                return (size_type) false;
            }

            //target uid to be removed
            size_type target_uid = a.get_node_uid();
            //current last node id of active nodes
            size_type last_uid = vec_activenodes_.back();
            //node's original index
            size_type index;

            //as long as there's still neighbor
            while((map_rootnode2adjnodes_.at(target_uid)).size() != 0){
                size_type neigh = map_rootnode2adjnodes_.at(target_uid)[0];
                size_type eid;
                if(neigh<target_uid){
                    eid = map_small2bignode_eid_.at(neigh).at(target_uid);
                } else {
                    eid = map_small2bignode_eid_.at(target_uid).at(neigh);
                }
                //get neighbor uid and edge id and remove edge
                remove_edge(Edge(this,eid,neigh,target_uid));
            }

            /* perform deletion on vec_activenodes_ */
            for(auto iter = vec_activenodes_.begin(); iter != vec_activenodes_.end();){
                //vector value (nid) matches target nid
                if((*iter) == target_uid){
                    //get current position among vec_activenodes_
                    index = std::distance(vec_activenodes_.begin(),iter);
                    std::swap(*iter, vec_activenodes_.back()); //swap
                    vec_activenodes_.pop_back(); //and pop
                    break;
                } else {
                    ++iter;
                }
            }
            /* perform revision on vec_nodeinfo_ (modifying internal struct) */
            for(auto i = vec_nodeinfo_.begin(); i != vec_nodeinfo_.end(); ++i){
                //get current position among vec_nodeinfo_
                size_type nid_count = std::distance(vec_nodeinfo_.begin(),i);
                //set the last one's node index to the deleted one's index (swap)
                if(nid_count == last_uid){
                    vec_nodeinfo_[nid_count].node_idx_ = index;
                }
                //nullify the deleted one (pop)
                if(nid_count == target_uid){
                    vec_nodeinfo_[nid_count].node_idx_ = invalid_index;
                }
            }
            //done removing node
            return a.get_node_uid();
        }

        /**
         * @brief Given a NodeIterator, remove the node that it points to from the graph
         * @param[in] n_it NodeIterator that points to a node
         * @returns NodeIterator that points to the beginning of iterator if successfully deleting node
         * or return a NodeIterator that points to the end if already in the end
         *
         * Complexity: O(num_nodes() + num_edges_), hopefully less.
         */
        NodeIterator remove_node(NodeIterator n_it){
            //if iterator already in the end, return this
            if(n_it == node_end()){
                return node_end();
                //otherwise, perform remove_node if node is there
            } else {
                Node n = *n_it;
                if(has_node(n)){
                    remove_node(n);
                }
                return node_begin();
            }
        }

        /** PRIVATE BLOCK FOR GRAPH */
    private:

        /** Struct for storing information associated with each Node. */
        struct NodeInfo {
            Point position_;
            size_type node_idx_;
            node_value_type node_val_;

            /** Constructor */
            NodeInfo(const Point& pos,
                    const size_type idx,
                    node_value_type val)
                : position_(pos),
                node_idx_(idx),
                node_val_(val){}
        };

        /** Struct for storing information associated with each Edge. */
        struct EdgeInfo {
            size_type edge_idx_;
            size_type edge_uid_;
            edge_value_type edge_val_;

            /** Constructor */
            EdgeInfo(const size_type idx,
                    const size_type eid,
                    edge_value_type val)
                : edge_idx_(idx),
                edge_uid_(eid),
                edge_val_(val){}
        };

        //A vector w/ node_uid_ as index; NodeInfo (Internal Struct) as value
        std::vector<NodeInfo> vec_nodeinfo_;
        //A vector storing currently active nodes w/ node_idx_ as index; node_uid_ as value
        std::vector<size_type> vec_activenodes_;
        //A vector x/ edge_uid_ as index; EdgeInfo (Internal Struct) as value
        std::vector<EdgeInfo> vec_edgeinfo_;
        //A vector storing currently active edges w/ edge_idx_ as index; edge_uid_ as value
        std::vector<size_type> vec_activeedges_;
        //A map with edge_uid_ as key; vec <smallnode_uid, bignode_uid> as value
        std::unordered_map<size_type,std::vector<size_type>> map_edge2nodes_;
        //A map with rootnode uid as key; vec of neighbor nodes' uid as value
        std::unordered_map<size_type,std::vector<size_type>> map_rootnode2adjnodes_;
        //A map of map with smaller node uid as outer key; bigger node uid is inner key; edge uid as value
        std::unordered_map<size_type,std::unordered_map<size_type,size_type>> map_small2bignode_eid_;

        /** A helper function that gives us sorted node indices pair for each edge.
         *
         * @param id1 the id of the first node
         * @param id2 the id of the second node
         * @return A tuple of size two: (smaller, larger)

         * @post std::get<0>(result) < std::get<1>(result)
         */
        std::tuple<size_type, size_type> sort_node_indices(size_type id1, size_type id2) const{

            size_type small, big;
            if(id1 < id2){
                small = id1;
                big = id2;
            } else {
                small = id2;
                big = id1;
            }
            std::tuple<size_type, size_type> pair{small, big};

            return pair;
        }

};//end Graph class

#endif // CME212_GRAPH_HPP
