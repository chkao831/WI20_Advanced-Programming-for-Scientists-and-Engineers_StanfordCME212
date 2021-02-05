/**
 * @file Graph.hpp
 *
 * @author Chih-Hsuan (Carolyn) Kao
 * Contact: chkao831@stanford.edu
 * Date: Feb 18, 2020
 *
 * @brief Reads in two files specified on the command line.
 * First file: 3D Points (one per line) defined by three doubles
 * Second file: Tetrahedra (one per line) defined by 4 indices into the point list
 */

#include <fstream>
#include <chrono>
#include <thread>

#include "thrust/for_each.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/system/omp/execution_policy.h"

#include "CME212/SFML_Viewer.hpp"
#include "CME212/Util.hpp"
#include "CME212/Color.hpp"
#include "CME212/Point.hpp"

#include "Graph.hpp"
#include "SpaceSearcher.hpp"

// Gravity in meters/sec^2
static constexpr double grav = 9.81;

/** Custom structure of data to store with Nodes */
struct NodeData {
    Point vel;    // Node velocity
    double mass;  // Node mass
    NodeData() : vel(0), mass(1) {}
};

/** Custom structure of data to store with Edges */
struct EdgeData {
    double K;     // Edge spring constant
    double L;     // Edge spring L
    EdgeData() : K(100), L(0) {}
};

// Definition of types
using GraphType = Graph<NodeData, EdgeData>;
using Node = typename GraphType::node_type;
using Edge = typename GraphType::edge_type;
using size_type = unsigned;

//
//
// HW2 PART 5 GENERALIZING FORCES
// PARENT: ZeroForce with 3 derived children Force classes
//
//
/** @brief Parent Force structure to simulate children forces on a given node. */
struct ZeroForce {

    /** Invalid default constructor, destructor and operato to be later implemented */
    ZeroForce(){}
    virtual Point operator()(Node n, double t) = 0;
    virtual ~ZeroForce(){}

};

/**
 * @brief Child Force structure to simulate force on a given node.
 * This struct implements the spring forces.
 * It returns the mass spring force applying to @a n at time @a t.
 */
struct MassSpringForce: public ZeroForce {

    MassSpringForce(){}

    virtual Point operator()(Node n, double t) {

        (void) t;
        Point f_spring = Point(0,0,0);
        for(auto i = n.edge_begin(); i!=n.edge_end(); ++i){
            auto e = *i;
            Point diff_pos = n.position() - e.node2().position();
            double norm_diff = norm(n.position() - e.node2().position());
            double K = e.value().K;
            double L = e.value().L;
            diff_pos *= -K*(norm_diff - L)/norm_diff;
            f_spring += diff_pos;
        }
        return f_spring;
    }

    virtual ~MassSpringForce(){}
};

/**
 * @brief Child Force structure to simulate force on a given node.
 * This struct implements the force of gravity.
 * It returns the gravity force applying to @a n at time @a t.
 */
struct GravityForce: public ZeroForce {

    GravityForce(){}

    virtual Point operator()(Node n, double t) {
        (void) t;
        return n.value().mass * Point(0,0,-grav);
    }

    virtual ~GravityForce(){}

};

/**
 * @brief Child Force structure to simulate force on a given node.
 * This struct implements damping force (a form of friction).
 * It returns the damping force applying to @a n at time @a t.
 */
struct DampingForce: public ZeroForce {

    /** constructors */
    DampingForce()
        : damp_const_(0) {}
    DampingForce(double c)
        : damp_const_(c) {}

    virtual Point operator()(Node n, double t) {

        (void) t;
        Point f_damp = n.value().vel;
        return f_damp *= -damp_const_;
    }

    virtual ~DampingForce(){}

    private:
    double damp_const_;

};

//
//
// HW2 PART 1
// Force function object for problem1
//
//
/** Return the force applying to @a n at time @a t.
 *
 * For HW2 #1, this is a combination of mass-spring force and gravity,
 * except that points at (0, 0, 0) and (1, 0, 0) never move. We can
 * model that by returning a zero-valued force.
 */
struct Problem1Force {

    template <typename NODE>
        Point operator()(NODE n, double t) {

            (void) t;
            //Edge case: constraining two corners of the cloth by returning zero force
            if(n.position() == Point(0,0,0) || n.position() == Point(1,0,0)){
                return Point(0,0,0);
            }

            MassSpringForce msf_obj = MassSpringForce();
            GravityForce gf_obj = GravityForce();
            Point f_spring = msf_obj(n,t);
            Point f_grav = gf_obj(n,t);

            f_spring += f_grav;

            return f_spring;
        }
};//end Problem1Force

/**
 * @brief This struct is constructed with a vector of different forces.
 * This applies to all forces with operator()(Node n, double t)
 */
struct CombinedForces {

    //constructor
    CombinedForces(std::vector<ZeroForce*> v)
        :input_force_vec_(v){
        };

    //operator
    Point operator()(Node n, double t) {

        //original position
        Point pt = Point(0,0,0);
        //iterate through all forces and update with forces
        for (auto it = input_force_vec_.begin(); it != input_force_vec_.end(); ++it){
            pt += (*(*it))(n,t);
        }
        return pt;
    }

    //private attribute
    private:
    std::vector<ZeroForce*> input_force_vec_;

};//end CombinedForce

/**
 * @brief Templated helper function to combine the effects of 2 forces in a vector.
 */
template <typename F1, typename F2>
CombinedForces make_combined_force(F1& f1, F2& f2){
    std::vector<ZeroForce*> v = {&f1, &f2};
    return CombinedForces(v);
}

//
//
// HW2 PART 6 GENERALIZING CONSTRAINTS
// PARENT: ZeroConstraint with 3 derived Constraint classes
//
//
/** @brief Abstract Parent Constraint struct to simulate children constraints for this graph. */
struct ZeroConstraint {

    ZeroConstraint(){}
    virtual void operator()(GraphType& g, double t) = 0;
    virtual ~ZeroConstraint(){}
};

/**
 * @brief Given a vector of nodes along with a vector of  positions,
 * resets node positions to original ones and set node velocities to zero.
 */
struct PinConstraint: public ZeroConstraint {

    PinConstraint(std::vector<Node>& fixed_nodes,
            std::vector<Point>& pos)
        : nodes_to_be_fixed_(fixed_nodes),
        positions_(pos){}

    /** Operator */
    virtual void operator()(GraphType& g, double t){
        (void) g;
        (void) t;
        for(Point::size_type k = 0; k < positions_.size(); k++){
            Node fix_node = nodes_to_be_fixed_[k];
            // Reset the node's position to default position
            fix_node.position() = positions_[k];
            // Set its velocity to zero
            fix_node.value().vel = Point(0,0,0);
        }
    };

    virtual ~PinConstraint(){}

    private:
    std::vector<Node> nodes_to_be_fixed_;
    std::vector<Point> positions_;
};

/**
 * @brief A constraint that projects certain nodes onto a plane.
 */
struct PlaneConstraint : public ZeroConstraint {

    /** Default Constructor */
    PlaneConstraint(double thresh)
        :threshold_(thresh){}

    /** Operator */
    virtual void operator()(GraphType& g, double t){

        (void) t;
        for(auto i = g.node_begin(); i != g.node_end(); ++i){
            Node n = *i;
            // Violated constraint
            if (dot(n.position(),Point(0,0,1)) < threshold_){
                // Projection: i.e. set the position to the nearest point on the plane
                n.position().z = threshold_;
                // Set the z-component velocity to 0
                n.value().vel.z = 0;
            }
        }

    }//end operator

    private:
    double threshold_;

}; //end PlaneConstraint

/**
 * @brief A constraint that wraps nodes around a sphere.
 */
struct SphereConstraint : public ZeroConstraint {

    /** Default Constructor */
    SphereConstraint(Point c, double r)
        : center_(c),
        radius_(r){}

    /** Operator */
    virtual void operator()(GraphType& g, double t){
        (void) t;
        for(auto i = g.node_begin(); i != g.node_end(); ++i){
            Node n = *i;
            //capture vec from center to point
            Point vec_r = n.position() - center_;
            //check if violate constraint
            if(norm(vec_r) < radius_){
                //normalize
                Point vec_unit_radius = vec_r/norm(vec_r);
                //Set the position to nearest point on sphere surface
                n.position() = vec_unit_radius*radius_ + center_;
                //Set the component of the velocity that is normal to sphere surface to zero
                n.value().vel -= dot(n.value().vel, vec_unit_radius)*vec_unit_radius;
            }
        }
    }//end operator

    private:
    Point center_;
    double radius_;

};//end SphereConstraint

//
//
// HW2 PART 7.2
// PARENT: ZeroForce with 3 derived children Force classes
//
//
/**
 * @brief A constraint that removes nodes inside a given sphere.
 */
struct CutHoleConstraint  : public ZeroConstraint {
    // Default constructor
    CutHoleConstraint(const Point center, double r)
        : center_(center),
        radius_(r){}

    //operator
    virtual void operator()(GraphType& g, double t){
        (void) t;

        for(auto ni = g.node_begin(); ni!= g.node_end(); ++ni){
            auto n = *ni;
            double dist = norm(n.position() - center_);
            // For all nodes n in graph, |n.position() - center_| >= radius_
            if(dist < radius_){
                ni = g.remove_node(ni);
            }
        }
    }

    virtual ~CutHoleConstraint(){};

    private:
    const Point center_; // Center of sphere of node removal
    const double radius_; // Radius of sphere of node removal

};//PART 7.2 END

//
//
// HW4 PART 4.3 Efficient Neighbor Search
//
//
/**
 * @brief Custom functor used in DetermineInfluence() to cut down
 * the velocities of certain nodes within a bounding box in parallel.
 *
 * @param n1     Node with position as the center of the sphere of nodes
 *           whose velocities will be cut down.
 * @param radius Square of the radius of the sphere of nodes
 */
struct CutDownVelocity {
    CutDownVelocity(Node& n, double ra)
        : n1(n), radius(ra) {
        }

    void operator()(Node n2){

        Point r = n1.position() - n2.position();
        double l2 = normSq(r);
        if (n1 != n2 && l2 < radius) {
            // Remove velocity component in r to prevent hitting
            n1.value().vel -= (dot(r, n1.value().vel) / l2) * r;
        }
    }
    Node n1;
    double radius;
};

/**
 * @brief A helper function that takes in two bounding box and finds intersection
 *       such that the resulting bounding box is always contained by the pass-in parameters
 *       This helper function is used in DetermineInfluence() in which searcher iterates through the box
 */
Box3D FindBoxIntersection(Box3D& smallbox, Box3D& bigbox){
    //obtain max_ and min_ of each box
    Point max_smallbox = smallbox.max();
    Point max_bigbox = bigbox.max();
    Point min_smallbox = smallbox.min();
    Point min_bigbox = bigbox.min();

    //declare points of resulting box and update them
    Point resulting_max = max_smallbox;
    Point resulting_min = min_smallbox;
    for(Point::size_type i = 0; i < min_smallbox.size(); ++i){
        if(max_smallbox[i] > max_bigbox[i]){
            resulting_max[i] = max_bigbox[i];
        }
        if(min_smallbox[i] < min_bigbox[i]){
            resulting_min[i] = min_bigbox[i];
        }
    }

    //cross-comparison
    for(Point::size_type i = 0; i < min_smallbox.size(); ++i){
        if(resulting_max[i] < min_bigbox[i]){
            resulting_max[i] = min_bigbox[i];
        }
        if(resulting_min[i] > max_bigbox[i]){
            resulting_min[i] = max_bigbox[i];
        }
    }
    return Box3D(resulting_min,resulting_max);
}//end FindBoxIntersection

/**
 * @brief Struct that determines squared radius and sets bounding box
 * to be iterated by searcher from SelfCollisionConstraint().
 *
 * @param[in] searcher_    SpaceSearcher that enforces a constraint
 *                      on nodes within a given bounding box in parallel.
 */
struct DetermineInfluence {
    DetermineInfluence(SpaceSearcher<Node>& ss): searcher_(ss){}
    void operator()(Node n){
        const Point& center = n.position();
        double radius2 = std::numeric_limits<double>::max();

        //Determine squared radius with std::min
        for (auto eit = n.edge_begin(); eit != n.edge_end(); ++eit){
            radius2 = std::min(radius2, normSq((*eit).node2().position() - center));
        }
        radius2 *= 0.9;

        //construct a small bounding box around the node using scaled radius
        Point upper = center - sqrt(radius2);
        Point lower = center + sqrt(radius2);
        Box3D SmallBB(lower, upper);
        //capture the big bounding box from the searcher
        Box3D BigBB = searcher_.bounding_box();
        //call FindBoxIntersection to get a intersection box contained by the big bounding box
        Box3D result_bb = FindBoxIntersection(SmallBB,BigBB);

        assert(searcher_.bounding_box().contains(result_bb));

        //a constraint is enforced within the bounding box in parallel
        thrust::for_each(searcher_.begin(result_bb), searcher_.end(result_bb), CutDownVelocity(n, radius2));
    }
    private:
    SpaceSearcher<Node>& searcher_;

};

/**
 * @brief Constraint that prevents the nodes within a graph from collision.
 * It is a child struct of ZeroConstraint.
 */
struct SelfCollisionConstraint: ZeroConstraint {

    void operator()(GraphType& g, double t) {
        (void) t;

        //lambda function for definition of big bounding box
        auto n2p = [](const Node& n) { return n.position(); };
        Box3D bigbb = Box3D(thrust::make_transform_iterator(g.node_begin(), n2p),
                thrust::make_transform_iterator(g.node_end(), n2p));

        //apply extension to the original big bounding box to avoid containment assertion error
        //as instructed by TA Kyle and Lewis
        Point extended_lower = bigbb.min();
        Point extended_upper = bigbb.max();
        for(size_type i = 0; i < extended_lower.size(); ++i){
            extended_lower[i] = -abs(extended_lower[i])*1.5;
            extended_upper[i] = abs(extended_upper[i])*1.5;
        }
        bigbb = Box3D(extended_lower, extended_upper);

        //declare a searcher with this bounding box
        SpaceSearcher<Node> searcher(bigbb, g.node_begin(), g.node_end(), n2p);
        // Enforce the constraint via DetermineInfluence on each node in graph in parallel
        thrust::for_each(g.node_begin(), g.node_end(), DetermineInfluence(searcher));
    }
};


/**
 * @brief This struct is constructed with a vector of different constraints.
 */
struct CombinedConstraints {

    std::vector<ZeroConstraint*> input_const_vec;

    //constructor
    CombinedConstraints(std::vector<ZeroConstraint*> c)
        :input_const_vec(c){};

    //operator
    void operator()(GraphType& g, double t){

        //iterate through all forces and update with forces
        for(unsigned int i = 0; i < input_const_vec.size(); ++i){
            (*input_const_vec[i])(g, t);
        }//end operator
    }
};//end CombinedConstraints

/**
 * @brief Templated function to combine vector of constraints
 */
template <typename C1, typename C2, typename C3>
CombinedConstraints make_combined_constraints(C1& c1,C2& c2 ,C3& c3){
    std::vector<ZeroConstraint*> vec;
    vec = {&c1, &c2, &c3};
    return CombinedConstraints(vec);
}

//
//
// HW4 PART 3 Parallel mass-spring state update
//
//
/**
 * @brief Helper fucntor for sym_euler_step() method to update the positions of nodes in parallel.
 */
struct ThrustUpdateNodePos  {

    /** constructor */
    ThrustUpdateNodePos(double difftime)
        : dt_(difftime){}

    /** operator */
    void operator()(Node n){
        n.position() += n.value().vel * dt_;
    }

    /** private attribute */
    double dt_;

}; //end ThrustUpdateNodePos

/**
 * @brief Helper fucntor for sym_euler_step() method to update the velocities of nodes in parallel.
 */
template <typename F>
struct ThrustUpdateNodeVel  {

    /** constructor */
    ThrustUpdateNodeVel(double time, double diff_time, F& force)
        : f_(force),
        t_(time),
        dt_(diff_time){}

    /** operator */
    void operator()(Node n){
        n.value().vel += f_(n, t_) * (dt_ / n.value().mass);
    }

    /** private attributes */
    F f_; //force
    double t_; //time
    double dt_; //delta_t

}; //end ThrustUpdateNodeVel

/** Change a graph's nodes according to a step of the symplectic Euler
 *    method with the given node force.
 * @param[in,out] g      Graph
 * @param[in]     t      The current time (useful for time-dependent forces)
 * @param[in]     dt     The time step
 * @param[in]     force  Function object defining the force per node
 * @return the next time step (usually @a t + @a dt)
 *
 * @tparam G type parameter for the graph
 * @tparam F is a function object called as @a force(n, @a t),
 *           where n is a node of the graph and @a t is the current time.
 *           @a force must return a Point representing the force vector on
 *           Node n at time @a t.
 * @tparam C is a function object called as @a constraint(g, @a t),
 *           where g is a  graph and @a t is the current time.
 *           @a constraint set specific constraints to graph g on
 *           Node n at time @a t.
 */
template <typename G, typename F, typename C>
double symp_euler_step(G& g, double t, double dt, F force, C constraint) {
    // Compute the t+dt position (sequentially)
    //  for (auto it = g.node_begin(); it != g.node_end(); ++it) {
    //    auto n = *it;
    //
    //    // Update the position of the node according to its velocity
    //    // x^{n+1} = x^{n} + v^{n} * dt
    //    n.position() += n.value().vel * dt;
    //  }

    thrust::for_each(thrust::omp::par, g.node_begin(), g.node_end(),
            ThrustUpdateNodePos(dt));

    // Applying specific constraints to the graph
    constraint(g,t);

    // Compute the t+dt velocity (sequentially)
    //  for (auto it = g.node_begin(); it != g.node_end(); ++it) {
    //    auto n = *it;
    //
    //    // v^{n+1} = v^{n} + F(x^{n+1},t) * dt / m
    //    n.value().vel += force(n, t) * (dt / n.value().mass);
    //  }

    thrust::for_each(thrust::omp::par, g.node_begin(), g.node_end(), ThrustUpdateNodeVel<F>(t,dt,force));

    return t + dt;
}

//
//
// HW4 PART 3 Parallel mass-spring state update
//
//
/**
 * @brief Functor that produces a unary function that initializes its position and velocitity
 * for the mass-spring simulation.
 *
 * @param num_nodes  Number of nodes
 * @param fixed_nodes Containing nodes whose positions will later be fixed
 * @param fixed_positions Containing points corresponding to the initial positions of fixed nodes
 */
struct ThrustNodeInitialization {

    /**constructor */
    ThrustNodeInitialization(Point::size_type num_nodes,
            std::vector<Node>& fixed_nodes,
            std::vector<Point>& fixed_positions)
        : num_nodes_(num_nodes),
        fixed_nodes_(fixed_nodes),
        fixed_positions_(fixed_positions){}

    /** operator */
    void operator() (Node n){
        //set initial conditions for nodes
        n.value().mass = 1.0/num_nodes_; //as requested
        n.value().vel = Point(0,0,0);
        //fixing points on corners
        if(n.position() == Point(0,0,0) ||
                n.position() == Point(1,0,0)){
            fixed_nodes_.push_back(n);
            fixed_positions_.push_back(n.position());
        }
    }

    /**private attributes */
    Point::size_type num_nodes_; //obtained from graph.num_nodes() outside
    std::vector<Node>& fixed_nodes_; //storing fixed nodes
    std::vector<Point>& fixed_positions_;

};//end ThrustNodeInitialization

int main(int argc, char** argv)
{
    // Check arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " NODES_FILE TETS_FILE\n";
        exit(1);
    }

    // Construct an empty graph
    GraphType graph;

    // Create a nodes_file from the first input argument
    std::ifstream nodes_file(argv[1]);
    // Interpret each line of the nodes_file as a 3D Point and add to the Graph
    Point p;
    std::vector<typename GraphType::node_type> nodes;
    while (CME212::getline_parsed(nodes_file, p))
        nodes.push_back(graph.add_node(p));

    // Create a tets_file from the second input argument
    std::ifstream tets_file(argv[2]);
    // Interpret each line of the tets_file as four ints which refer to nodes
    std::array<int,4> t;
    while (CME212::getline_parsed(tets_file, t)) {
        graph.add_edge(nodes[t[0]], nodes[t[1]]);
        graph.add_edge(nodes[t[0]], nodes[t[2]]);
#if 0
        // Diagonal edges: include as of HW2 #2
        graph.add_edge(nodes[t[0]], nodes[t[3]]);
        graph.add_edge(nodes[t[1]], nodes[t[2]]);
#endif
        graph.add_edge(nodes[t[1]], nodes[t[3]]);
        graph.add_edge(nodes[t[2]], nodes[t[3]]);
    }

    //Customizable section for fixing points
    std::vector<Node> fixed_corners;
    std::vector<Point> fixed_positions;

    /* Sequential version to set initial conditions for all edges */
    
//    for (auto it = graph.node_begin(); it != graph.node_end(); ++it) {
//        auto n = *it;
//        n.value().mass = (double) 1/graph.size();
//        n.value().vel = Point(0,0,0);
//
//        if(n.position() == Point(0,0,0) || n.position() == Point(1,0,0)){
//            fixed_corners.push_back(n);
//            fixed_positions.push_back(n.position());
//        }
//    }

    /* Parallel version to set initial conditions for all edges */
    thrust::for_each(thrust::omp::par,
            graph.node_begin(),
            graph.node_end(),
            ThrustNodeInitialization(graph.num_nodes(),
                fixed_corners,
                fixed_positions));

    // Set initial conditions for all edges (sequentially)
    const double spring_const_K = 100;
    for (auto it = graph.edge_begin(); it != graph.edge_end(); ++it) {
        auto e = *it;
        e.value().L = e.length();
        e.value().K = spring_const_K;
    }

    // Print out the stats
    std::cout << graph.num_nodes() << " " << graph.num_edges() << std::endl;

    // Launch the Viewer
    CME212::SFML_Viewer viewer;
    auto node_map = viewer.empty_node_map(graph);

    viewer.add_nodes(graph.node_begin(), graph.node_end(), node_map);
    viewer.add_edges(graph.edge_begin(), graph.edge_end(), node_map);

    viewer.center_view();

    // We want viewer interaction and the simulation at the same time
    // Viewer is thread-safe, so launch the simulation in a child thread
    bool interrupt_sim_thread = false;
    auto sim_thread = std::thread([&](){

            //TIMING
            CME212::Clock clock;
            // Begin the mass-spring simulation
            double dt = 0.001;
            double t_start = 0;
            double t_end = 5.0;

            clock.start();
            for (double t = t_start; t < t_end && !interrupt_sim_thread; t += dt) {

            GravityForce F1 = GravityForce();
            MassSpringForce F2 = MassSpringForce();
            CombinedForces combined_force_fn = make_combined_force(F1, F2);

            Point center = Point(0.5,0.5,-0.5);
            double radius = 0.15;
            double plane_thresh = -0.75;

            PinConstraint C1 = PinConstraint(fixed_corners, fixed_positions);
            PlaneConstraint C2 = PlaneConstraint(plane_thresh);
            CutHoleConstraint C3 = CutHoleConstraint(center, radius);
            SelfCollisionConstraint C4 = SelfCollisionConstraint();
            CombinedConstraints combined_constraint_fn = make_combined_constraints(C1, C2, C4);

            // Apply all forces and constraints to to the graph
            symp_euler_step(graph, t, dt, combined_force_fn, combined_constraint_fn);

            viewer.clear();
            node_map.clear();

            // Update viewer with nodes' new positions
            viewer.add_nodes(graph.node_begin(), graph.node_end(), node_map);
            viewer.add_edges(graph.edge_begin(), graph.edge_end(), node_map);
            viewer.set_label(t);

            // These lines slow down the animation for small graphs, like grid0_*.
            // Feel free to remove them or tweak the constants.
            if (graph.size() < 100)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        
            double total_runtime = clock.seconds();
            std::cout << "Time: " << total_runtime/5000.0 << std::endl;
        
    });  // simulation thread

    viewer.event_loop();

    // If we return from the event loop, we've killed the window.
    interrupt_sim_thread = true;
    sim_thread.join();

    return 0;
}
