## For detailed coursework, please refer to the 'Assignment Prompts' folder

### This is the README file for the final homework (hw4)
Documenting Runtime:

By the end of HW4 Part 3.2, we are asked to document the runtime of the sequential version of symp_euler_step versus its parallel version (of HW4) for given data grids. 
\
The timer is set up in main() as below:\
(line 754) CME212::Clock clock;\
(line 756) double dt = 0.001;\
(line 757) double t_start = 0;\
(line 758) double t_end = 5.0;\
(line 760) clock.start();\
(line 761) for (double t = t_start; t < t_end && !interrupt_sim_thread; t += dt) {\
...(line 792) //end for loop \
(line 794) double total_runtime = clock.seconds();\
(line 795) std::cout << "Time: " << total_runtime/5000.0 << std::endl;\
\
Particularly, this process times how long it takes to run symp_euler_step for 5000 time steps given the combined forces of GravityForce() and MassSpringForce()  as well as the combined constraints of PinConstraint with positions (0,0,0) and (1,0,0) and PlaneConstraint(-0.75) and SelfCollisionConstraint(). Finally, the average computation time per individual time step is printed to command line.
\
The results on my laptop (quad cores, per 5000 steps) are the following, \
[Sequential] ./mass_spring data/grid0.nodes data/grid0.tets\
0.0560164\
[Sequential] ./mass_spring data/grid1.nodes data/grid1.tets\
0.962762\
[Sequential] ./mass_spring data/grid2.nodes data/grid2.tets\
3.03862\
[Sequential] ./mass_spring data/grid3.nodes data/grid3.tets\
12.9836\
[Sequential] ./mass_spring data/grid4.nodes data/grid4.tets\
1304.38\
\
[Parallel] ./mass_spring data/grid0.nodes data/grid0.tets\
0.103201\
[Parallel] ./mass_spring data/grid1.nodes data/grid1.tets\
0.813502\
[Parallel] ./mass_spring data/grid2.nodes data/grid2.tets\
1.52245\
[Parallel] ./mass_spring data/grid3.nodes data/grid3.tets\
4.65147\
[Parallel] ./mass_spring data/grid4.nodes data/grid4.tets\
384.996\
\
From this, we can see that if the grid size is too small (like grid0, grid1), the parallelized version does not outperform the sequential version; however, with enough grid size (starting from grid2 onward), we could see significant improvement (especially on grid 4). 
