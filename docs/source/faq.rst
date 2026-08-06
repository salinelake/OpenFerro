FAQ
=====

Troubleshooting
---------------

**JAX reports that constant folding is taking a long time**

Constant folding occurs during JIT compilation. For large captured constants,
including an Ewald kernel, the warning can indicate material startup time or
memory use. Measure compilation separately from steady-state execution and do
not dismiss repeated warnings without checking a compile profile and array
placement.


**The simulation takes a long time to run the first step**

This is because JAX, which OpenFerro is based on, will do a just-in-time (JIT) compilation of the code at the first execution of a function. 
The JIT compilation is a time-consuming process. When the lattice is large, the JIT compilation may take time more than running 1000 steps after the compilation. 
If you want an estimation of the compilation time, you can run the simulation with smaller lattices and plot the time cost with different lattice sizes. 



