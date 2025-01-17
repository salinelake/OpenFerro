FAQ
=====

Troubleshooting
--------------

**I got the warning "Constant folding an instruction is taking > *s"**

This is a warning from JAX, which means that JAX is folding the constant instruction. 
This may appear when you are running a simulation with a large lattice containing let's say 1 million sites. 
This is not a problem, and you can ignore it.


**The simulation takes a long time to run the first step**

This is because JAX, which OpenFerro is based on, will do a just-in-time (JIT) compilation of the code at the first execution of a function. 
The JIT compilation is a time-consuming process. When the lattice is large, the JIT compilation may take time more than running 1000 steps after the compilation. 
If you want an estimation of the compilation time, you can run the simulation with smaller lattices and plot the time cost with different lattice sizes. 



