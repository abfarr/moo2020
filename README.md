# moo2020
### Firefighter Problem Multi-Objective Optimization for AORS October 2020

The Firefighter Problem is a discrete time model of a fire spreading through the nodes of a graph as a firefighter defends against the fire. The Firefighter Problem has applications to a variety of contexts to include epidemiology and economics. Most objectives for the firefighter optimization problem are NP-hard to decide for real-world data graphs. Maximum nodes saved (MAXSAVE) and minimum time to containment (*t*-FIRE) are two common objectives. We modify the traditional construction of these individual objectives to simultaneously satisfy both MAXSAVE and *t*-FIRE with Pareto optimality. We examine the tradeoff between nodes saved and time to containment on data graphs. This work also begins to characterize the features of networks that make them resilient against malicious spread and determines computationally effective containment strategies.

Graph: Power Grid (http://networkrepository.com/inf-power.php)
Sub graphs of random segements of the network (simulate hacking at random) of no more than X size

Testing:
Single fire and firefighter (may test multiples)

MOO solution algorithms for ILP: TBD (NSGA-II, ...)
