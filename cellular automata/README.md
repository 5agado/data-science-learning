# Cellular Automata
or "discrete mathematical models of Artificial Life"

The repository contains a collection of classes and utils to generate, evolve and render multiple types of cellular automata. Rendering is mostly accomplished via [Blender](https://www.blender.org).

The code started with purely Conway's Game of Life in mind, but is now generalize to N-dimensional automata.

### 3D Grid with Color Encoding
This option sees the encoding of a 3D automaton grid onto a static 2D grid, where color is used as index of a cell status. The color of each object in the static grid provides the information about the corresponding "column" current configuration.
One issue is how to encode the "column" information as a color, especially if demanding the freedom on grid size (now constrained to a fixed side for the 3D version). Notice that the "column" information is a binary array, with a state of 1 or 0 corresponding to the cells states. Also consider as first basic color representation a RGB value, which can however be extended (see RGBA).

The first idea is to adapt to any array size and allowing len(array)/3 bits per color (BPC). Color value is then computed based on number of true bits out of the allowed size, considering a continuous array in the form ``[R*BPC G\*BPC B\*BPC]``. Downside of this is that order doesn't matter, so for example, a value of [1,0,0,0] will generate same results as the value ``[0,0,0,1]`` (for all channels). This choice also limits the amount of representable colors to be equal to be BPC, instead of 2^BPC.

It naturally follows as next idea to use instead of the number of true bits out of the allowed size, the binary code number divided by the max representable code number. We consider only even BPC, meaning that BPC is always the int part of len(array)/3, and that len(array)%3 bits at the end will be discarded.