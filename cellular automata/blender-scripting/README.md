# 3D animations of Conway's Game of Life
Repository with classes and utils for rendering Conway's GOL in [Blender](https://www.blender.org).

### 3D Grid with Color Encoding
This option sees the encoding of a 3D GOL grid onto a static 2D grid, where color is used as index of the 3D GOL status. The color of each object in the static grid provides the information about the corresponding GOL "column" current configuration.
One issue is how to encode the "column" information as a color, especially if demanding the freedom on grid size (no constrained to a fixed side for the 3D GOL). Notice that the "column" information is a binary array, with a state of 1 or 0 corresponding to GOL cells states. Also consider as first basic color representation a RGB value, which can however be extended (see RGBA).

The first idea is to adapt to any array size and allowing len(array)/3 bits per color (BPC). Color value is then computed based on number of true bits out of the allowed size, considering a continuous array in the form [R*BPC G*BPC B*BPC]. Downside of this is that order doesn't matter, so for example, a value of [1,0,0,0] will generate same results as the value [0,0,0,1] (for all channels). This choice also limits the amount of representable colors to be equal to be BPC, instead of 2^BPC.

It naturally follows as next idea to use instead of the number of true bits out of the allowed size, the binary code number divided by the max representable code number. We consider only even BPC, meaning that BPC is always the int part of len(array)/3, and that len(array)%3 bits at the end will be discarded.

# Instructions
To run from Blender, first run *\__init__.py* or manually add the *src* folder path to Python system path.

Then run one of *conway_ND.py* file, which will instantiate your objects and frame change handlers. Next you should start the animation yourself from Blender, and stop it as desired. Finally remember to clean up all handlers if you want to play around more with the animation without having GOL updated reflected. This clean up can be obtained from the console via

    bpy.app.handlers.frame_change_pre.clear()
	


# TODO
* see how to animate a 3D and equivalent 2D projection aligned
* color as index of cell age (epochs for which it stayed alive)
* expand to 4D (possible of using color as encoder of the 4th dimension)
* analysis of population (avg num of live cells, survival time) for different config values

# Resources
* http://koaning.s3-website-us-west-2.amazonaws.com/html/blenderpython.html
* http://blenderscripting.blogspot.co.uk/2012/09/python-driven-animaion.html
* http://wiki.theprovingground.org/blender-py-mesh
* https://wiki.blender.org/index.php/Dev:Py/Scripts/Cookbook