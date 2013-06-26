
This is RTSC, the Real-Time Suggestive Contour viewer for 3D meshes.
http://www.cs.princeton.edu/gfx/proj/sugcon/

The code incorporates techniques and algorithms from the following papers.
If you don't understand what's going on, these papers might help :-)

    Doug DeCarlo, Adam Finkelstein, Szymon Rusinkiewicz, Anthony Santella.
    Suggestive Contours for Conveying Shape,
    ACM Transactions on Graphics (Proc. SIGGRAPH 2003), Vol. 22, No. 3,
        pp. 848-855, July, 2003.

    Doug DeCarlo, Adam Finkelstein, and Szymon Rusinkiewicz.
    Interactive Rendering of Suggestive Contours with Temporal Coherence,
    Symposium on Non-Photorealistic Animation and Rendering (NPAR),
        June, 2004. 

    Doug DeCarlo and Szymon Rusinkiewicz.
    Highlight Lines for Conveying Shape,
    Interactive Rendering of Suggestive Contours with Temporal Coherence,
    Symposium on Non-Photorealistic Animation and Rendering (NPAR),
        July, 2007. 

    Tilke Judd, Fredo Durand, and Edward Adelson.
    Apparent Ridges for Line Drawing,
    ACM Transactions on Graphics (Proc. SIGGRAPH 2007), Vol 26, No. 3,
        July, 2007. 

The authors of RTSC are Szymon Rusinkiewicz and Doug DeCarlo, and
the code is distributed under the GNU General Public License (GPL).


Usage:
------
        rtsc file.ply

The program takes a 3D triangle mesh as input.  Most flavors of PLY files
are supported, as are many OFF, 3DS, and Wavefront OBJ files.


Interacting with the program:
-----------------------------
If the mesh was loaded correctly, the program should start up showing a
contours-and-suggestive-contours view of the model.  You can change the
view using the mouse buttons as follows:

        Left:        Rotate (release while moving to spin the object)
        Middle:      Translate left, right, up, down
        Left+right:  Translate left, right, up, down
        Right:       Translate forward and back
        Mouse wheel: Translate forward and back
        Space bar:   Reset to initial view

Clicking the "Options" button opens a window with several checkboxes
that affect the behavior of the program.

 - The first column selects what lines will be drawn: the exterior
   silhouette, contours, suggestive contours, zeros of Gaussian
   curvature (K = 0), zeros of mean curvature (H = 0), lines at which
   the derivative of radial curvature in the projected view direction
   equals a threshold, ridge lines, valleys, mesh boundaries (i.e.,
   edges adjacent to only one triangle), and isophotes. 

 - The "Draw hidden lines" button disables z-buffering, so that
   backfacing and occluded lines are visible.

 - The "Trim inside contours" button affects whether contours with
   negative radial curvature are eliminated.  These will usually only be
   visible if "draw hidden lines" is checked.

 - The "Trim SC" button affects whether the (DwKr > thresh) test is
   applied to suggestive contours.

 - The "SC thresh" slider changes the DwKr threshold that is used.

 - The "Trim RV" button affects whether ridges with negative maximum
   curvature or valleys with positive maximum curvature are drawn.

 - The "RV thresh" slider changes the curvature threshold used for trimming
   ridge and valley lines.

 - The third column affects how lines are drawn.  You can select whether
   to use the texture-map algorithm, whether to draw lines in a single
   weight or fade them out, whether to differentiate between contours
   and suggestive contours by drawing them in different colors (green
   and blue, respectively), and whether to use Hermite interpolation for
   slightly nicer lines when zoomed in.

 - The bottom section of the third color affects mesh coloring and whether
   edges of the mesh are drawn (i.e., a wireframe).  The color can be
   constant white or gray, a color-coded visualization of principal
   curvatures (red = both +; yellow = +,0; green = +,-; cyan = 0,-;
   blue = both -), or a grayscale visualization of mean curvature.

 - The fourth column controls lighting.  You can select between Lambertian
   shading (using gamma=1 or gamma=2), a hemisphere of light (which eliminates
   the complete shadowing of the back when the light is not at the camera),
   a toon shader, or a Gooch-inspired yellow-to-blue color ramp.  The
   trackball widget controls lighting direction, and you can select whether
   the light is fixed relative to the camera or the object.

 - The next column lets you turn on per-vertex vectors (normals, principal
   directions, etc.), or switch to "dual viewport" mode in which you can
   draw lines (such as contours) relative to one viewpoint while looking at
   them from another.

 - The final column contains buttons that let you apply various algorithms
   to the mesh.  You can smooth the mesh geometry, normals, curvatures,
   or derivatives of curvature separately, or do an iteration of Loop
   subdivision (all of these help in generating nice lines from noisy
   and/or coarsely tesselated meshes).  The "screencap" button captures
   an image to a PPM file named "img###.ppm" in the current directory.


Interesting things to try:
--------------------------

 - Turn off "trim SC" and turn on "Draw in color" and "DwKr = thresh".
   You can see how the Kr = 0 loops (in blue) are clipped to the
   regions where DwKr > thresh (bounded by the red lines).

 - Try turning on the "K = 0" lines (zeros of Gaussian curvature).
   Notice how the suggestive contours tend to be located close by,
   as you rotate the object.

 - Seeing the effects of Hermite interpolation using "Hermite interp"
   requires that you zoom in sufficiently to see wiggles in the lines.
   Turning on "Draw edges" might also help explain some of the behavior
   you see.


Compiling the code:
-------------------

The code is written in C++, and is known to compile using a recent (3.x)
version of g++.  (Note that g++ versions 3.4.2 and 4.0.0 are known to
miscompile rtsc.)  Compiling under Windows is possible using Cygwin or
Mingw32.  Other compilers have not been tested - let us know of any
successes or failures.

RTSC uses the trimesh2 library, as well as the freeglut and GLUI
libraries distributed with it.  The whole package is available at
http://www.cs.princeton.edu/gfx/proj/trimesh2/

A sample GNU Makefile is provided - be sure to set the "TRIMESHDIR"
variable to point to the trimesh2 directory.  You can adjust compiler
parameters as needed in the Makerules.* files distributed with trimesh2.


 - Szymon Rusinkiewicz
   smr@princeton.edu

