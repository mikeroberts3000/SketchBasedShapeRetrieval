SketchBasedShapeRetrieval
=========================

Easy-to-read C++/Python implementation of the paper <a href='http://cybertron.cg.tu-berlin.de/eitz/projects/sbsr/'>Sketch-Based Shape Retrieval</a>.

#### Generating Line Drawings

To generate line drawings, I hacked the <a href='http://gfx.cs.princeton.edu/gfx/proj/sugcon/'>Suggestive Contours</a> software to take a 3D mesh file as command-line input and to produce a series of images as output. The images show the mesh from different views rendered with the <a href='http://people.csail.mit.edu/tjudd/apparentridges.html'>Apparent Ridges</a> algorithm.

Note that the <a href='http://gfx.cs.princeton.edu/gfx/proj/sugcon/'>Suggestive Contours</a> software implements several other line drawing algorithms. Moreover, in the interest of simplicity, I sample the space of views uniformally rather than performing importance sampling.

The code for generating line drawings is checked into the LineDrawingExporter folder.

#### Running the Shape Matching Pipeline

The rest of the shape matching pipeline can be browsed online with the <a href='http://nbviewer.ipython.org/'>IPython Notebook Viewer</a> using the links below.

1. <a href='http://nbviewer.ipython.org/urls/raw.github.com/mroberts3000/SketchBasedShapeRetrieval/master/SketchBasedShapeRetrieval/IPython/01 Generate Gabor Filters.ipynb'>Generate Gabor Filter Images</a>
2. <a href='http://nbviewer.ipython.org/urls/raw.github.com/mroberts3000/SketchBasedShapeRetrieval/master/SketchBasedShapeRetrieval/IPython/02 Compute Gabor Filter Responses.ipynb'>Compute Gabor Filter Responses</a>
3. <a href='http://nbviewer.ipython.org/urls/raw.github.com/mroberts3000/SketchBasedShapeRetrieval/master/SketchBasedShapeRetrieval/IPython/03 Compute GALIF Local Patch Features.ipynb'>Compute GALIF Local Patch Features</a>
4. <a href='http://nbviewer.ipython.org/urls/raw.github.com/mroberts3000/SketchBasedShapeRetrieval/master/SketchBasedShapeRetrieval/IPython/04 Sample GALIF Local Patch Features.ipynb'>Sample GALIF Local Patch Features</a>
5. <a href='http://nbviewer.ipython.org/urls/raw.github.com/mroberts3000/SketchBasedShapeRetrieval/master/SketchBasedShapeRetrieval/IPython/05 Cluster Sampled GALIF Local Patch Features.ipynb'>Cluster Sampled GALIF Local Patch Features</a>
6. <a href='http://nbviewer.ipython.org/urls/raw.github.com/mroberts3000/SketchBasedShapeRetrieval/master/SketchBasedShapeRetrieval/IPython/06 Compute Global Features.ipynb'>Compute Global Features</a>
7. <a href='http://nbviewer.ipython.org/urls/raw.github.com/mroberts3000/SketchBasedShapeRetrieval/master/SketchBasedShapeRetrieval/IPython/07 Compute Matches from Input Sketch.ipynb'>Compute Matches from Input Sketch</a>
8. <a href='http://nbviewer.ipython.org/urls/raw.github.com/mroberts3000/SketchBasedShapeRetrieval/master/SketchBasedShapeRetrieval/IPython/08 Compute Matches from Input Sketch (Cython).ipynb'>Compute Matches from Input Sketch using Cython for Extra Speed</a>

There is also some useful debugging code that you can find by digging through the source. As above, you can browse this code online using the <a href='http://nbviewer.ipython.org/'>IPython Notebook Viewer</a>.

[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/3df96319dd517d78163c9bf0d26a1047 "githalytics.com")](http://githalytics.com/mroberts3000/SketchBasedShapeRetrieval)
