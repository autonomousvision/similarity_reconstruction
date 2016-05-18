ply2vri
release 1.1
October 25, 2002
http://grail.cs.washington.edu/software-data/ply2vri


*** For license information, please see license.txt. ***


Overview
========

This is a simple command line tool for converting triangle meshes in PLY 
format into volumetric grids in VRI format.


Getting it to run
=================

For Windows users, and executable has been provided (ply2vri.exe).  If you 
want to compile the source code, the Visual Studio v6 project files are 
included.

Under Linux, run make.  With luck, an executable will be produced.

To run the tool, type "ply2vri foo.ply foo.vri" from a command prompt, 
where "foo.ply" is the name of the PLY file to read, and "foo.vri" is 
the name of the VRI file to output.  More options are avaiable; run
"ply2vri" with no parameters to see a list.

Additional documentation can be found on the web page:
  http://grail.cs.washington.edu/software-data/ply2vri


Problems?
=========

Contact Brett Allen (allen@cs.washington.edu).


Release History
===============

Version 1.1 - Oct 25, 2002
  - changed command line parameter syntax
  - fixed a bug that would cause noise and flipped signs
  - added per-voxel algorithm
  - added template VRI support

Version 1.0 - Oct 16, 2002
  - initial release
