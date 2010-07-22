External Packages for Topographica

This directory contains source-code packages for external
(non-topographica) software needed for topographica development,
including such things as Python, Numeric, PIL, etc.

Each source package is included as a gzipped tar file.  The Makefile
will untar each package, build it, and install it in the proper place
in the development directory (e.g. $HOME/topographica).  In the
process, it creates the standard unix installation directories: bin,
lib, man, include, etc.


CONVENTIONS FOR ADDING PACKAGES:

To make writing the makefile easier, packages should be added as
Gzipped tar files with the same name as the top-level directory inside
the tar file, followed by the extension .tgz.  Conventions for writing
makefile entries on new packages are forthcoming.  For now, use one of
the existing Makefile sections as an example.
