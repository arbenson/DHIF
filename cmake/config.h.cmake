#pragma once
#ifndef DHIF_CONFIG_H
#define DHIF_CONFIG_H 1

#define DHIF_VERSION_MAJOR @DHIF_VERSION_MAJOR@
#define DHIF_VERSION_MINOR @DHIF_VERSION_MINOR@
#cmakedefine RELEASE
#cmakedefine TIME_MULTIPLY
#cmakedefine MEMORY_INFO
#cmakedefine BLAS_POST
#cmakedefine LAPACK_POST
#cmakedefine HAVE_QT5
#cmakedefine AVOID_COMPLEX_MPI
#cmakedefine HAVE_MPI_IN_PLACE

#define RESTRICT @RESTRICT@

#endif /* ifndef DHIF_CONFIG_H */
