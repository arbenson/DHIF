#include "DHIF.hpp"

namespace DHIF {
namespace mat_tools {

template<typename Scalar>
void ConvertSubmatrix
( Dense<Scalar>& D, const Sparse<Scalar>& S,
  int iStart, int jStart, int height, int width )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::ConvertSubmatrix (Dense,Sparse)");
#endif
    // Initialize the dense matrix to all zeros
    //TODO: set type for sparse matrix
    D.SetType( GENERAL );
#ifndef RELEASE
    if( D.Symmetric() && height != width )
        throw std::logic_error("Invalid submatrix of symmetric sparse matrix.");
#endif

    Vector<int> iidx(height);
    Vector<int> jidx(width);
    for( int i=0; i<height; ++i )
        iidx[i] = iStart+i;
    for( int j=0; j<width; ++j )
        jidx[j] = jStart+j;
    S.Find(iidx, jidx, D);
}

template void ConvertSubmatrix
(       Dense<float>& D,
  const Sparse<float>& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void ConvertSubmatrix
(       Dense<double>& D,
  const Sparse<double>& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void ConvertSubmatrix
(       Dense<std::complex<float> >& D,
  const Sparse<std::complex<float> >& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void ConvertSubmatrix
(       Dense<std::complex<double> >& D,
  const Sparse<std::complex<double> >& S,
  int iStart, int iEnd, int jStart, int jEnd );

} // namespace mat_tools
} // namespace DHIF
