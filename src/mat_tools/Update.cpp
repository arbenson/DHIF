#include "DHIF.hpp"

namespace DHIF {
namespace mat_tools {

// Dense B := alpha A + beta B
template<typename Scalar>
void Update
( Scalar alpha, const Dense<Scalar>& A,
  Scalar beta,        Dense<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Update (D := D + D)");
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        throw std::logic_error("Tried to update with nonconforming matrices.");
    // TODO: Allow for A to be symmetric when B is general
    if( A.Symmetric() && B.General() )
        throw std::logic_error("A-symmetric/B-general not yet implemented.");
    if( A.General() && B.Symmetric() )
        throw std::logic_error
        ("Cannot update a symmetric matrix with a general one");
#endif
    const int m = A.Height();
    const int n = A.Width();
    if( A.Symmetric() )
    {
        for( int j=0; j<n; ++j )
        {
            Scalar* RESTRICT BCol = B.Buffer(0,j);
            const Scalar* RESTRICT ACol = A.LockedBuffer(0,j);
            for( int i=j; i<m; ++i )
                BCol[i] = alpha*ACol[i] + beta*BCol[i];
        }
    }
    else
    {
        for( int j=0; j<n; ++j )
        {
            Scalar* RESTRICT BCol = B.Buffer(0,j);
            const Scalar* RESTRICT ACol = A.LockedBuffer(0,j);
            for( int i=0; i<m; ++i )
                BCol[i] = alpha*ACol[i] + beta*BCol[i];
        }
    }
}

// Dense update B := alpha A + beta B
template void Update
( float alpha, const Dense<float>& A,
  float beta,        Dense<float>& B );
template void Update
( double alpha, const Dense<double>& A,
  double beta,        Dense<double>& B );
template void Update
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
  std::complex<float> beta,        Dense<std::complex<float> >& B );
template void Update
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
  std::complex<double> beta,        Dense<std::complex<double> >& B );

} // namespace mat_tools
} // namespace DHIF
