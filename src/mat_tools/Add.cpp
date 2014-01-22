#include "DHIF.hpp"

namespace DHIF {
namespace mat_tools {

// Dense C := alpha A + beta B
template<typename Scalar>
void Add
( Scalar alpha, const Dense<Scalar>& A,
  Scalar beta,  const Dense<Scalar>& B,
                      Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Add (D := D + D)");
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        throw std::logic_error("Tried to add nonconforming matrices.");
#endif
    const int m = A.Height();
    const int n = A.Width();

    if( A.Type() == B.Type() )
        C.SetType( A.Type() );
    else
        C.SetType( GENERAL );

    C.Resize( m, n );

    if( C.Symmetric() )
    {
        for( int j=0; j<n; ++j )
        {
            const Scalar* RESTRICT ACol = A.LockedBuffer(0,j);
            const Scalar* RESTRICT BCol = B.LockedBuffer(0,j);
            Scalar* RESTRICT CCol = C.Buffer(0,j);
            for( int i=j; i<m; ++i )
                CCol[i] = alpha*ACol[i] + beta*BCol[i];
        }
    }
    else
    {
        for( int j=0; j<n; ++j )
        {
            const Scalar* RESTRICT ACol = A.LockedBuffer(0,j);
            const Scalar* RESTRICT BCol = B.LockedBuffer(0,j);
            Scalar* RESTRICT CCol = C.Buffer(0,j);
            for( int i=0; i<m; ++i )
                CCol[i] = alpha*ACol[i] + beta*BCol[i];
        }
    }
}

// Dense C := alpha A + beta B
template<typename Scalar>
void Axpy
( Scalar alpha, const Dense<Scalar>& A,
                Dense<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Axpy (D := D + D)");
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        throw std::logic_error("Tried to add nonconforming matrices.");
#endif
    const int m = A.Height();
    const int n = A.Width();

    for( int j=0; j<n; ++j )
    {
        const Scalar* RESTRICT ACol = A.LockedBuffer(0,j);
        Scalar* RESTRICT BCol = B.Buffer(0,j);
        for( int i=0; i<m; ++i )
            BCol[i] = alpha*ACol[i] + BCol[i];
    }
}

// Dense C := alpha A + beta B
template void Add
( float alpha, const Dense<float>& A,
  float beta,  const Dense<float>& B,
                     Dense<float>& C );
template void Add
( double alpha, const Dense<double>& A,
  double beta,  const Dense<double>& B,
                      Dense<double>& C );
template void Add
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
  std::complex<float> beta,  const Dense<std::complex<float> >& B,
                                   Dense<std::complex<float> >& C );
template void Add
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
  std::complex<double> beta,  const Dense<std::complex<double> >& B,
                                    Dense<std::complex<double> >& C );

// Dense C := alpha A + beta B
template void Axpy
( float alpha, const Dense<float>& A,
               Dense<float>& B );
template void Axpy
( double alpha, const Dense<double>& A,
                Dense<double>& B );
template void Axpy
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             Dense<std::complex<float> >& B );
template void Axpy
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              Dense<std::complex<double> >& B );

} // namespace mat_tools
} // namespace DHIF
