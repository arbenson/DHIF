#include "DHIF.hpp"

namespace DHIF {
namespace mat_tools {

// Dense C := alpha A B^H
template<typename Scalar>
void MultiplyAdjoint
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
                      Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::MultiplyAdjoint (D := D D^H)");
#endif
    C.SetType( GENERAL );
    C.Resize( A.Height(), B.Height() );
    MultiplyAdjoint( alpha, A, B, Scalar(0), C );
}

// Dense C := alpha A B^H + beta C
template<typename Scalar>
void MultiplyAdjoint
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry
    ("mat_tools::MultiplyAdjoint (D := D D^H + D)");
    if( A.Width() != B.Width() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( A.Height() != C.Height() )
        throw std::logic_error("The height of A and C are nonconformal.");
    if( B.Height() != C.Width() )
        throw std::logic_error("The width of B and C are nonconformal.");
    if( B.Symmetric() || A.Symmetric() )
        throw std::logic_error("BLAS does not support symm times trans.");
    if( C.Symmetric() )
        throw std::logic_error("Update will probably not be symmetric.");
#endif
    blas::Gemm
    ( 'N', 'C', C.Height(), C.Width(), A.Width(),
      alpha, A.LockedBuffer(), A.LDim(), B.LockedBuffer(), B.LDim(),
      beta, C.Buffer(), C.LDim() );
}

// Dense C := alpha A B^H
template void MultiplyAdjoint
( float alpha, const Dense<float>& A,
               const Dense<float>& B,
                     Dense<float>& C );
template void MultiplyAdjoint
( double alpha, const Dense<double>& A,
                const Dense<double>& B,
                      Dense<double>& C );
template void MultiplyAdjoint
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Dense<std::complex<float> >& B,
                                   Dense<std::complex<float> >& C );
template void MultiplyAdjoint
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Dense<std::complex<double> >& B,
                                    Dense<std::complex<double> >& C );

// Dense C := alpha A B^H + beta C
template void MultiplyAdjoint
( float alpha, const Dense<float>& A,
               const Dense<float>& B,
  float beta,        Dense<float>& C );
template void MultiplyAdjoint
( double alpha, const Dense<double>& A,
                const Dense<double>& B,
  double beta,        Dense<double>& C );
template void MultiplyAdjoint
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Dense<std::complex<float> >& B,
  std::complex<float> beta,        Dense<std::complex<float> >& C );
template void MultiplyAdjoint
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Dense<std::complex<double> >& B,
  std::complex<double> beta,        Dense<std::complex<double> >& C );

} // namespace mat_tools
} // namespace DHIF
