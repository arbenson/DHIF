#include "DHIF.hpp"

namespace DHIF {
namespace mat_tools {

// Dense C := alpha A^T B
template<typename Scalar>
void TransposeMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
                      Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::TransposeMultiply (D := D^T D)");
#endif
    C.SetType( GENERAL );
    C.Resize( A.Width(), B.Width() );
    TransposeMultiply( alpha, A, B, Scalar(0), C );
}

// Dense C := alpha A^T B + beta C
template<typename Scalar>
void TransposeMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::TransposeMultiply (D := D^T D + D)");
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( B.Symmetric() )
        throw std::logic_error("BLAS does not support symm times trans");
#endif
    if( A.Symmetric() )
    {
        blas::Symm
        ( 'L', 'L', C.Height(), C.Width(),
          alpha, A.LockedBuffer(), A.LDim(), B.LockedBuffer(), B.LDim(),
          beta, C.Buffer(), C.LDim() );
    }
    else
    {
        blas::Gemm
        ( 'T', 'N', C.Height(), C.Width(), A.Height(),
          alpha, A.LockedBuffer(), A.LDim(), B.LockedBuffer(), B.LDim(),
          beta, C.Buffer(), C.LDim() );
    }
}

// Dense C := alpha A^T B
template void TransposeMultiply
( float alpha, const Dense<float>& A,
               const Dense<float>& B,
                     Dense<float>& C );
template void TransposeMultiply
( double alpha, const Dense<double>& A,
                const Dense<double>& B,
                      Dense<double>& C );
template void TransposeMultiply
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Dense<std::complex<float> >& B,
                                   Dense<std::complex<float> >& C );
template void TransposeMultiply
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Dense<std::complex<double> >& B,
                                    Dense<std::complex<double> >& C );

// Dense C := alpha A^T B + beta C
template void TransposeMultiply
( float alpha, const Dense<float>& A,
               const Dense<float>& B,
  float beta,        Dense<float>& C );
template void TransposeMultiply
( double alpha, const Dense<double>& A,
                const Dense<double>& B,
  double beta,        Dense<double>& C );
template void TransposeMultiply
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Dense<std::complex<float> >& B,
  std::complex<float> beta,        Dense<std::complex<float> >& C );
template void TransposeMultiply
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Dense<std::complex<double> >& B,
  std::complex<double> beta,        Dense<std::complex<double> >& C );

} // namespace mat_tools
} // namespace DHIF
