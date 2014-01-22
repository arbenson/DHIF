#include "DHIF.hpp"

namespace DHIF {
namespace mat_tools {

// Dense C := alpha A B
template<typename Scalar>
void Multiply
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
                      Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Multiply (D := D D)");
#endif
    C.SetType( GENERAL );
    C.Resize( A.Height(), B.Width() );
    Multiply( alpha, A, B, Scalar(0), C );
}

// Dense C := alpha A B + beta C
template<typename Scalar>
void Multiply
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Multiply (D := D D + D)");
    if( A.Width() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( C.Height() != A.Height() || C.Width() != B.Width() )
        throw std::logic_error("C does not conform with AB");
    if( A.Symmetric() && B.Symmetric() )
        throw std::logic_error("Product of symmetric matrices not supported.");
    if( C.Symmetric() )
        throw std::logic_error("Update will probably not be symmetric.");
#endif
    if( A.Symmetric() )
    {
        blas::Symm
        ( 'L', 'L', C.Height(), C.Width(),
          alpha, A.LockedBuffer(), A.LDim(), B.LockedBuffer(), B.LDim(),
          beta, C.Buffer(), C.LDim() );
    }
    else if( B.Symmetric() )
    {
        blas::Symm
        ( 'R', 'L', C.Height(), C.Width(),
          alpha, B.LockedBuffer(), B.LDim(), A.LockedBuffer(), A.LDim(),
          beta, C.Buffer(), C.LDim() );
    }
    else
    {
        blas::Gemm
        ( 'N', 'N', C.Height(), C.Width(), A.Width(),
          alpha, A.LockedBuffer(), A.LDim(), B.LockedBuffer(), B.LDim(),
          beta, C.Buffer(), C.LDim() );
    }
}

// Dense C := alpha A B
template void Multiply
( float alpha, const Dense<float>& A,
               const Dense<float>& B,
                     Dense<float>& C );
template void Multiply
( double alpha, const Dense<double>& A,
                const Dense<double>& B,
                      Dense<double>& C );
template void Multiply
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Dense<std::complex<float> >& B,
                                   Dense<std::complex<float> >& C );
template void Multiply
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Dense<std::complex<double> >& B,
                                    Dense<std::complex<double> >& C );

// Dense C := alpha A B + beta C
template void Multiply
( float alpha, const Dense<float>& A,
               const Dense<float>& B,
  float beta,        Dense<float>& C );
template void Multiply
( double alpha, const Dense<double>& A,
                const Dense<double>& B,
  double beta,        Dense<double>& C );
template void Multiply
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Dense<std::complex<float> >& B,
  std::complex<float> beta,        Dense<std::complex<float> >& C );
template void Multiply
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Dense<std::complex<double> >& B,
  std::complex<double> beta,        Dense<std::complex<double> >& C );

} // namespace mat_tools
} // namespace DHIF
