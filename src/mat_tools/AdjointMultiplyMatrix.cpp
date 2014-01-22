#include "DHIF.hpp"

namespace DHIF {
namespace mat_tools {

// Dense C := alpha A^H B
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
                      Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::AdjointMultiply (D := D^H D)");
#endif
    C.SetType( GENERAL );
    C.Resize( A.Width(), B.Width() );
    AdjointMultiply( alpha, A, B, Scalar(0), C );
}

// Dense C := alpha A^H B + beta C
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry
    ("mat_tools::AdjointMultiply (D := D^H D + D)");
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( B.Symmetric() )
        throw std::logic_error("BLAS does not support symm times trans.");
    if( C.Symmetric() )
        throw std::logic_error("Update will probably not be symmetric.");
#endif
    if( A.Symmetric() )
    {
        const int m = A.Height();
        const int n = B.Width();
        if( m <= 2*n )
        {
            // C := alpha A^H B + beta C
            //    = alpha conj(A) B + beta C
            //
            // AConj := conj(A)
            // C := alpha AConj B + beta C

            Dense<Scalar> AConj(m,m);
            Conjugate( A, AConj );
            blas::Symm
            ( 'L', 'L', C.Height(), C.Width(),
              alpha, AConj.LockedBuffer(), AConj.LDim(),
                     B.LockedBuffer(), B.LDim(),
              beta,  C.Buffer(), C.LDim() );
        }
        else
        {
            // C := alpha A^H B + beta C
            //    = alpha conj(A) B + beta C
            //    = conj(conj(alpha) A conj(B) + conj(beta) conj(C))
            //
            // BConj := conj(B)
            // CConj := conj(C)
            // C := conj(alpha) A BConj + conj(beta) CConj
            // C := conj(C)

            Dense<Scalar> BConj(m,n);
            Conjugate( B, BConj );
            Conjugate( C );
            blas::Symm
            ( 'L', 'L', C.Height(), C.Width(),
              Conj(alpha), A.LockedBuffer(), A.LDim(),
                           BConj.LockedBuffer(), BConj.LDim(),
              Conj(beta),  C.Buffer(), C.LDim() );
            Conjugate( C );
        }
    }
    else
    {
        blas::Gemm
        ( 'C', 'N', C.Height(), C.Width(), A.Height(),
          alpha, A.LockedBuffer(), A.LDim(), B.LockedBuffer(), B.LDim(),
          beta, C.Buffer(), C.LDim() );
    }
}

// Dense C := alpha A^H B
template void AdjointMultiply
( float alpha, const Dense<float>& A,
               const Dense<float>& B,
                     Dense<float>& C );
template void AdjointMultiply
( double alpha, const Dense<double>& A,
                const Dense<double>& B,
                      Dense<double>& C );
template void AdjointMultiply
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Dense<std::complex<float> >& B,
                                   Dense<std::complex<float> >& C );
template void AdjointMultiply
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Dense<std::complex<double> >& B,
                                    Dense<std::complex<double> >& C );

// Dense C := alpha A^H B + beta C
template void AdjointMultiply
( float alpha, const Dense<float>& A,
               const Dense<float>& B,
  float beta,        Dense<float>& C );
template void AdjointMultiply
( double alpha, const Dense<double>& A,
                const Dense<double>& B,
  double beta,        Dense<double>& C );
template void AdjointMultiply
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Dense<std::complex<float> >& B,
  std::complex<float> beta,        Dense<std::complex<float> >& C );
template void AdjointMultiply
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Dense<std::complex<double> >& B,
  std::complex<double> beta,        Dense<std::complex<double> >& C );

} // namespace mat_tools
} // namespace DHIF
