#include "DHIF.hpp"

namespace DHIF {
namespace mat_tools {

// Dense y := alpha A^H x + beta y
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::AdjointMultiply (y := D^H x + y)");
#endif
    if( A.Symmetric() )
    {
        Vector<Scalar> xConj;
        Conjugate( x, xConj );
        Conjugate( y );
        blas::Symv
        ( 'L', A.Height(),
          Conj(alpha), A.LockedBuffer(), A.LDim(),
                       xConj.Buffer(),   1,
          Conj(beta),  y.Buffer(),       1 );
        Conjugate( y );
    }
    else
    {
        blas::Gemv
        ( 'C', A.Height(), A.Width(),
          alpha, A.LockedBuffer(), A.LDim(),
                 x.LockedBuffer(), 1,
          beta,  y.Buffer(),       1 );
    }
}

// Dense y := alpha A^H x
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const Vector<Scalar>& x,
                      Vector<Scalar>& y )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::AdjointMultiply (y := D^H x)");
#endif
    y.Resize( A.Width() );
    if( A.Symmetric() )
    {
        Vector<Scalar> xConj;
        Conjugate( x, xConj );
        blas::Symv
        ( 'L', A.Height(),
          Conj(alpha), A.LockedBuffer(), A.LDim(),
                       xConj.Buffer(),   1,
          0,           y.Buffer(),       1 );
        Conjugate( y );
    }
    else
    {
        blas::Gemv
        ( 'C', A.Height(), A.Width(),
          alpha, A.LockedBuffer(), A.LDim(),
                 x.LockedBuffer(), 1,
          0,     y.Buffer(),       1 );
    }
}

template void AdjointMultiply
( float alpha, const Dense<float>& A,
               const Vector<float>& x,
  float beta,        Vector<float>& y );
template void AdjointMultiply
( double alpha, const Dense<double>& A,
                const Vector<double>& x,
  double beta,        Vector<double>& y );
template void AdjointMultiply
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Vector<std::complex<float> >& x,
  std::complex<float> beta,        Vector<std::complex<float> >& y );
template void AdjointMultiply
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Vector<std::complex<double> >& x,
  std::complex<double> beta,        Vector<std::complex<double> >& y );

template void AdjointMultiply
( float alpha, const Dense<float>& A,
               const Vector<float>& x,
                     Vector<float>& y );
template void AdjointMultiply
( double alpha, const Dense<double>& A,
                const Vector<double>& x,
                      Vector<double>& y );
template void AdjointMultiply
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Vector<std::complex<float> >& x,
                                   Vector<std::complex<float> >& y );
template void AdjointMultiply
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Vector<std::complex<double> >& x,
                                    Vector<std::complex<double> >& y );

} // namespace mat_tools
} // namespace DHIF
