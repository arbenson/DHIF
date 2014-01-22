#include "DHIF.hpp"

namespace DHIF {
namespace mat_tools {

template<typename Scalar>
void Invert( Dense<Scalar>& D )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Invert");
    if( D.Height() != D.Width() )
        throw std::logic_error("Tried to invert a non-square dense matrix.");
#endif
    const int n = D.Height();
    std::vector<int> ipiv( n );
    if( D.Symmetric() )
    {
        const int lworkLDLT = lapack::LDLTWorkSize( n );
        const int lworkInvertLDLT = lapack::InvertLDLTWorkSize( n );
        const int lwork = std::max( lworkLDLT, lworkInvertLDLT );
        std::vector<Scalar> work( lwork );

        lapack::LDLT( 'L', n, D.Buffer(), D.LDim(), &ipiv[0], &work[0], lwork );
        lapack::InvertLDLT( 'L', n, D.Buffer(), D.LDim(), &ipiv[0], &work[0] );
    }
    else
    {
        const int lwork = lapack::InvertLUWorkSize( n );
        std::vector<Scalar> work( lwork );

        lapack::LU( n, n, D.Buffer(), D.LDim(), &ipiv[0] );
        lapack::InvertLU( n, D.Buffer(), D.LDim(), &ipiv[0], &work[0], lwork );
    }
}

template void Invert( Dense<float>& D );
template void Invert( Dense<double>& D );
template void Invert( Dense<std::complex<float> >& D );
template void Invert( Dense<std::complex<double> >& D );

} // namespace mat_tools
} // namespace DHIF
