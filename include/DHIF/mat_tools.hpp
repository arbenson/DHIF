#pragma once
#ifndef DHIF_MAT_TOOLS_HPP
#define DHIF_MAT_TOOLS_HPP 1

#include <complex>
#include <cmath>
#include <cstdlib> // for integer abs
#include <vector>

#include "DHIF/core/timer.hpp"

#include "DHIF/core/blas.hpp"
#include "DHIF/core/lapack.hpp"

#include "DHIF/core/dense.hpp"
#include "DHIF/core/vector.hpp"

#include "DHIF/core/sparse.hpp"

#include "DHIF/core/random.hpp"

namespace DHIF {
namespace mat_tools {

/*
 *  Convert a subset of a sparse matrix to dense/low-rank form
 */
template<typename Scalar>
void ConvertSubmatrix
( Dense<Scalar>& D, const Sparse<Scalar>& S,
  int iStart, int jStart, int height, int width );

/*
 *  Generalized addition of two dense/low-rank matrices, C := alpha A + beta B
 */
// D := alpha D + beta D
template<typename Scalar>
void Add
( Scalar alpha, const Dense<Scalar>& A,
  Scalar beta,  const Dense<Scalar>& B,
                      Dense<Scalar>& C );
// D := alpha D + beta D
template<typename Scalar>
void Axpy
( Scalar alpha, const Dense<Scalar>& A,
                Dense<Scalar>& B );

/*
 *  Generalized update of two dense/low-rank matrices, B := alpha A + beta B
 */
// D := alpha D + beta D
template<typename Scalar>
void Update
( Scalar alpha, const Dense<Scalar>& A,
  Scalar beta,        Dense<Scalar>& B );

/*
 *  Matrix Matrix multiply, C := alpha A B
 *
 *  When the resulting matrix is dense, an update form is also provided, i.e.,
 *  C := alpha A B + beta C
 */
// D := alpha D D
template<typename Scalar>
void Multiply
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
                      Dense<Scalar>& C );
// D := alpha D D + beta D
template<typename Scalar>
void Multiply
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C );

/*
 *  Matrix Transpose Matrix Multiply, C := alpha A^T B
 *
 *  When the resulting matrix is dense, an update form is also provided, i.e.,
 *  C := alpha A^T B + beta C
 */
// D := alpha D^T D
template<typename Scalar>
void TransposeMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
                      Dense<Scalar>& C );
// D := alpha D^T D + beta D
template<typename Scalar>
void TransposeMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C );

/*
 *  Matrix Matrix Transpose Multiply, C := alpha A B^T
 *
 *  When the resulting matrix is dense, an update form is also provided, i.e.,
 *  C := alpha A B^T + beta C
 */
// D := alpha D D^T
template<typename Scalar>
void MultiplyTranspose
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
                      Dense<Scalar>& C );
// D := alpha D D^T + beta D
template<typename Scalar>
void MultiplyTranspose
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C );

/*
 *  Matrix-Adjoint Matrix Multiply, C := alpha A^H B
 *
 *  When the resulting matrix is dense, an update form is also provided, i.e.,
 *  C := alpha A^H B + beta C
 */
// D := alpha D^H D
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
                      Dense<Scalar>& C );
// D := alpha D^H D + beta D
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C );

/*
 *  Matrix Matrix-Adjoint Multiply, C := alpha A B^H
 *
 *  When the resulting matrix is dense, an update form is also provided, i.e.,
 *  C := alpha A B^H + beta C
 */
// D := alpha D D^H
template<typename Scalar>
void MultiplyAdjoint
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
                      Dense<Scalar>& C );
// D := alpha D D^H + beta D
template<typename Scalar>
void MultiplyAdjoint
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C );

/*
 *  Matrix-Vector multiply, y := alpha A x + beta y
 */
// y := alpha D x + beta y
template<typename Scalar>
void Multiply
( Scalar alpha, const Dense<Scalar>& D,
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y );

/*
 *  Matrix-Vector multiply, y := alpha A x
 */
// y := alpha D x
template<typename Scalar>
void Multiply
( Scalar alpha, const Dense<Scalar>& D,
                const Vector<Scalar>& x,
                      Vector<Scalar>& y );

/*
 *  Matrix-Transpose-Vector multiply, y := alpha A^T x + beta y
 */
// y := alpha D^T x + beta y
template<typename Scalar>
void TransposeMultiply
( Scalar alpha, const Dense<Scalar>& D,
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y );

/*
 *  Matrix-Transpose-Vector multiply, y := alpha A^T x
 */
// y := alpha D^T x
template<typename Scalar>
void TransposeMultiply
( Scalar alpha, const Dense<Scalar>& D,
                const Vector<Scalar>& x,
                      Vector<Scalar>& y );

/*
 *  Matrix-Hermitian-Transpose-Vector multiply, y := alpha A^H x + beta y
 */
// y := alpha D^H x + beta y
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const Dense<Scalar>& D,
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y );

/*
 *  Matrix-Hermitian-Transpose-Vector multiply, y := alpha A^H x
 */
// y := alpha D^H x
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const Dense<Scalar>& D,
                const Vector<Scalar>& x,
                      Vector<Scalar>& y );

/*
 *  Dense inversion, D := inv(D)
 */
template<typename Scalar>
void Invert( Dense<Scalar>& D );

/*
 *  Compute a vector's two-norm
 */
template<typename Real>
Real TwoNorm( const Vector<Real>& x );
template<typename Real>
Real TwoNorm( const Vector< std::complex<Real> >& x );

/*
 *  Scale a vector or matrix
 */
template<typename Scalar>
void Scale( Scalar alpha, Vector<Scalar>& x );
template<typename Scalar>
void Scale( Scalar alpha, Dense<Scalar>& D );

/*
 *  Copy a vector or matrix
 */
template<typename Scalar>
void Copy( const Vector<Scalar>& x, Vector<Scalar>& y );
template<typename Scalar>
void Copy( const std::vector<Scalar>& x, std::vector<Scalar>& y );
template<typename Scalar>
void Copy( const Vector<Scalar>& x, std::vector<Scalar>& y );
template<typename Scalar>
void Copy( const std::vector<Scalar>& x, Vector<Scalar>& y );
template<typename Scalar>
void Copy( const Dense<Scalar>& A, Dense<Scalar>& B );

/*
 *  Conjugate a vector or matrix
 */
template<typename Real>
void Conjugate( Vector<Real>& x );
template<typename Real>
void Conjugate( Vector< std::complex<Real> >& x );

template<typename Real>
void Conjugate
( const Vector<Real>& x,
        Vector<Real>& y );
template<typename Real>
void Conjugate
( const Vector< std::complex<Real> >& x,
        Vector< std::complex<Real> >& y );

template<typename Real>
void Conjugate( std::vector<Real>& x );
template<typename Real>
void Conjugate( std::vector< std::complex<Real> >& x );

template<typename Real>
void Conjugate
( const std::vector<Real>& x,
        std::vector<Real>& y );
template<typename Real>
void Conjugate
( const std::vector< std::complex<Real> >& x,
        std::vector< std::complex<Real> >& y );

template<typename Real>
void Conjugate
( const Vector<Real>& x,
        std::vector<Real>& y );
template<typename Real>
void Conjugate
( const Vector< std::complex<Real> >& x,
        std::vector< std::complex<Real> >& y );

template<typename Real>
void Conjugate
( const std::vector<Real>& x,
        Vector<Real>& y );
template<typename Real>
void Conjugate
( const std::vector< std::complex<Real> >& x,
        Vector< std::complex<Real> >& y );

template<typename Real>
void Conjugate( Dense<Real>& D );
template<typename Real>
void Conjugate( Dense< std::complex<Real> >& D );

template<typename Real>
void Conjugate( const Dense<Real>& D1, Dense<Real>& D2 );
template<typename Real>
void Conjugate
( const Dense< std::complex<Real> >& D1, Dense< std::complex<Real> >& D2 );

/*
 *  Transpose a matrix: B := A^T
 */
template<typename Scalar>
void Transpose( const Dense<Scalar>& A, Dense<Scalar>& B );

/*
 *  Hermitian-transpose a matrix: B := A^H
 */
template<typename Scalar>
void Adjoint( const Dense<Scalar>& A, Dense<Scalar>& B );

/*
 *  For computing the in-place QR decomposition of stacked s x r  and t x r
 *  upper-triangular matrices with their nonzeros packed columnwise.
 *
 * tau should be of length min(r,s+t) and work must be of size t-1.
 */
template<typename Scalar>
void PackedQR
( const int r, const int s, const int t,
  Scalar* RESTRICT packedA, Scalar* RESTRICT tau, Scalar* RESTRICT work );

/*
 * For overwriting B with Q B or Q' B using the Q from a packed QR.
 *
 * tau should be of length min(r,s+t) and work must be of size n.
 */
template<typename Scalar>
void ApplyPackedQFromLeft
( const int r, const int s, const int t,
  const Scalar* RESTRICT packedA, const Scalar* RESTRICT tau,
  Dense<Scalar>& B, Scalar* RESTRICT work );
template<typename Scalar>
void ApplyPackedQAdjointFromLeft
( const int r, const int s, const int t,
  const Scalar* RESTRICT packedA, const Scalar* RESTRICT tau,
  Dense<Scalar>& B, Scalar* RESTRICT work );

/*
 * For overwriting B with B Q or B Q' using the Q from a packed QR.
 *
 * tau should be of length min(r,s+t) and work must be of size n.
 */
template<typename Scalar>
void ApplyPackedQFromRight
( const int r, const int s, const int t,
  const Scalar* RESTRICT packedA, const Scalar* RESTRICT tau,
  Dense<Scalar>& B, Scalar* RESTRICT work );
template<typename Scalar>
void ApplyPackedQAdjointFromRight
( const int r, const int s, const int t,
  const Scalar* RESTRICT packedA, const Scalar* RESTRICT tau,
  Dense<Scalar>& B, Scalar* RESTRICT work );

template<typename Scalar>
void PrintPacked
( const std::string msg,
  const int r, const int s, const int t, const Scalar* packedA,
  std::ostream& os=std::cout );

//--------------------------------------------------------------------------//
// Header implementations                                                   //
//--------------------------------------------------------------------------//

/*
 *  Copy a vector or matrix
 */
template<typename Scalar>
void Copy( const Vector<Scalar>& x, Vector<Scalar>& y )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Copy (Vector,Vector)");
#endif
    y.Resize( x.Height() );
    MemCopy( y.Buffer(), x.LockedBuffer(), x.Height() );
}

template<typename Scalar>
void Copy( const std::vector<Scalar>& x, std::vector<Scalar>& y )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Copy (vector,vector)");
#endif
    y.resize( x.size() );
    MemCopy( &y[0], &x[0], x.size() );
}

template<typename Scalar>
void Copy( const Vector<Scalar>& x, std::vector<Scalar>& y )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Copy (Vector,vector)");
#endif
    y.resize( x.Height() );
    MemCopy( &y[0], x.LockedBuffer(), x.Height() );
}

template<typename Scalar>
void Copy( const std::vector<Scalar>& x, Vector<Scalar>& y )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Copy (vector,Vector)");
#endif
    y.Resize( x.size() );
    MemCopy( y.Buffer(), &x[0], x.size() );
}

template<typename Scalar>
void Copy( const Dense<Scalar>& A, Dense<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Copy (Dense,Dense)");
#endif
    const int m = A.Height();
    const int n = A.Width();
    B.SetType( A.Type() ); B.Resize( m, n );
    if( A.Symmetric() )
    {
        for( int j=0; j<n; ++j )
            MemCopy( B.Buffer(j,j), A.LockedBuffer(j,j), m-j );
    }
    else
    {
        for( int j=0; j<n; ++j )
            MemCopy( B.Buffer(0,j), A.LockedBuffer(0,j), m );
    }
}

/*
 *  Conjugate a vector or matrix
 */

template<typename Real>
void Conjugate( Vector<Real>& x )
{ }

template<typename Real>
void Conjugate( Vector<std::complex<Real> >& x )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Conjugate (Vector)");
#endif
    const int n = x.Height();
    std::complex<Real>* xBuffer = x.Buffer();
    for( int i=0; i<n; ++i )
        xBuffer[i] = Conj( xBuffer[i] );
}

template<typename Real>
void Conjugate( const Vector<Real>& x, Vector<Real>& y )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Conjugate (Vector,Vector)");
#endif
    y.Resize( x.Height() );
    MemCopy( y.Buffer(), x.LockedBuffer(), x.Height() );
}

template<typename Real>
void Conjugate
( const Vector<std::complex<Real> >& x, Vector<std::complex<Real> >& y )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Conjugate (Vector,Vector)");
#endif
    const int n = x.Height();
    y.Resize( n );
    const std::complex<Real>* RESTRICT xBuffer = x.LockedBuffer();
    std::complex<Real>* RESTRICT yBuffer = y.Buffer();
    for( int i=0; i<n; ++i )
        yBuffer[i] = Conj( xBuffer[i] );
}

template<typename Real>
void Conjugate( std::vector<Real>& x )
{ }

template<typename Real>
void Conjugate( std::vector<std::complex<Real> >& x )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Conjugate (vector)");
#endif
    const int n = x.size();
    std::complex<Real>* xBuffer = &x[0];
    for( int i=0; i<n; ++i )
        xBuffer[i] = Conj( xBuffer[i] );
}

template<typename Real>
void Conjugate( const std::vector<Real>& x, std::vector<Real>& y )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Conjugate (vector,vector)");
#endif
    y.resize( x.size() );
    MemCopy( &y[0], &x[0], x.size() );
}

template<typename Real>
void Conjugate
( const std::vector<std::complex<Real> >& x,
        std::vector<std::complex<Real> >& y )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Conjugate (vector,vector)");
#endif
    const int n = x.size();
    y.resize( n );
    const std::complex<Real>* RESTRICT xBuffer = &x[0];
    std::complex<Real>* RESTRICT yBuffer = &y[0];
    for( int i=0; i<n; ++i )
        yBuffer[i] = Conj( xBuffer[i] );
}

template<typename Real>
void Conjugate( const Vector<Real>& x, std::vector<Real>& y )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Conjugate (Vector,vector)");
#endif
    y.resize( x.Height() );
    MemCopy( &y[0], x.Buffer(), x.Height() );
}

template<typename Real>
void Conjugate
( const Vector<std::complex<Real> >& x, std::vector<std::complex<Real> >& y )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Conjugate (Vector,vector)");
#endif
    const int n = x.Height();
    y.resize( n );
    const std::complex<Real>* RESTRICT xBuffer = x.LockedBuffer();
    std::complex<Real>* RESTRICT yBuffer = &y[0];
    for( int i=0; i<n; ++i )
        yBuffer[i] = Conj( xBuffer[i] );
}

template<typename Real>
void Conjugate( const std::vector<Real>& x, Vector<Real>& y )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Conjugate (vector,Vector)");
#endif
    y.Resize( x.size() );
    MemCopy( y.Buffer(), &x[0], x.size() );
}

template<typename Real>
void Conjugate
( const std::vector<std::complex<Real> >& x,
        Vector<std::complex<Real> >& y )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Conjugate (vector,Vector)");
#endif
    const int n = x.size();
    y.Resize( n );
    const std::complex<Real>* xBuffer = &x[0];
    std::complex<Real>* yBuffer = y.Buffer();
    for( int i=0; i<n; ++i )
        yBuffer[i] = Conj( xBuffer[i] );
}

template<typename Real>
void Conjugate( Dense<Real>& D )
{ }

template<typename Real>
void Conjugate( Dense<std::complex<Real> >& D )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Conjugate (Dense)");
#endif
    const int m = D.Height();
    const int n = D.Width();
    for( int j=0; j<n; ++j )
    {
        std::complex<Real>* DCol = D.Buffer(0,j);
        for( int i=0; i<m; ++i )
            DCol[i] = Conj( DCol[i] );
    }
}

template<typename Real>
void Conjugate( const Dense<Real>& D1, Dense<Real>& D2 )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Conjugate (Dense,Dense)");
#endif
    const int m = D1.Height();
    const int n = D1.Width();
    D2.SetType( D1.Type() );
    D2.Resize( m, n );
    if( D1.Symmetric() )
    {
        for( int j=0; j<n; ++j )
            MemCopy( D2.Buffer(j,j), D1.LockedBuffer(j,j), m-j );
    }
    else
    {
        for( int j=0; j<n; ++j )
            MemCopy( D2.Buffer(0,j), D1.LockedBuffer(0,j), m );
    }
}

template<typename Real>
void Conjugate
( const Dense<std::complex<Real> >& D1, Dense<std::complex<Real> >& D2 )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Conjugate (Dense,Dense)");
#endif
    const int m = D1.Height();
    const int n = D1.Width();
    D2.SetType( D1.Type() );
    D2.Resize( m, n );
    if( D1.Symmetric() )
    {
        for( int j=0; j<n; ++j )
        {
            const std::complex<Real>* RESTRICT D1Col = D1.LockedBuffer(0,j);
            std::complex<Real>* RESTRICT D2Col = D2.Buffer(0,j);
            for( int i=j; i<m; ++i )
                D2Col[i] = Conj( D1Col[i] );
        }
    }
    else
    {
        for( int j=0; j<n; ++j )
        {
            const std::complex<Real>* RESTRICT D1Col = D1.LockedBuffer(0,j);
            std::complex<Real>* RESTRICT D2Col = D2.Buffer(0,j);
            for( int i=0; i<m; ++i )
                D2Col[i] = Conj( D1Col[i] );
        }
    }
}

/*
 *  Transpose a matrix, B := A^T
 */

template<typename Scalar>
void Transpose( const Dense<Scalar>& A, Dense<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Transpose (Dense)");
#endif
    if( B.Symmetric() )
        Copy( A, B );
    else
    {
        B.Resize( A.Width(), A.Height() );
        const int m = A.Height();
        const int n = A.Width();
        const int ALDim = A.LDim();
        const int BLDim = B.LDim();
        const Scalar* RESTRICT ABuffer = A.LockedBuffer();
        Scalar* RESTRICT BBuffer = B.Buffer();
        for( int j=0; j<n; ++j )
            for( int i=0; i<m; ++i )
                BBuffer[j+i*BLDim] = ABuffer[i+j*ALDim];
    }
}

/*
 *  Hermitian-transpose a matrix, B := A^H
 */

template<typename Scalar>
void Adjoint( const Dense<Scalar>& A, Dense<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Adjoint (Dense)");
#endif
    if( B.Symmetric() )
        Conjugate( A, B );
    else
    {
        B.Resize( A.Width(), A.Height() );
        const int m = A.Height();
        const int n = A.Width();
        const int ALDim = A.LDim();
        const int BLDim = B.LDim();
        const Scalar* RESTRICT ABuffer = A.LockedBuffer();
        Scalar* RESTRICT BBuffer = B.Buffer();
        for( int j=0; j<n; ++j )
            for( int i=0; i<m; ++i )
                BBuffer[j+i*BLDim] = Conj(ABuffer[i+j*ALDim]);
    }
}

/*
 *  For compute vector two-norms
 */
template<typename Real>
Real TwoNorm( const Vector<Real>& x )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::TwoNorm");
#endif
    return blas::Nrm2( x.Height(), x.LockedBuffer(), 1 );
}

template<typename Real>
Real TwoNorm( const Vector<std::complex<Real> >& x )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::TwoNorm");
#endif
    return blas::Nrm2( x.Height(), x.LockedBuffer(), 1 );
}

/*
 *  For scaling vectors and matrices
 */

template<typename Scalar>
void Scale( Scalar alpha, Vector<Scalar>& x )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Scale (Vector)");
#endif
    if( alpha == Scalar(0) )
        MemZero( x.Buffer(), x.Height() );
    else
        blas::Scal( x.Height(), alpha, x.Buffer(), 1 );
}

template<typename Scalar>
void Scale( Scalar alpha, Dense<Scalar>& D )
{
#ifndef RELEASE
    CallStackEntry entry("mat_tools::Scale (Dense)");
#endif
    const int m = D.Height();
    const int n = D.Width();

    if( alpha == Scalar(1) )
        return;

    if( D.Symmetric() )
    {
        if( alpha == Scalar(0) )
            for( int j=0; j<n; ++j )
                MemZero( D.Buffer(j,j), m-j );
        else
            for( int j=0; j<n; ++j )
                blas::Scal( m-j, alpha, D.Buffer(j,j), 1 );
    }
    else
    {
        if( alpha == Scalar(0) )
            for( int j=0; j<n; ++j )
                MemZero( D.Buffer(0,j), m );
        else
            for( int j=0; j<n; ++j )
                blas::Scal( m, alpha, D.Buffer(0,j), 1 );
    }
}

} // namespace mat_tools
} // namespace DHIF

#endif // ifndef DHIF_MAT_TOOLS_HPP
