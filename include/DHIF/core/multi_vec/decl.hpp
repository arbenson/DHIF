#pragma once
#ifndef DHIF_CORE_MULTIVEC_DECL_HPP
#define DHIF_CORE_MULTIVEC_DECL_HPP

namespace DHIF {

template<typename T>
class MultiVec
{
public:
    // Constructors and destructors
    MultiVec();
    MultiVec( int height, int width );
    // TODO: Constructor for building from a MultiVec
    ~MultiVec();

    // High-level information
    int Height() const;
    int Width() const;

    // Data
    T Get( int row, int col = 0 ) const;
    void Set( int row, int col = 0, T value );
    void Update( int row, int col = 0, T value );

    // For modifying the size of the multi-vector
    void Empty();
    void Resize( int height, int width = 1 );

    // Assignment
    const MultiVec<T>& operator=( const MultiVec<T>& X );

private:
    Matrix<T> multiVec_;
};

// Set all of the entries of X to zero
template<typename T>
void MakeZeros( MultiVec<T>& X );

// Draw the entries of X uniformly from the unitball in T
template<typename T>
void MakeUniform( MultiVec<T>& X );

// Just column-wise l2 norms for now
template<typename F>
void Norms( const MultiVec<F>& X, std::vector<BASE(F)>& norms );

// Just column-wise l2 norms for now
template<typename F>
BASE(F) Norm( const MultiVec<F>& x );

// Y := alpha X + Y
template<typename T>
void Axpy( T alpha, const MultiVec<T>& X, MultiVec<T>& Y );

} // namespace DHIF

#endif // ifndef DHIF_CORE_MULTIVEC_DECL_HPP
