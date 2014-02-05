#pragma once
#ifndef DHIF_DISTHMAT3D_HPP
#define DHIF_DISTHMAT3D_HPP 1

#include "DHIF.hpp"

namespace DHIF {

extern Timer timerGlobal;
// A distributed H-matrix class that assumes a 3d box domain and requires
// a power of two number of processes. It does not yet support implicit
// symmetry.
template<typename Scalar>
class DistHIFDE3D
{
public:
    typedef BASE(Scalar) Real;

    /*
     * Public data structures
     */
    class Teams
    {
    private:
        Vector<mpi::Comm> teams_, crossTeams_;
    public:
        Teams( mpi::Comm comm );
        ~Teams();

        int NumLevels() const;
        int TeamLevel( int level ) const;
        mpi::Comm Team( int level ) const;
        mpi::Comm CrossTeam( int inverseLevel ) const;

        void TreeSums
        ( Vector<Scalar>& buffer, const Vector<int>& sizes ) const;

        void TreeSumToRoots
        ( Vector<Scalar>& buffer, const Vector<int>& sizes ) const;

        void TreeBroadcasts
        ( Vector<Scalar>& buffer, const Vector<int>& sizes ) const;

        void TreeBroadcasts
        ( Vector<int>& buffer, const Vector<int>& sizes ) const;
    };

    /*
     * Public non-static member functions
     */

    DistHIFDE3D( int xSize, int ySize, int zSize );
    DistHIFDE3D
    ( int numLevels, int level, int xOffset, int yOffset, int zOffset,
      int xSize, int ySize, int zSize, bool own );
    ~DistHIFDE3D();

    void Clear();

    int Height() const;
    int Width() const;
    int MaxRank() const;
    int NumLevels() const;
    int Tns2Gen(Index3 tns) const;

private:

    struct Node
    {
        Vector<DistHIFDE3D*> children;
        Node();
        ~Node();
        DistHIFDE3D& Child( int t );
        const DistHIFDE3D& Child( int t ) const;
    };
    Node* NewNode() const;

    /*
     * Private non-static member functions
     */

    void BuildTree( const NumTns<Scalar>& A, const NumTns<Scalar>& V );

    /*
     * Private data
     */
    int numLevels_;
    int level_;
    bool leaflevel_;
    // All Offset_ start from 1, 0 is for the boundary
    int xOffset_, yOffset_, zOffset_;
    int xSize_, ySize_, zSize_;
    int DOF_;
    bool own_;

    Vector<Vec3T<int> > degreeList_;

    const Teams* teams_;
    Node node_;

    Dense<Scalar> D_, A_self_;

    Vector<DistHIFDE3D> children_;
};

//----------------------------------------------------------------------------//
// Inlined implementations                                                    //
//----------------------------------------------------------------------------//

/*
 * Public member functions
 */

template<typename Scalar>
inline int
DistHIFDE3D<Scalar>::NumLevels() const
{ return numLevels_; }

template<typename Scalar>
inline int
DistHIFDE3D<Scalar>::Tns2Gen(Index3 tns) const
{ return tns[0]*zSize_*ySize_+tns[1]*zSize_+tns[2]; }

/*
 * Public structures member functions
 */

template<typename Scalar>
inline
DistHIFDE3D<Scalar>::Teams::Teams( mpi::Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("DistHIFDE3D::Teams::Teams");
#endif
    const int rank = mpi::CommRank( comm );
    const int p = mpi::CommSize( comm );
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");

    // Simple (yet slow) method for computing the number of teams
    // (and how many we're the root of)
    int numLevels=1;
    unsigned teamSize=p;
    while( teamSize != 1 )
    {
        ++numLevels;
        if( teamSize >= 8 )
            teamSize >>= 3;
        else if( teamSize != 1 )
            throw std::logic_error
                ("Must use a power of eight number of processes");
    }

    teams_.Resize( numLevels );
    mpi::CommDup( comm, teams_[0] );
    teamSize = p;
    for( int level=1; level<numLevels; ++level )
    {
        if( teamSize >= 8 )
            teamSize >>= 3;
        else
            teamSize = 1;
        const int color = rank/teamSize;
        const int key = rank - color*teamSize;
        mpi::CommSplit( comm, color, key, teams_[level] );
    }

    crossTeams_.Resize( numLevels );
    mpi::CommDup( teams_[numLevels-1], crossTeams_[0] );
    for( int inverseLevel=1; inverseLevel<numLevels; ++inverseLevel )
    {
        const int level = numLevels-1-inverseLevel;
        teamSize = mpi::CommSize( teams_[level] );
        const int teamSizePrev = mpi::CommSize( teams_[level+1] );

        int color, key;
        const int mod = rank % teamSizePrev;
        color = (rank/teamSize)*teamSizePrev + mod;
        key = (rank/teamSizePrev) % 8;

        mpi::CommSplit( comm, color, key, crossTeams_[inverseLevel] );
    }
}

template<typename Scalar>
inline
DistHIFDE3D<Scalar>::Teams::~Teams()
{
#ifndef RELEASE
    CallStackEntry entry("DistHIFDE3D::Teams::~Teams");
#endif
    for( int i=0; i<teams_.Size(); ++i )
        mpi::CommFree( teams_[i] );
    for( int i=0; i<crossTeams_.Size(); ++i )
        mpi::CommFree( crossTeams_[i] );
}

template<typename Scalar>
inline int
DistHIFDE3D<Scalar>::Teams::NumLevels() const
{ return teams_.Size(); }

template<typename Scalar>
inline int
DistHIFDE3D<Scalar>::Teams::TeamLevel( int level ) const
{ return std::min(level,teams_.Size()-1); }

template<typename Scalar>
inline mpi::Comm
DistHIFDE3D<Scalar>::Teams::Team( int level ) const
{ return teams_[TeamLevel(level)]; }

template<typename Scalar>
inline mpi::Comm
DistHIFDE3D<Scalar>::Teams::CrossTeam( int inverseLevel ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHIFDE3D::Teams::CrossTeam");
    if( inverseLevel >= crossTeams_.Size() )
        throw std::logic_error("Invalid cross team request");
#endif
    return crossTeams_[inverseLevel];
}

template<typename Scalar>
inline void
DistHIFDE3D<Scalar>::Teams::TreeSums
( Vector<Scalar>& buffer, const Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHIFDE3D::Teams::TreeSums");
#endif
    const int numLevels = NumLevels();
    const int numAllReduces = numLevels-1;

    int totalSize = 0;
    for( int i=0; i<numAllReduces; ++i )
        totalSize += sizes[i];

    if( totalSize == 0 )
        return;

    // Use O(log(p)) custom method:
    // - AllReduce over each cross communicator
    int partialSize = totalSize;
    for( int i=0; i<numAllReduces; ++i )
    {
        if( partialSize == 0 )
            break;
        mpi::Comm crossTeam = CrossTeam( i+1 );
        mpi::AllReduce
        ( (const Scalar*)MPI_IN_PLACE, &buffer[0], partialSize, mpi::SUM,
          crossTeam );
        partialSize -= sizes[numAllReduces-1-i];
    }
}

template<typename Scalar>
inline void
DistHIFDE3D<Scalar>::Teams::TreeSumToRoots
( Vector<Scalar>& buffer, const Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHIFDE3D::Teams::TreeSumToRoots");
#endif
    const int numLevels = NumLevels();
    const int numReduces = numLevels-1;

    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];

    if( totalSize == 0 )
        return;

    // Use O(log(p)) custom method:
    // - Reduce to the root of each cross communicator
    int partialSize = totalSize;
    for( int i=0; i<numReduces; ++i )
    {
        if( partialSize == 0 )
            break;
        mpi::Comm crossTeam = CrossTeam( i+1 );
        const int crossTeamRank = mpi::CommRank( crossTeam );
        if( crossTeamRank == 0 )
            mpi::Reduce
            ( (const Scalar*)MPI_IN_PLACE, &buffer[0],
              partialSize, mpi::SUM, 0, crossTeam );
        else
            mpi::Reduce
            ( &buffer[0], (Scalar*)0, partialSize, mpi::SUM, 0, crossTeam );
        partialSize -= sizes[numReduces-1-i];
    }
}

template<typename Scalar>
inline void
DistHIFDE3D<Scalar>::Teams::TreeBroadcasts
( Vector<Scalar>& buffer, const Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHIFDE3D::Teams::TreeBroadcasts");
#endif
    const int numLevels = NumLevels();
    const int numBroadcasts = numLevels-1;

    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];

    if( totalSize == 0 )
        return;

    // Use O(log(p)) custom method:
    // - Broadcast over each cross communicator
    int partialSize = totalSize;
    for( int i=0; i<numBroadcasts; ++i )
    {
        if( partialSize == 0 )
            break;
        mpi::Comm crossTeam = CrossTeam( i+1 );
        mpi::Broadcast( &buffer[0], partialSize, 0, crossTeam );
        partialSize -= sizes[numBroadcasts-1-i];
    }
}

template<typename Scalar>
inline void
DistHIFDE3D<Scalar>::Teams::TreeBroadcasts
( Vector<int>& buffer, const Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHIFDE3D::Teams::TreeBroadcasts");
#endif
    const int numLevels = NumLevels();
    const int numBroadcasts = numLevels-1;

    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];

    if( totalSize == 0 )
        return;

    // Use O(log(p)) custom method:
    // - Broadcast over each cross communicator
    int partialSize = totalSize;
    for( int i=0; i<numBroadcasts; ++i )
    {
        if( partialSize == 0 )
            break;
        mpi::Comm crossTeam = CrossTeam( i+1 );
        mpi::Broadcast( &buffer[0], partialSize, 0, crossTeam );
        partialSize -= sizes[numBroadcasts-1-i];
    }
}

} // namespace DHIF

#endif // ifndef DHIF_DISTHMAT3D_HPP
