#pragma once
#ifndef DHIF_DISTHMAT3D_HPP
#define DHIF_DISTHMAT3D_HPP 1

#include "DHIF.hpp"

namespace DHIF {

enum PointType
{
    INTERIOR=0;
    UP;
    DOWN;
    LEFT;
    RIGHT;
    FRONT;
    BACK;
    EDGE;
    CORNER;
}

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
        std::vector<mpi::Comm> teams_, crossTeams_;
    public:
        Teams( mpi::Comm comm );
        ~Teams();

        int NumLevels() const;
        int TeamLevel( int level ) const;
        mpi::Comm Team( int level ) const;
        mpi::Comm CrossTeam( int inverseLevel ) const;

        void TreeSums
        ( std::vector<Scalar>& buffer, const std::vector<int>& sizes ) const;

        void TreeSumToRoots
        ( std::vector<Scalar>& buffer, const std::vector<int>& sizes ) const;

        void TreeBroadcasts
        ( std::vector<Scalar>& buffer, const std::vector<int>& sizes ) const;

        void TreeBroadcasts
        ( std::vector<int>& buffer, const std::vector<int>& sizes ) const;
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
    int Tns2PT(Index3 tns) const;
    int CrossTns2Gen(Index3 tns, Index3 tnsori) const;
    bool IsLeader() const;

private:

    struct Node
    {
        std::vector<DistHIFDE3D*> children;
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
    // Local size of the domain
    int xSize_, ySize_, zSize_;
    // Global size of the whole domain
    int xSizeGlobal_, ySizeGlobal_, zSizeGlobal_;
    int DOF_;
    bool own_;

    const Teams* teams_;
    Node node_;

    Grid* grid_;

    // Interior, Up, Left, Front interact with
    // Interior, Up, Down, Left, Right, Front, Back, Edge
    // This is a 8 * 6 block matrix, but the third and fifth
    // columns are empty, just for coding simplicity
    std::vector<std::vector<DistMatrix<Scalar,STAR,VC> > >
        A_(8,std::vector<DistMatrix<Scalar,STAR,VC> >(6));

    // Store the interaction between points themselves which
    // will not be touched by children
    std::vector<Matrix<Scalar> >
        Asub_(6);

    // 6 blocks of row vector, the global position list of
    // the corresponding column index in matrix A_
    std::vector<DistMatrix<Index3d,STAR,VC> > globalPosList_(6);

    // 6 blocks of row vector, the global position list of
    // the corresponding column index in matrix Asub_
    std::vector<Matrix<Index3d> > globalPosListsub_(6);

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
{
    switch Tns2PT(tns):
    {
        case INTERIOR:
            return (tns[0]-xOffset_)*zSize_*ySize_+
                (tns[1]-yOffset_)*zSize_+tns[2]-zOffset_;
        case UP:
            return (tns[0]-xOffset_)*ySize_ + (tns[1]-yOffset_);
        case DOWN:
            return (tns[0]-xOffset_)*ySize_ + (tns[1]-yOffset_);
        case LEFT:
            return (tns[0]-xOffset_)*zSize_ + (tns[2]-zOffset_);
        case RIGHT:
            return (tns[0]-xOffset_)*zSize_ + (tns[2]-zOffset_);
        case FRONT:
            return (tns[1]-yOffset_)*zSize_ + (tns[2]-zOffset_);
        case BACK:
            return (tns[1]-yOffset_)*zSize_ + (tns[2]-zOffset_);
        case EDGE:
            int idx = ( tns[0] == xOffset_-1 ? 0 :
                        ( tns[0]%xSizeGlobal_ == (xOffset_+xSize_)%xSizeGlobal_
                          ? 1 : -1 ));
            int idy = ( tns[1] == yOffset_-1 ? 0 :
                        ( tns[1]%ySizeGlobal_ == (yOffset_+ySize_)%ySizeGlobal_
                          ? 1 : -1 ));
            int idz = ( tns[2] == zOffset_-1 ? 0 :
                        ( tns[2]%zSizeGlobal_ == (zOffset_+zSize_)%zSizeGlobal_
                          ? 1 : -1 ));
            if( idz == -1 )
                return (idx+idy)*zSize_+tns[2];
            if( idy == -1 )
                return (idx+idz)*ySize_+tns[1] + zSize_*4;
            if( idx == -1 )
                return (idz+idy)*xSize_+tns[0] + (zSize_+ySize_)*4;
        case CORNER:
        default:
            DEBUG_ONLY( LogicError("CORNER points are not supported") );
    }
}

template<typename Scalar>
inline PointType
DistHIFDE3D<Scalar>::Tns2PT(Index3 tns) const
{
    int idx = ( tns[0] == xOffset_-1 ? -1 :
                ( tns[0]%xSizeGlobal_ == (xOffset_+xSize_)%xSizeGlobal_ ? 1 :
                  0 ));
    int idy = ( tns[1] == yOffset_-1 ? -1 :
                ( tns[1]%ySizeGlobal_ == (yOffset_+ySize_)%ySizeGlobal_ ? 1 :
                  0 ));
    int idz = ( tns[2] == zOffset_-1 ? -1 :
                ( tns[2]%zSizeGlobal_ == (zOffset_+zSize_)%zSizeGlobal_ ? 1 :
                  0 ));
    if( idx == 0 && idy == 0 && idz == 0 )
        return INTERIOR;
    if( idx == 0 && idy == 0 && idz == -1 )
        return UP;
    if( idx == 0 && idy == 0 && idz == 1 )
        return DOWN;
    if( idx == 0 && idy == -1 && idz == 0 )
        return LEFT;
    if( idx == 0 && idy == 1 && idz == 0 )
        return RIGHT;
    if( idx == -1 && idy == 0 && idz == 0 )
        return FRONT;
    if( idx == 1 && idy == 0 && idz == 0 )
        return BACK;
    if( (idx + idy + idz)%2 == 0 )
        return EDGE;
    return CORNER;

}

template<typename Scalar>
inline int
DistHIFDE3D<Scalar>::CrossTns2Gen(Index3 tns, Index3 tnsori) const
{
    PointType tnsoriPT = Tns2PT( tnsori );
    PointType tnsPT = Tns2PT( tns );
    const int xC = xSize_/2 + xOffset_-1;
    const int yC = ySize_/2 + yOffset_-1;
    const int zC = zSize_/2 + zOffset_-1;
    switch( tnsoriPT )
    {
        case INTERIOR:
            switch( tnsPT )
            {
                case INTERIOR:
                    if( tns[0] == xC && tns[1] == yC && tns[2] == zC )
                        return 0;
                    if( tns[0] < xC && tns[1] == yC && tns[2] == zC )
                        return tns[0]-xOffset_+1;
                    if( tns[0] > xC && tns[1] == yC && tns[2] == zC )
                        return tns[0]-xOffset_;
                    if( tns[0] == xC && tns[1] < yC && tns[2] == zC )
                        return xSize_+tns[1]-yOffset_;
                    if( tns[0] == xC && tns[1] > yC && tns[2] == zC )
                        return xSize_+tns[1]-yOffset_-1;
                    if( tns[0] == xC && tns[1] == yC && tns[2] < zC )
                        return xSize_+ySize_-1+tns[2]-zOffset_;
                    if( tns[0] == xC && tns[1] == yC && tns[2] > zC )
                        return xSize_+ySize_-1+tns[2]-zOffset_-1;
                case UP:
                    return xSize_+ySize_+zSize_-2;
                case DOWN:
                    return xSize_+ySize_+zSize_-1;
                case LEFT:
                    return xSize_+ySize_+zSize_;
                case RIGHT:
                    return xSize_+ySize_+zSize_+1;
                case FRONT:
                    return xSize_+ySize_+zSize_+2;
                case BACK:
                    return xSize_+ySize_+zSize_+3;
                default:
                    return -1;
            }
        case UP:
            switch( tnsPT )
            {
                case UP:
                    if( tns[0] == xC && tns[1] == yC )
                        return 0;
                    if( tns[0] < xC && tns[1] == yC )
                        return tns[0]-xOffset_+1;
                    if( tns[0] > xC && tns[1] == yC )
                        return tns[0]-xOffset_;
                    if( tns[0] == xC && tns[1] < yC )
                        return xSize_+tns[1]-yOffset_;
                    if( tns[0] == xC && tns[1] > yC )
                        return xSize_+tns[1]-yOffset_-1;
                case EDGE:
                    if( tnsori[1] - tns[1] == 1 )
                        return xSize_+ySize_-1;
                    if( (tnsori[1]-tns[1]+ySizeGlobal_)%ySizeGlobal_
                                == ySizeGlobal_-1 )
                        return xSize_+ySize_;
                    if( tnsori[0] - tns[0] == 1 )
                        return xSize_+ySize_+1;
                    if( (tnsori[0]-tns[0]+xSizeGlobal_)%xSizeGlobal_
                                == xSizeGlobal_-1 )
                        return xSize_+ySize_+2;
                default:
                    return -1;
            }
        case LEFT:
            switch( tnsPT )
            {
                case LEFT:
                    if( tns[0] == xC && tns[2] == zC )
                        return 0;
                    if( tns[0] < xC && tns[2] == zC )
                        return tns[0]-xOffset_+1;
                    if( tns[0] > xC && tns[2] == zC )
                        return tns[0]-xOffset_;
                    if( tns[0] == xC && tns[2] < zC )
                        return xSize_+tns[2]-zOffset_;
                    if( tns[0] == xC && tns[2] > zC )
                        return xSize_+tns[2]-zOffset_-1;
                case EDGE:
                    if( tnsori[2] - tns[2] == 1 )
                        return xSize_+zSize_-1;
                    if( (tnsori[2]-tns[2]+zSizeGlobal_)%zSizeGlobal_
                                == zSizeGlobal_-1 )
                        return xSize_+zSize_;
                    if( tnsori[0] - tns[0] == 1 )
                        return xSize_+zSize_+1;
                    if( (tnsori[0]-tns[0]+xSizeGlobal_)%xSizeGlobal_
                                == xSizeGlobal_-1 )
                        return xSize_+zSize_+2;
                default:
                    return -1;
            }
        case FRONT:
            switch( tnsPT )
            {
                case FRONT:
                    if( tns[1] == yC && tns[2] == zC )
                        return 0;
                    if( tns[1] < yC && tns[2] == zC )
                        return tns[1]-yOffset_+1;
                    if( tns[1] > yC && tns[2] == zC )
                        return tns[1]-yOffset_;
                    if( tns[1] == yC && tns[2] < zC )
                        return ySize_+tns[2]-zOffset_;
                    if( tns[1] == yC && tns[2] > zC )
                        return ySize_+tns[2]-zOffset_-1;
                case EDGE:
                    if( tnsori[2] - tns[2] == 1 )
                        return ySize_+zSize_-1;
                    if( (tnsori[2]-tns[2]+zSizeGlobal_)%zSizeGlobal_
                                == zSizeGlobal_-1 )
                        return ySize_+zSize_;
                    if( tnsori[1] - tns[1] == 1 )
                        return ySize_+zSize_+1;
                    if( (tnsori[1]-tns[1]+ySizeGlobal_)%ySizeGlobal_
                                == ySizeGlobal_-1 )
                        return ySize_+zSize_+2;
                default:
                    return -1;
            }
        default:
            return -1;
    }
}

template<typename Scalar>
inline int
DistHIFDE3D<Scalar>::IsLeader() const
{
    mpi::Comm team = teams_->Team(level_);
    return mpi::CommRank( team ) == 0;
}

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

    teams_.resize( numLevels );
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

    crossTeams_.resize( numLevels );
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
    for( int i=0; i<teams_.size(); ++i )
        mpi::CommFree( teams_[i] );
    for( int i=0; i<crossTeams_.size(); ++i )
        mpi::CommFree( crossTeams_[i] );
}

template<typename Scalar>
inline int
DistHIFDE3D<Scalar>::Teams::NumLevels() const
{ return teams_.size(); }

template<typename Scalar>
inline int
DistHIFDE3D<Scalar>::Teams::TeamLevel( int level ) const
{ return std::min(level,int(teams_.size()-1)); }

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
    if( inverseLevel >= crossTeams_.size() )
        throw std::logic_error("Invalid cross team request");
#endif
    return crossTeams_[inverseLevel];
}

template<typename Scalar>
inline void
DistHIFDE3D<Scalar>::Teams::TreeSums
( std::vector<Scalar>& buffer, const std::vector<int>& sizes ) const
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
( std::vector<Scalar>& buffer, const std::vector<int>& sizes ) const
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
( std::vector<Scalar>& buffer, const std::vector<int>& sizes ) const
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
( std::vector<int>& buffer, const std::vector<int>& sizes ) const
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
