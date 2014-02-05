#include "DHIF.hpp"

namespace {

int numDmhmInits=0;
bool DHIFInitializedMpi = false;

DHIF::MpiArgs* args = 0;

#ifndef RELEASE
std::stack<std::string> callStack;
#endif

#ifdef MEMORY_INFO
double memoryUsage=0;
double peakMemoryUsage=0;
std::map<int,double> memoryUsageMap;
std::map<int,double> peakMemoryUsageMap;
#endif

}

namespace DHIF {

bool Initialized()
{ return ::numDmhmInits > 0; }

void Initialize( int& argc, char**& argv )
{
    if( ::numDmhmInits > 0 )
    {
        ++::numDmhmInits;
        return;
    }

    ::args = new MpiArgs( argc, argv );

    ::numDmhmInits = 1;
    if( !mpi::Initialized() )
    {
        if( mpi::Finalized() )
        {
            throw std::logic_error
            ("Cannot initialize DHIF after finalizing MPI");
        }
        mpi::Initialize( argc, argv );
        ::DHIFInitializedMpi = true;
    }

    // Seed the random number generators
    const int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const int commSize = mpi::CommSize( mpi::COMM_WORLD );
    UInt64 seed;
    seed.d[0] = 17U;
    seed.d[1] = 21U;
    SeedSerialLcg( seed );
    SeedParallelLcg( commRank, commSize, seed );
}

void Finalize()
{
#ifndef RELEASE
    CallStackEntry entry("Finalize");
#endif
    if( ::numDmhmInits <= 0 )
        throw std::logic_error("Finalized DHIF more than initialized");
    --::numDmhmInits;

    if( mpi::Finalized() )
        std::cerr << "Warning: MPI was finalized before DHIF" << std::endl;
    if( ::numDmhmInits == 0 )
    {
        delete ::args;
        ::args = 0;

        if( ::DHIFInitializedMpi )
            mpi::Finalize();
    }

}
MpiArgs& GetArgs()
{
    if( args == 0 )
        throw std::runtime_error("No available instance of MpiArgs");
    return *::args;
}

#ifndef RELEASE
void PushCallStack( const std::string s )
{ ::callStack.push(s); }

void PopCallStack()
{ ::callStack.pop(); }

void DumpCallStack()
{
    std::cout << "Dumping call stack of size " << ::callStack.size()
              << std::endl;
    std::ostringstream msg;
    while( ! ::callStack.empty() )
    {
        msg << "[" << ::callStack.size() << "]: " << ::callStack.top() << "\n";
        ::callStack.pop();
    }
    std::cerr << msg.str() << std::endl;
}
#endif

#ifdef MEMORY_INFO
void ResetMemoryCount( int key )
{
    if( key < 0 )
    {
        ::memoryUsage = 0;
        ::peakMemoryUsage = 0;
    }
    else
    {
        std::map<int,double>::iterator it;
        it = ::peakMemoryUsageMap.find( key );
        if( it != ::peakMemoryUsageMap.end() )
        {
            ::memoryUsageMap[key] = 0;
            ::peakMemoryUsageMap[key] = 0;
        }
    }
}

void NewMemoryCount( int key )
{
    ::memoryUsageMap[key] = 0;
    ::peakMemoryUsageMap[key] = 0;
}

void EraseMemoryCount( int key )
{
    ::memoryUsageMap.erase( key );
    ::peakMemoryUsageMap.erase( key );
}

void AddToMemoryCount( double size )
{
    ::memoryUsage += size;
    if( ::memoryUsage > ::peakMemoryUsage )
        ::peakMemoryUsage = ::memoryUsage;
    std::map<int,double>::iterator it;
    for( it=::memoryUsageMap.begin(); it!=::memoryUsageMap.end(); ++it )
    {
        it->second += size;
        if( it->second > ::peakMemoryUsageMap[it->first] )
            ::peakMemoryUsageMap[it->first] = it->second;
    }
}

double MemoryUsage( int key )
{
    if( key < 0 )
        return ::memoryUsage;
    else
    {
        std::map<int,double>::iterator it;
        it = ::memoryUsageMap.find( key );
        if( it == ::memoryUsageMap.end() )
            return 0;
        else
            return ::memoryUsageMap[key];
    }
}

double PeakMemoryUsage( int key )
{
    if( key < 0 )
        return ::peakMemoryUsage;
    else
    {
        std::map<int,double>::iterator it;
        it = ::peakMemoryUsageMap.find( key );
        if( it == ::peakMemoryUsageMap.end() )
            return 0;
        else
            return ::peakMemoryUsageMap[key];
    }
}
#endif

void PrintGlobal( double num, const std::string tag, std::ostream& os )
{
    mpi::Comm team = mpi::COMM_WORLD;
    const int teamRank = mpi::CommRank( team );
    double maxnum, minnum;
    mpi::Reduce( &num, &maxnum, 1, mpi::MAX, 0, team );
    mpi::Reduce( &num, &minnum, 1, mpi::MIN, 0, team );
    if( teamRank == 0 )
        os << tag << maxnum << "(max), " << minnum << "(min).\n";
}

} // namespace DHIF
