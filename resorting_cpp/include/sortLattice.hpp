#pragma once

#include "sortParallel.hpp"

// 0 means never, 1 always, 2 every second, ...
#define CHECK_FOR_DIRECT_REMOVAL_EVERY_X_MOVES 20

#define VALUE_FILLED_DESIRED 1.
#define VALUE_USED_UNDESIRED 0.5
#define VALUE_CLEARED_UNDESIRED 0.2
#define VALUE_CLEARED_UNDESIRED_UNUSABLE 0.8
#define VALUE_CLEARED_OUTSIDE_UNUSABLE 0.1

double inline pythagorasDist(double d1, double d2)
{
    return sqrt(d1 * d1 + d2 * d2);
}

enum TargetState
{
    EMPTY, OCCUPIED, IRRELEVANT
};

struct ArrayBounds
{
    unsigned int firstRowOrXC, lastRowOrXCExcl, firstColOrAC, lastColOrACExcl;
};

struct MinimalArrayInformation
{
    std::vector<int> bufferRows, bufferCols, dumpingIndicesAC;
    bool vertical;
    ArrayBounds normalIndices;
};

struct ArrayInformation
{
    std::vector<std::vector<int>> usableAtomsPerXCIndex, unusableAtomsPerXCIndex, targetSitesPerXCIndex, parkingSitesPerXCIndex;
    std::vector<int> bufferRows, bufferCols, dumpingIndicesAC;
    bool vertical;

    // Across channel dir will be abbreviated as XC, along channel as AC
    unsigned int arraySizeXC, arraySizeAC, maxTonesXC, maxTonesAC, dumpingIndicesLow, dumpingIndicesHigh;
    double spacingXC, spacingAC;
    int targetGapXC, targetGapAC, sortingChannelWidth;

    ArrayBounds normalIndices, normalIndicesXCAC, relevantIndicesXCAC;
};

std::optional<std::vector<ParallelMove>> sortLatticeGreedyParallel(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& targetGeometry);

std::optional<std::pair<std::vector<ParallelMove>,MinimalArrayInformation>> sortLatticeByRowParallel(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    const py::array_t<TargetState>& targetGeometry);
std::optional<std::vector<ParallelMove>> fixLatticeByRowSortingDeficiencies(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    const py::array_t<TargetState>& targetGeometry, MinimalArrayInformation& arrayInfo);

Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> generateMask(double distance, double spacingFraction = 1);
Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic> generatePathway(size_t borderRows, size_t borderCols, 
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &occupancy,
    double distFromOcc = Config::getInstance().recommendedDistFromOccSites, 
    double distFromEmpty = Config::getInstance().recommendedDistFromEmptySites,
    std::optional<size_t> minRow = std::nullopt, std::optional<size_t> maxRow = std::nullopt, 
    std::optional<size_t> minCol = std::nullopt, std::optional<size_t> maxCol = std::nullopt);
std::optional<ArrayInformation> conductInitialAnalysis(ArrayAccessor& stateArray, 
    pybind11::detail::unchecked_reference<TargetState, 2>& targetGeometry, std::shared_ptr<spdlog::logger> logger);