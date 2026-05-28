#include "sortLattice.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <set>
#include <chrono>
#include <ranges>

#include "config.hpp"

struct compareOnlyTones
{
    bool operator()(const std::tuple<std::vector<size_t>,std::vector<size_t>,std::vector<std::tuple<bool,size_t,int>>>& lhs, 
        const std::tuple<std::vector<size_t>,std::vector<size_t>,std::vector<std::tuple<bool,size_t,int>>>& rhs) const
    {
        // https://en.cppreference.com/w/cpp/utility/tuple/operator_cmp
        if (std::get<0>(lhs) < std::get<0>(rhs)) return true;
        if (std::get<0>(rhs) < std::get<0>(lhs)) return false;
        return std::get<1>(lhs) < std::get<1>(rhs);
    }
};

double moveCost(const std::vector<std::tuple<bool,size_t,int>>& path)
{
    double cost = Config::getInstance().moveCostOffset;
    bool verticalSegment = false;
    std::map<size_t,double> segmentDist;
    bool first = true;
    for(const auto& [vertical, index, dist] : path)
    {
        if(first)
        {
            first = false;
            verticalSegment = vertical;
        }
        else if(vertical != verticalSegment)
        {
            for(auto& v : segmentDist)
            {
                v.second = abs(v.second);
            }
            cost += costPerSubMove(std::max_element(segmentDist.begin(), segmentDist.end(), 
                [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; })->second);
            verticalSegment = vertical;
            segmentDist.clear();
        }
        segmentDist[index] += (double)dist / 2 * (vertical ? Config::getInstance().rowSpacing : Config::getInstance().columnSpacing); // / 2 Because the path is expressed in units of half steps
    }
    for(auto& v : segmentDist)
    {
        v.second = abs(v.second);
    }
    cost += costPerSubMove(std::max_element(segmentDist.begin(), segmentDist.end(), 
        [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; })->second);
    return cost;
}

std::tuple<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>, unsigned int> labelPathway(
    const Eigen::Ref<const Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>>& pathway)
{
    Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic> labelledPathway = 
        Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>::Zero(pathway.rows(), pathway.cols());
    
    unsigned int pathwayIndex = 1;
    std::vector<std::tuple<Eigen::Index,Eigen::Index>> unvisitedLocs;
    for(Eigen::Index r = 0; r < pathway.rows(); r++)
    {
        for(Eigen::Index c = 0; c < pathway.cols(); c++)
        {
            if(pathway(r, c) == 0 && labelledPathway(r, c) == 0)
            {
                unvisitedLocs.push_back(std::tuple(r,c));
                while(!unvisitedLocs.empty())
                {
                    const auto [rP, cP] = unvisitedLocs.back();
                    unvisitedLocs.pop_back();
                    labelledPathway(rP, cP) = pathwayIndex;

                    for(int dir = 0; dir < 4; dir++)
                    {
                        int shiftedRow = dir % 2 == 0 ? rP + dir - 1 : rP;
                        int shiftedCol = dir % 2 == 1 ? cP + dir - 2 : cP;
                        if(shiftedRow >= 0 && shiftedRow < pathway.rows() && shiftedCol >= 0 && 
                            shiftedCol < pathway.cols() && pathway(shiftedRow, shiftedCol) == 0 && 
                            labelledPathway(shiftedRow, shiftedCol) == 0)
                        {
                            unvisitedLocs.push_back(std::tuple(shiftedRow,shiftedCol));
                        }
                    }
                }
                pathwayIndex++;
            }
        }
    }
    return std::tuple(labelledPathway, pathwayIndex - 1);
}

std::vector<std::set<std::tuple<size_t,size_t>>> findTargetSitesPerPathway(
    const Eigen::Ref<const Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>>& penalizedPathway, 
    const Eigen::Ref<const Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>>& labelledPathway, 
    std::set<std::tuple<size_t,size_t>>& usableTargetSites,
    size_t borderRows, size_t borderCols, unsigned int labelCount,  size_t compZoneRowStart, size_t compZoneRowEnd,
    size_t compZoneColStart, size_t compZoneColEnd,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry,
    std::shared_ptr<spdlog::logger> logger)
{
    auto [labelledPenalizedPathway, pLabelCount] = labelPathway(penalizedPathway);
    std::vector<std::set<std::tuple<size_t,size_t>>> targetSitesPerPPathway;
    targetSitesPerPPathway.resize(pLabelCount);

    std::vector<std::set<unsigned int>> pathwaysPerPPathway;
    pathwaysPerPPathway.resize(pLabelCount);

    for(size_t r = 0; r < 2 * (size_t)stateArray.rows() - 1; r++)
    {
        Eigen::Index pathwayRowIndex = r + borderRows;
        for(size_t c = 0; c < 2 * (size_t)stateArray.cols() - 1; c++)
        {
            Eigen::Index pathwayColIndex = c + borderRows;
            auto pPathwayIndex = labelledPenalizedPathway(pathwayRowIndex, pathwayColIndex);
            if(pPathwayIndex != 0)
            {
                if(r % 2 == 0 && c % 2 == 0 && r / 2 >= compZoneRowStart && r / 2 < compZoneRowEnd && 
                    c / 2 >= compZoneColEnd && c / 2 < compZoneColEnd && 
                    targetGeometry(r / 2 - compZoneRowStart, c / 2 - compZoneColStart) &&
                    !stateArray(r / 2, c / 2) && usableTargetSites.contains(std::tuple(r / 2, c / 2)))
                {
                    targetSitesPerPPathway[pPathwayIndex - 1].insert(std::tuple(r / 2, c / 2));
                }
                auto pathwayIndex = labelledPathway(pathwayRowIndex, pathwayColIndex);
                if(pathwayIndex != 0)
                {
                    pathwaysPerPPathway[pPathwayIndex - 1].insert(pathwayIndex - 1);
                }
            }
        }
    }

    std::vector<std::set<std::tuple<size_t,size_t>>> targetSitesPerPathway;
    targetSitesPerPathway.resize(labelCount);
    for(size_t pPathwayIndex = 0; pPathwayIndex < pLabelCount; pPathwayIndex++)
    {
        for(const auto& pathwayIndex : pathwaysPerPPathway[pPathwayIndex])
        {
            targetSitesPerPathway[pathwayIndex] = targetSitesPerPPathway[pPathwayIndex];
        }
    }

    return targetSitesPerPathway;
}

bool isDirectMove(std::set<std::tuple<size_t,size_t>> rows, std::set<std::tuple<size_t,size_t>> cols, std::shared_ptr<spdlog::logger> logger)
{
    unsigned int maxRowDist = 0;
    unsigned int maxColDist = 0;
    for(const auto& [start,end] : rows)
    {
        unsigned int dist = abs((int)end - (int)start);
        if(dist > maxRowDist)
        {
            maxRowDist = dist;
        }
    }
    for(const auto& [start,end] : cols)
    {
        unsigned int dist = abs((int)end - (int)start);
        if(dist > maxColDist)
        {
            maxColDist = dist;
        }
    }
    return (maxRowDist == 0 && maxColDist <= 1) || (maxRowDist <= 1 && maxColDist == 0);
}

std::optional<std::tuple<std::vector<size_t>,std::vector<size_t>,std::vector<std::tuple<bool,size_t,int>>,double>> checkMoveValidity(
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    const Eigen::Ref<const Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>>& labelledPathway, 
    const Eigen::Ref<const Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>>& penalizedPathway,
    std::set<std::tuple<size_t,size_t>>& unusableAtoms,  size_t compZoneRowStart, size_t compZoneColStart, 
    size_t borderRows, size_t borderCols, std::set<std::tuple<size_t,size_t>> cols, std::set<std::tuple<size_t,size_t>> rows,
    double bestFilledPerCost, std::shared_ptr<spdlog::logger> logger)
{
    double rowSpacing = Config::getInstance().rowSpacing;
    double colSpacing = Config::getInstance().columnSpacing;

    // Stores possible states after each movement step
    // A state contains the vector of row and col tones
    // Also it contains the list of moves that have been performed
    std::vector<std::set<std::tuple<std::vector<size_t>,std::vector<size_t>,
        std::vector<std::tuple<bool,size_t,int>>>,compareOnlyTones>> possibleTonesAfterStep;
    std::vector<size_t> initialRows;
    std::vector<size_t> initialCols;
    std::vector<size_t> targetRows;
    std::vector<size_t> targetCols;
    for(const auto& [startRow, endRow] : rows)
    {
        initialRows.push_back(2 * startRow);
        targetRows.push_back(2 * endRow);
    }
    for(const auto& [startCol, endCol] : cols)
    {
        initialCols.push_back(2 * startCol);
        targetCols.push_back(2 * endCol);
    }
    if(isDirectMove(rows, cols, logger))
    {
        logger->debug("Direct move possible");
        std::vector<std::tuple<bool,size_t,int>> path;
        if(targetRows != initialRows)
        {
            for(size_t i = 0; i < targetRows.size(); i++)
            {
                int diff = (int)targetRows[i] - (int)initialRows[i];
                if(diff != 0)
                {
                    path.push_back(std::tuple(true, i, diff));
                }
            }
        }
        if(targetCols != initialCols)
        {
            for(size_t i = 0; i < targetCols.size(); i++)
            {
                int diff = (int)targetCols[i] - (int)initialCols[i];
                if(diff != 0)
                {
                    path.push_back(std::tuple(false, i, diff));
                }
            }
        }
        return std::tuple(initialRows, initialCols, path, (double)(initialRows.size() * initialCols.size()) / moveCost(path));
    }
    std::optional<std::tuple<std::vector<size_t>,std::vector<size_t>,std::vector<std::tuple<bool,size_t,int>>,double>> bestSubsetMove = std::nullopt;

    std::tuple<std::vector<size_t>,std::vector<size_t>,std::vector<std::tuple<bool,size_t,int>>> 
        targetState(targetRows, targetCols, std::vector<std::tuple<bool,size_t,int>>());
    possibleTonesAfterStep.push_back(std::set<std::tuple<std::vector<size_t>,std::vector<size_t>,std::vector<std::tuple<bool,size_t,int>>>,compareOnlyTones>());
    possibleTonesAfterStep[0].insert(std::tuple(initialRows, initialCols, std::vector<std::tuple<bool,size_t,int>>()));
    for(size_t steps = 0; steps < possibleTonesAfterStep.size(); steps++)
    {
        logger->debug("{} possible locations after {} steps", possibleTonesAfterStep[steps].size(), steps);
        if(possibleTonesAfterStep[steps].empty())
        {
            return bestSubsetMove;
        }
        possibleTonesAfterStep.push_back(std::set<std::tuple<std::vector<size_t>,std::vector<size_t>,std::vector<std::tuple<bool,size_t,int>>>,compareOnlyTones>());
        for(const auto& [rowState, colState, path] : possibleTonesAfterStep[steps])
        {
            for(bool vertical : {true, false})
            {
                double halfSpacing = vertical ? (rowSpacing / 2.) : (colSpacing/ 2.);
                const auto& changingState = vertical ? rowState : colState;
                const auto& otherState = vertical ? colState : rowState;
                for(size_t toneIndex = 0; toneIndex < changingState.size(); toneIndex++)
                {
                    if(path.size() == 0 || std::get<0>(path.back()) != vertical || std::get<1>(path.back()) != toneIndex)
                    {
                        size_t tone = changingState[toneIndex];
                        for(int dir : {-1, 1})
                        {
                            bool movedIntoPenalizedPathway = false;
                            int maxDistIntoPenalizedPathway = Config::getInstance().maxSubmoveDistInPenalizedArea / halfSpacing;
                            for(int dist = 1;;dist++)
                            {
                                bool propagateState = true;
                                bool stopAfterChecking = false;
                                bool stopImmediately = false;
                                for(size_t otherToneIndex = 0; otherToneIndex < otherState.size(); otherToneIndex++)
                                {
                                    size_t otherTone = otherState[otherToneIndex];
                                    int r = vertical ? ((int)tone + dir * dist) : otherTone;
                                    int c = vertical ? otherTone : ((int)tone + dir * dist);
                                    if(r < 0 || r > 2 * (stateArray.rows() - 1) || c < 0 || c > 2 * (stateArray.cols() - 1))
                                    {
                                        stopImmediately = true;
                                        break;
                                    }
                                    else if(dist % 2 == 0 && stateArray(r / 2, c / 2))
                                    {
                                        stopImmediately = true;
                                        break;
                                    }
                                    else if((toneIndex > 0 && ((int)tone + dir * dist) <= (int)changingState[toneIndex - 1]) ||
                                        (toneIndex < (changingState.size() - 1) && ((int)tone + dir * dist) >= (int)changingState[toneIndex + 1]))
                                    {
                                        stopImmediately = true;
                                        break;
                                    }
                                    else if(labelledPathway(borderRows + r, borderCols + c) == 0)
                                    {
                                        if(penalizedPathway(borderRows + r, borderCols + c) <= 
                                            (pythagorasDist((r - (int)initialRows[vertical ? toneIndex : otherToneIndex]) * rowSpacing, 
                                            (c - (int)initialCols[vertical ? otherToneIndex : toneIndex]) * colSpacing) < 2 * Config::getInstance().minDistFromOccSites ? 1 : 0))
                                        {
                                            movedIntoPenalizedPathway = true;
                                        }
                                        else
                                        {
                                            stopImmediately = true;
                                        }
                                    }
                                }
                                std::vector<size_t> newRowState = rowState;
                                std::vector<size_t> newColState = colState;
                                if(vertical)
                                {
                                    newRowState[toneIndex] += dir * dist;
                                }
                                else
                                {
                                    newColState[toneIndex] += dir * dist;
                                }
                                if(movedIntoPenalizedPathway)
                                {
                                    bool toneWasLastMoved = false;
                                    for(auto step = path.rbegin(); step != path.rend(); ++step)
                                    {
                                        if(std::get<0>(*step) != vertical)
                                        {
                                            break;
                                        }
                                        else if(std::get<1>(*step) == toneIndex)
                                        {
                                            toneWasLastMoved = true;
                                            break;
                                        }
                                    }
                                    if(toneWasLastMoved)
                                    {
                                        stopImmediately = true;
                                    }
                                    else
                                    {
                                        maxDistIntoPenalizedPathway--;
                                        if(maxDistIntoPenalizedPathway < 0)
                                        {
                                            stopImmediately = true;
                                            break;
                                        }
                                    }
                                }
                                if(stopImmediately)
                                {
                                    break;
                                }
                                std::vector<std::tuple<bool,size_t,int>> newPath = path;
                                newPath.push_back(std::tuple(vertical, toneIndex, dir * dist));
                                std::tuple<std::vector<size_t>,std::vector<size_t>,std::vector<std::tuple<bool,size_t,int>>> 
                                    newState(newRowState, newColState, newPath);
                                double newCost = moveCost(newPath);
                                std::vector<size_t> rowIntersectionIndices;
                                for(size_t i = 0; i < targetRows.size(); i++)
                                {
                                    if(newRowState[i] == targetRows[i])
                                    {
                                        rowIntersectionIndices.push_back(i);
                                    }
                                }
                                std::vector<size_t> colIntersectionIndices;
                                for(size_t i = 0; i < targetCols.size(); i++)
                                {
                                    if(newColState[i] == targetCols[i])
                                    {
                                        colIntersectionIndices.push_back(i);
                                    }
                                }
                                double filledPerCost = (rowIntersectionIndices.size() * colIntersectionIndices.size()) / newCost;
                                if(filledPerCost > bestFilledPerCost && filledPerCost > DOUBLE_EQUIVALENCE_THRESHOLD && (!bestSubsetMove.has_value() || filledPerCost > std::get<3>(bestSubsetMove.value())))
                                {
                                    logger->debug("Writing new best subset move with filledPerCost {}", filledPerCost);
                                    std::vector<std::tuple<bool,size_t,int>> subsetPath;
                                    for(const auto& [vertical, index, dist] : newPath)
                                    {
                                        std::vector<size_t> *intersectionIndices;
                                        if(vertical)
                                        {
                                            intersectionIndices = &rowIntersectionIndices;
                                        }
                                        else
                                        {
                                            intersectionIndices = &colIntersectionIndices;
                                        }
                                        const auto iter = std::find(intersectionIndices->begin(), intersectionIndices->end(), index);
                                        if(iter != intersectionIndices->end())
                                        {
                                            subsetPath.push_back(std::tuple(vertical, std::distance(intersectionIndices->begin(), iter), dist));
                                        }
                                    }
                                    std::vector<size_t> initialRowsSubset;
                                    for(const auto& i : rowIntersectionIndices)
                                    {
                                        initialRowsSubset.push_back(initialRows[i]);
                                    }
                                    std::vector<size_t> initialColsSubset;
                                    for(const auto& i : colIntersectionIndices)
                                    {
                                        initialColsSubset.push_back(initialCols[i]);
                                    }
                                    bestSubsetMove = std::tuple(initialRowsSubset, initialColsSubset, subsetPath, filledPerCost);
                                }
                                if(bestSubsetMove.has_value() && targetRows == newRowState && targetCols == newColState)
                                {
                                    logger->debug("Full move possible");
                                    for(const auto& [vertical, index, dist] : std::get<2>(bestSubsetMove.value()))
                                    {
                                        logger->debug("Move {} at index {} by {}", vertical ? "vertically" : "horizontally", index, dist);
                                    }
                                    std::stringstream startIndices;
                                    startIndices << "StartRows: (";
                                    for(const auto& row : std::get<0>(bestSubsetMove.value()))
                                    {
                                        startIndices << row << " ";
                                    }
                                    startIndices << "), startCols: (";
                                    for(const auto& col : std::get<1>(bestSubsetMove.value()))
                                    {
                                        startIndices << col << " ";
                                    }
                                    startIndices << ")";
                                    logger->debug(startIndices.str());
                                    return bestSubsetMove;
                                }
                                if(propagateState)
                                {
                                    double bestPossibleFilledPerCost = (initialRows.size() * initialCols.size()) / newCost;
                                    if(bestPossibleFilledPerCost > bestFilledPerCost && (!bestSubsetMove.has_value() || bestPossibleFilledPerCost > std::get<3>(bestSubsetMove.value())))
                                    {
                                        bool alreadyLookedAt = false;
                                        for(size_t steps = 0; steps < possibleTonesAfterStep.size() - 1; steps++)
                                        {
                                            if(possibleTonesAfterStep[steps].contains(newState))
                                            {
                                                alreadyLookedAt = true;
                                                break;
                                            }
                                        }
                                        if(!alreadyLookedAt)
                                        {
                                            possibleTonesAfterStep[steps + 1].insert(newState);
                                        }
                                    }
                                }
                                if(stopAfterChecking)
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return bestSubsetMove;
}

std::optional<std::tuple<ParallelMove,double>> findDistOneMove(
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> penalizedPathway,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry,
    std::set<std::tuple<size_t,size_t>>& unusableAtoms,
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd,
    size_t borderRows, size_t borderCols, std::shared_ptr<spdlog::logger> logger)
{
    double minDistFromOccSites = Config::getInstance().minDistFromOccSites;
    double rowSpacing = Config::getInstance().rowSpacing;
    double colSpacing = Config::getInstance().columnSpacing;
    
    std::vector<RowBitMask> rowsToMoveIndexDown;
    std::vector<RowBitMask> rowsToMoveIndexUp;
    std::vector<RowBitMask> colsToMoveIndexDown;
    std::vector<RowBitMask> colsToMoveIndexUp;

    std::vector<RowBitMask> allowedRowsToMoveIndexDown;
    std::vector<RowBitMask> allowedRowsToMoveIndexUp;
    std::vector<RowBitMask> allowedColsToMoveIndexDown;
    std::vector<RowBitMask> allowedColsToMoveIndexUp;

    for(Eigen::Index row = 0; row < targetGeometry.rows(); row++)
    {
        colsToMoveIndexDown.push_back(RowBitMask(targetGeometry.cols(), row));
        colsToMoveIndexUp.push_back(RowBitMask(targetGeometry.cols(), row));
        allowedColsToMoveIndexDown.push_back(RowBitMask(targetGeometry.cols(), row));
        allowedColsToMoveIndexUp.push_back(RowBitMask(targetGeometry.cols(), row));
    }
    for(Eigen::Index col = 0; col < targetGeometry.cols(); col++)
    {
        rowsToMoveIndexDown.push_back(RowBitMask(targetGeometry.rows(), col));
        rowsToMoveIndexUp.push_back(RowBitMask(targetGeometry.rows(), col));
        allowedRowsToMoveIndexDown.push_back(RowBitMask(targetGeometry.rows(), col));
        allowedRowsToMoveIndexUp.push_back(RowBitMask(targetGeometry.rows(), col));
    }

    for(Eigen::Index row = 0; row < targetGeometry.rows(); row++)
    {
        for(Eigen::Index col = 0; col < targetGeometry.cols(); col++)
        {
            if(stateArray(row + compZoneRowStart, col + compZoneColStart) && !targetGeometry(row, col) && 
                penalizedPathway(borderRows + 2 * (row + compZoneRowStart), borderCols + 2 * (col + compZoneColStart)) <= 
                (minDistFromOccSites > 0 ? 1 : 0) && !unusableAtoms.contains(std::tuple(row + compZoneRowStart, col + compZoneColStart)))
            {
                if(row > 0 && !stateArray(row - 1 + compZoneRowStart, col + compZoneColStart) && targetGeometry(row - 1, col) && 
                    penalizedPathway(borderRows + 2 * (row - 1 + compZoneRowStart), borderCols + 2 * (col + compZoneColStart)) <= 
                    (minDistFromOccSites > rowSpacing ? 1 : 0))
                {
                    rowsToMoveIndexDown[col].set(row, true);
                }
                if(row < targetGeometry.rows() - 1 && !stateArray(row + 1 + compZoneRowStart, col + compZoneColStart) && targetGeometry(row + 1, col) && 
                    penalizedPathway(borderRows + 2 * (row + 1 + compZoneRowStart), borderCols + 2 * (col + compZoneColStart)) <= 
                    (minDistFromOccSites > rowSpacing ? 1 : 0))
                {
                    rowsToMoveIndexUp[col].set(row, true);
                }
                if(col > 0 && !stateArray(row + compZoneRowStart, col - 1 + compZoneColStart) && targetGeometry(row, col - 1) && 
                    penalizedPathway(borderRows + 2 * (row + compZoneRowStart), borderCols + 2 * (col - 1 + compZoneColStart)) <= 
                    (minDistFromOccSites > colSpacing ? 1 : 0))
                {
                    colsToMoveIndexDown[row].set(col, true);
                }
                if(col < targetGeometry.cols() - 1 && !stateArray(row + compZoneRowStart, col + 1 + compZoneColStart) && targetGeometry(row, col + 1) && 
                    penalizedPathway(borderRows + 2 * (row + compZoneRowStart), borderCols + 2 * (col + 1 + compZoneColStart)) <= 
                    (minDistFromOccSites > colSpacing ? 1 : 0))
                {
                    colsToMoveIndexUp[row].set(col, true);
                }
            }
            if(penalizedPathway(borderRows + 2 * (row + compZoneRowStart), borderCols + 2 * (col + compZoneColStart)) == 0)
            {
                if(row > 0 && penalizedPathway(borderRows + 2 * (row - 1 + compZoneRowStart), borderCols + 2 * (col + compZoneColStart)) == 0)
                {
                    allowedRowsToMoveIndexDown[col].set(row, true);
                }
                if(row < targetGeometry.rows() - 1 && penalizedPathway(borderRows + 2 * (row + 1 + compZoneRowStart), borderCols + 2 * (col + compZoneColStart)) == 0)
                {
                    allowedRowsToMoveIndexUp[col].set(row, true);
                }
                if(col > 0 && penalizedPathway(borderRows + 2 * (row + compZoneRowStart), borderCols + 2 * (col - 1 + compZoneColStart)) == 0)
                {
                    allowedColsToMoveIndexDown[row].set(col, true);
                }
                if(col < targetGeometry.cols() - 1 && penalizedPathway(borderRows + 2 * (row + compZoneRowStart), borderCols + 2 * (col + 1 + compZoneColStart)) == 0)
                {
                    allowedColsToMoveIndexUp[row].set(col, true);
                }
            }
        }
    }

    std::vector<std::tuple<RowBitMask,RowBitMask,bool,unsigned int>> bestOverlaps;
    std::vector<RowBitMask> perRowBitMaskIndexDown;
    std::vector<RowBitMask> perRowBitMaskIndexUp;
    for(Eigen::Index row = 0; row < targetGeometry.rows(); row++)
    {
        RowBitMask downBitMask = RowBitMask::fromOr(colsToMoveIndexDown[row], allowedColsToMoveIndexDown[row]);
        downBitMask.indices.pop_back();
        RowBitMask upBitMask = RowBitMask::fromOr(colsToMoveIndexUp[row], allowedColsToMoveIndexUp[row]);
        upBitMask.indices.pop_back();

        perRowBitMaskIndexDown.push_back(downBitMask);
        perRowBitMaskIndexUp.push_back(upBitMask);

        unsigned int sum = colsToMoveIndexDown[row].bitsSet() + colsToMoveIndexUp[row].bitsSet();
        if(sum > 0)
        {
            std::tuple<RowBitMask,RowBitMask,bool,unsigned int> elem(std::move(downBitMask), std::move(upBitMask), false, sum);
            auto insertLoc = std::upper_bound(bestOverlaps.begin(), bestOverlaps.end(), elem, 
                [](const std::tuple<RowBitMask,RowBitMask,bool,unsigned int>& lhs, 
                    const std::tuple<RowBitMask,RowBitMask,bool,unsigned int>& rhs)
                { return std::get<3>(lhs) < std::get<3>(rhs); });
            bestOverlaps.insert(insertLoc, std::move(elem));
        }
    }
    std::vector<RowBitMask> perColBitMaskIndexDown;
    std::vector<RowBitMask> perColBitMaskIndexUp;
    for(Eigen::Index col = 0; col < targetGeometry.cols(); col++)
    {
        RowBitMask downBitMask = RowBitMask::fromOr(rowsToMoveIndexDown[col], allowedRowsToMoveIndexDown[col]);
        downBitMask.indices.pop_back();
        RowBitMask upBitMask = RowBitMask::fromOr(rowsToMoveIndexUp[col], allowedRowsToMoveIndexUp[col]);
        upBitMask.indices.pop_back();

        perColBitMaskIndexDown.push_back(downBitMask);
        perColBitMaskIndexUp.push_back(upBitMask);

        unsigned int sum = rowsToMoveIndexDown[col].bitsSet() + rowsToMoveIndexUp[col].bitsSet();
        if(sum > 0)
        {
            std::tuple<RowBitMask,RowBitMask,bool,unsigned int> elem(std::move(downBitMask), std::move(upBitMask), true, sum);
            auto insertLoc = std::upper_bound(bestOverlaps.begin(), bestOverlaps.end(), elem, 
                [](const std::tuple<RowBitMask,RowBitMask,bool,unsigned int>& lhs, 
                    const std::tuple<RowBitMask,RowBitMask,bool,unsigned int>& rhs)
                { return std::get<3>(lhs) < std::get<3>(rhs); });
            bestOverlaps.insert(insertLoc, std::move(elem));
        }
    }

    std::optional<std::tuple<RowBitMask,RowBitMask,bool,unsigned int>> bestOverlap = std::nullopt;

    size_t iter = 0;
    while(!bestOverlaps.empty() && iter++ < 100)
    {
        auto currentElem = bestOverlaps.back();
        auto [downBitMask,upBitMask,vertical,overlap] = currentElem;
        bestOverlaps.pop_back();

        if(!bestOverlap.has_value() || overlap > std::get<3>(bestOverlap.value()))
        {
            logger->debug("Writing new best overlap with {} rows and {} cols", vertical ? downBitMask.bitsSet() + upBitMask.bitsSet() : downBitMask.indices.size(), 
                vertical ? downBitMask.indices.size() : (downBitMask.bitsSet() + upBitMask.bitsSet()));
            bestOverlap.emplace(currentElem);
        }
        std::vector<RowBitMask> *indexDownBitMasks;
        std::vector<RowBitMask> *indexUpBitMasks;

        std::vector<RowBitMask> *usefulIndicesDown;
        std::vector<RowBitMask> *usefulIndicesUp;

        if(vertical)
        {
            indexDownBitMasks = &perColBitMaskIndexDown;
            indexUpBitMasks = &perColBitMaskIndexUp;

            usefulIndicesDown = &rowsToMoveIndexDown;
            usefulIndicesUp = &rowsToMoveIndexUp;
        }
        else
        {
            indexDownBitMasks = &perRowBitMaskIndexDown;
            indexUpBitMasks = &perRowBitMaskIndexUp;

            usefulIndicesDown = &colsToMoveIndexDown;
            usefulIndicesUp = &colsToMoveIndexUp;
        }

        for(size_t newIndex = downBitMask.indices.back() + 1; newIndex < indexDownBitMasks->size(); newIndex++)
        {
            if((*usefulIndicesDown)[newIndex].bitsSet() > 0 || (*usefulIndicesUp)[newIndex].bitsSet() > 0)
            {
                auto newDownBitMask = RowBitMask::fromAnd(downBitMask, (*indexDownBitMasks)[newIndex]);
                auto newUpBitMask = RowBitMask::fromAnd(upBitMask, (*indexUpBitMasks)[newIndex]);
                unsigned int filledSites = 0;
                for(auto index : newDownBitMask.indices)
                {
                    filledSites += RowBitMask::fromAnd(newDownBitMask, (*usefulIndicesDown)[index]).bitsSet();
                    filledSites += RowBitMask::fromAnd(newUpBitMask, (*usefulIndicesUp)[index]).bitsSet();
                }
                if(filledSites > 0 && (newDownBitMask.bitsSet() + newUpBitMask.bitsSet()) > 1)
                {
                    std::tuple<RowBitMask,RowBitMask,bool,unsigned int> elem(std::move(newDownBitMask), 
                        std::move(newUpBitMask), vertical, filledSites);
                    auto insertLoc = std::upper_bound(bestOverlaps.begin(), bestOverlaps.end(), elem, 
                        [](const std::tuple<RowBitMask,RowBitMask,bool,unsigned int>& lhs, 
                            const std::tuple<RowBitMask,RowBitMask,bool,unsigned int>& rhs)
                        { return std::get<3>(lhs) < std::get<3>(rhs); });
                    bestOverlaps.insert(insertLoc, std::move(elem));
                }
            }
        }
    }
    if(bestOverlap.has_value())
    {
        logger->debug("Best overlap for dist one move: DownCount: {}, UpCount: {}, Verti: {}, Overlap: {}", 
            std::get<0>(bestOverlap.value()).bitsSet(), std::get<1>(bestOverlap.value()).bitsSet(),
            std::get<2>(bestOverlap.value()), std::get<3>(bestOverlap.value()));
        
        ParallelMove move;
        ParallelMove::Step start;
        ParallelMove::Step end;
        int lastTarget = -1;
        unsigned int actualOverlap = 0;
        if(std::get<2>(bestOverlap.value()))
        {
            RowBitMask usefulRowsDown(targetGeometry.rows());
            RowBitMask usefulRowsUp(targetGeometry.rows());

            start.colSelection.reserve(std::get<0>(bestOverlap.value()).indices.size());
            end.colSelection.reserve(std::get<0>(bestOverlap.value()).indices.size());
            for(auto& col : std::get<0>(bestOverlap.value()).indices)
            {
                if(RowBitMask::fromAnd(std::get<0>(bestOverlap.value()), rowsToMoveIndexDown[col]).bitsSet() > 0 ||
                    RowBitMask::fromAnd(std::get<1>(bestOverlap.value()), rowsToMoveIndexUp[col]).bitsSet() > 0)
                {
                    usefulRowsDown |= rowsToMoveIndexDown[col];
                    usefulRowsUp |= rowsToMoveIndexUp[col];
                    start.colSelection.push_back(col + compZoneColStart);
                    end.colSelection.push_back(col + compZoneColStart);
                }
            }
            for(size_t row = 0; row < std::get<0>(bestOverlap.value()).count; row++)
            {
                if(std::get<0>(bestOverlap.value())[row] && usefulRowsDown[row] && (int)row > lastTarget + 1)
                {
                    for(auto& col : start.colSelection)
                    {
                        if(rowsToMoveIndexDown[col - compZoneColStart][row])
                        {
                            actualOverlap++;
                        }
                    }
                    start.rowSelection.push_back(row + compZoneRowStart);
                    end.rowSelection.push_back(row + compZoneRowStart - 1);
                }
                else if(std::get<1>(bestOverlap.value())[row] && usefulRowsUp[row])
                {
                    for(auto& col : start.colSelection)
                    {
                        if(rowsToMoveIndexUp[col - compZoneColStart][row])
                        {
                            actualOverlap++;
                        }
                    }
                    start.rowSelection.push_back(row + compZoneRowStart);
                    end.rowSelection.push_back(row + compZoneRowStart + 1);
                    lastTarget = row + 1;
                }
            }
        }
        else
        {
            RowBitMask usefulColsDown(targetGeometry.cols());
            RowBitMask usefulColsUp(targetGeometry.cols());

            start.rowSelection.reserve(std::get<0>(bestOverlap.value()).indices.size());
            end.rowSelection.reserve(std::get<0>(bestOverlap.value()).indices.size());
            for(auto& row : std::get<0>(bestOverlap.value()).indices)
            {
                if(RowBitMask::fromAnd(std::get<0>(bestOverlap.value()), colsToMoveIndexDown[row]).bitsSet() > 0 ||
                    RowBitMask::fromAnd(std::get<1>(bestOverlap.value()), colsToMoveIndexUp[row]).bitsSet() > 0)
                {
                    usefulColsDown |= colsToMoveIndexDown[row];
                    usefulColsUp |= colsToMoveIndexUp[row];
                    start.rowSelection.push_back(row + compZoneRowStart);
                    end.rowSelection.push_back(row + compZoneRowStart);
                }
            }
            for(size_t col = 0; col < std::get<0>(bestOverlap.value()).count; col++)
            {
                if(std::get<0>(bestOverlap.value())[col] && usefulColsDown[col] && (int)col > lastTarget + 1)
                {
                    for(auto& row : start.rowSelection)
                    {
                        if(colsToMoveIndexDown[row - compZoneRowStart][col])
                        {
                            actualOverlap++;
                        }
                    }
                    start.colSelection.push_back(col + compZoneColStart);
                    end.colSelection.push_back(col + compZoneColStart - 1);
                }
                else if(std::get<1>(bestOverlap.value())[col] && usefulColsUp[col])
                {
                    for(auto& row : start.rowSelection)
                    {
                        if(colsToMoveIndexUp[row - compZoneRowStart][col])
                        {
                            actualOverlap++;
                        }
                    }
                    start.colSelection.push_back(col + compZoneColStart);
                    end.colSelection.push_back(col + compZoneColStart + 1);
                    lastTarget = col + 1;
                }
            }
        }
        move.steps.push_back(std::move(start));
        move.steps.push_back(std::move(end));

        double filledPerCost = (VALUE_FILLED_DESIRED + VALUE_USED_UNDESIRED) * (double)actualOverlap / move.cost();
        logger->debug("Returning distance one move that sorts {}, filledPerCost: {}", actualOverlap, filledPerCost);
        return std::tuple(move, filledPerCost);
    }
    else
    {
        logger->info("No better distance one move could be found");
        return std::nullopt;
    }
}

std::optional<std::tuple<ParallelMove,double>> findComplexMove(
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> penalizedPathway, 
    const Eigen::Ref<const Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>>& labelledPathway, unsigned int labelCount,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry,
    std::set<std::tuple<size_t,size_t>>& unusableAtoms, std::set<std::tuple<size_t,size_t>>& usableTargetSites,
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd,
    size_t borderRows, size_t borderCols, double bestIntPerCost, std::shared_ptr<spdlog::logger> logger)
{
    double rowSpacing = Config::getInstance().rowSpacing;
    double colSpacing = Config::getInstance().columnSpacing;
    
    std::optional<std::vector<size_t>> bestMoveStartRows = std::nullopt;
    std::optional<std::vector<size_t>> bestMoveStartCols = std::nullopt;
    std::optional<std::vector<std::tuple<bool,size_t,int>>> bestMovePath = std::nullopt;

    std::vector<std::vector<std::set<std::tuple<size_t,size_t>>>> movableRowsPerTargetColPerSourceCol;
    for(size_t col = 0; col < (size_t)stateArray.cols(); col++)
    {
        std::vector<std::set<std::tuple<size_t,size_t>>> movableRowsPerTargetCol;
        for(size_t tCol = 0; tCol < (size_t)stateArray.cols(); tCol++)
        {
            movableRowsPerTargetCol.push_back(std::set<std::tuple<size_t,size_t>>());
        }
        movableRowsPerTargetColPerSourceCol.push_back(std::move(movableRowsPerTargetCol));
    }

    auto targetSitesNeighboringPathways = findTargetSitesPerPathway(penalizedPathway,
            labelledPathway, usableTargetSites, borderRows, borderCols, labelCount, 
            compZoneRowStart, compZoneRowEnd, compZoneColStart, compZoneColEnd,
            stateArray, targetGeometry, logger);

    std::vector<std::tuple<size_t,size_t,size_t,size_t>> possibleMoves;

    auto occMask = generateMask(Config::getInstance().minDistFromOccSites, 0.5);
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> localAccessibility(occMask.rows() + 2, occMask.cols() + 2);

    for(size_t r = 0; r < (size_t)stateArray.rows(); r++)
    {
        Eigen::Index pathwayRowIndex = 2 * r + borderRows;
        for(size_t c = 0; c < (size_t)stateArray.cols(); c++)
        {
            if(stateArray(r, c) && !unusableAtoms.contains(std::tuple(r, c)) && 
                (r < compZoneRowStart || r >= compZoneRowEnd || c < compZoneColStart || c >= compZoneColEnd || 
                !targetGeometry(r - compZoneRowStart, c - compZoneColStart)))
            {
                Eigen::Index pathwayColIndex = 2 * c + borderCols;
                std::set<std::tuple<size_t,size_t>> accessibleTargetSites;
                bool changesWereMade = true;
                localAccessibility.fill(false);
                localAccessibility(localAccessibility.rows() / 2, localAccessibility.cols() / 2) = true;
                while(changesWereMade)
                {
                    changesWereMade = false;
                    for(Eigen::Index lRow = 0; lRow < localAccessibility.rows(); lRow++)
                    {
                        for(Eigen::Index lCol = 0; lCol < localAccessibility.cols(); lCol++)
                        {
                            if(localAccessibility(lRow, lCol))
                            {
                                if((localAccessibility.rows() / 2 - lRow) % 2 == 0 && (localAccessibility.cols() / 2 - lCol) % 2 == 0 && 
                                    usableTargetSites.contains(std::tuple(r - (localAccessibility.rows() / 2 - lRow) / 2,
                                        c - (localAccessibility.cols() / 2 - lCol) / 2)))
                                {
                                    accessibleTargetSites.insert(std::tuple(r - (localAccessibility.rows() / 2 - lRow) / 2,
                                        c - (localAccessibility.cols() / 2 - lCol) / 2));
                                }
                                if(labelledPathway(pathwayRowIndex - localAccessibility.rows() / 2 + lRow,
                                    pathwayColIndex - localAccessibility.cols() / 2 + lCol) != 0)
                                {
                                    for(const auto& targetSite : targetSitesNeighboringPathways[
                                        labelledPathway(pathwayRowIndex - localAccessibility.rows() / 2 + lRow,
                                            pathwayColIndex - localAccessibility.cols() / 2 + lCol) - 1])
                                    {
                                        accessibleTargetSites.insert(targetSite);
                                    }
                                }
                            }
                            else
                            {
                                bool withinOwnAtomsProhibitedArea = lRow > 0 && lRow < localAccessibility.rows() - 1 && 
                                    lCol > 0 && lCol < localAccessibility.cols() - 1 && occMask(lRow - 1, lCol - 1);
                                bool notInOtherPathway = penalizedPathway(pathwayRowIndex - localAccessibility.rows() / 2 + lRow,
                                    pathwayColIndex - localAccessibility.cols() / 2 + lCol) <= (withinOwnAtomsProhibitedArea ? 1 : 0);
                                bool adjacentToAccessibleLoc = 
                                    (lRow > 0 && localAccessibility(lRow - 1, lCol)) || 
                                    (lRow < localAccessibility.rows() - 1 && localAccessibility(lRow + 1, lCol)) || 
                                    (lCol > 0 && localAccessibility(lRow, lCol - 1)) || 
                                    (lCol < localAccessibility.cols() - 1 && localAccessibility(lRow, lCol + 1));
                                if(notInOtherPathway && adjacentToAccessibleLoc)
                                {
                                    localAccessibility(lRow, lCol) = true;
                                    changesWereMade = true;
                                }
                            }
                        }
                    }
                }
                for(const auto& [targetRow, targetCol] : accessibleTargetSites)
                {
                    possibleMoves.push_back(std::tuple(r, c, targetRow, targetCol));
                }
            }
        }
    }

    logger->debug("{} possible single moves found", possibleMoves.size());

    for(const auto& [pMStartRow, pMStartCol, pMEndRow, pMEndCol] : possibleMoves)
    {
        movableRowsPerTargetColPerSourceCol[pMStartCol][pMEndCol].insert(
            std::tuple(pMStartRow, pMEndRow));
    }

    for(size_t col = 0; col < (size_t)stateArray.cols(); col++)
    {
        for(size_t tCol = 0; tCol < (size_t)stateArray.cols(); tCol++)
        {
            double minCost = Config::getInstance().moveCostOffset + costPerSubMove(abs((int)tCol - (int)col) * colSpacing);
            if(movableRowsPerTargetColPerSourceCol[col][tCol].size() > 0 && (movableRowsPerTargetColPerSourceCol[col][tCol].size() / minCost) > bestIntPerCost)
            {
                std::vector<std::tuple<double,std::tuple<size_t,size_t>>> elemDist;
                for(size_t i = 0; i < movableRowsPerTargetColPerSourceCol[col][tCol].size(); i++)
                {
                    const auto& [startRow, endRow] = *std::next(movableRowsPerTargetColPerSourceCol[col][tCol].begin(), i);
                    std::tuple<double,std::tuple<size_t,size_t>> elem = std::tuple(abs((int)endRow - (int)startRow), std::tuple(startRow,endRow));
                    if(std::find_if(elemDist.begin(), elemDist.end(), [&startRow, &endRow](auto const& v) {
                        return (std::get<0>(std::get<1>(v)) == startRow) || (std::get<1>(std::get<1>(v)) == endRow);}) == elemDist.end())
                    {
                        elemDist.insert(std::upper_bound(elemDist.begin(), elemDist.end(), elem, 
                            [](const std::tuple<double,std::tuple<size_t,size_t>>& lhs, const std::tuple<double,std::tuple<size_t,size_t>>& rhs)
                            { return std::get<0>(lhs) < std::get<0>(rhs); }), elem);
                    }
                }
                double intPerCost = -1;
                std::optional<std::set<std::tuple<size_t,size_t>>> bestSubset = std::nullopt;
                std::set<std::tuple<size_t,size_t>> currentSubset;
                for(size_t usedRowCount = 0; usedRowCount < elemDist.size(); usedRowCount++)
                {
                    currentSubset.insert(std::get<1>(elemDist[usedRowCount]));
                    double localCountPerCost = (usedRowCount + 1) / (minCost + costPerSubMove(std::get<0>(elemDist[usedRowCount]) * rowSpacing));
                    if(localCountPerCost > intPerCost)
                    {
                        intPerCost = localCountPerCost;
                        bestSubset = currentSubset;
                    }
                }
                if(bestSubset.has_value() && intPerCost > bestIntPerCost)
                {
                    std::stringstream checkingMoveStream;
                    checkingMoveStream << "Checking move col " << col << " -> " << tCol << " with rows (";
                    for(const auto& [startRow,endRow] : bestSubset.value())
                    {
                        checkingMoveStream << startRow << " ";
                    }
                    checkingMoveStream << ") -> (";
                    for(const auto& [startRow,endRow] : bestSubset.value())
                    {
                        checkingMoveStream << endRow << " ";
                    }
                    checkingMoveStream << ")";
                    logger->debug(checkingMoveStream.str());
                    std::set<std::tuple<size_t,size_t>> cols;
                    cols.insert(std::tuple(col, tCol));
                    auto move = checkMoveValidity(stateArray, labelledPathway, penalizedPathway, unusableAtoms, compZoneRowStart, compZoneColStart, 
                        borderRows, borderCols, cols, bestSubset.value(), bestIntPerCost, logger);
                    if(move.has_value())
                    {
                        logger->debug("Returned move with {} rows and {} cols", std::get<0>(move.value()).size(), std::get<1>(move.value()).size());
                        if(std::get<3>(move.value()) > bestIntPerCost)
                        {
                            std::tie(bestMoveStartRows, bestMoveStartCols, bestMovePath, bestIntPerCost) = move.value();
                        }
                    }
                }
            }
        }
    }
    if(bestMoveStartRows.has_value() && bestMoveStartCols.has_value() && bestMovePath.has_value())
    {
        ParallelMove move;
        ParallelMove::Step step;
        for(const auto& row : bestMoveStartRows.value())
        {
            step.rowSelection.push_back((double)row / 2);
        }
        for(const auto& col : bestMoveStartCols.value())
        {
            step.colSelection.push_back((double)col / 2);
        }
        move.steps.push_back(step);
        bool verticalSegment = false;
        std::map<size_t,double> segmentDist;
        bool first = true;
        for(const auto& [vertical, index, dist] : bestMovePath.value())
        {
            if(first)
            {
                first = false;
                verticalSegment = vertical;
            }
            else if(vertical != verticalSegment)
            {
                std::vector<double> *changingSelection = &(verticalSegment ? step.rowSelection : step.colSelection);
                for(const auto& [segIndex, segDist] : segmentDist)
                {
                    (*changingSelection)[segIndex] += segDist;
                }
                move.steps.push_back(step);

                verticalSegment = vertical;
                segmentDist.clear();
            }
            segmentDist[index] += (double)dist / 2; // / 2 Because the path is expressed in units of half steps
        }
        std::vector<double> *changingSelection = &(verticalSegment ? step.rowSelection : step.colSelection);
        for(const auto& [segIndex, segDist] : segmentDist)
        {
            (*changingSelection)[segIndex] += segDist;
        }
        move.steps.push_back(step);

        logger->info("Returning complex move sorts {} rows and {} cols within computational zone, filledPerCost: {}", 
            move.steps[0].rowSelection.size(), move.steps[0].colSelection.size(), 2 * bestIntPerCost);
        return std::tuple(move, (VALUE_FILLED_DESIRED + VALUE_USED_UNDESIRED) * bestIntPerCost);
    }
    else
    {
        logger->info("No better complex move could be found");
        return std::nullopt;
    }
}

std::optional<ParallelMove> checkIntersectingPathwayMove(
    const std::vector<std::set<unsigned int>>& intersectingPathways,
    const std::map<unsigned int, std::tuple<std::tuple<size_t, std::vector<size_t>>, size_t>>& bestStartIPerPathway, 
    const std::map<unsigned int, std::tuple<std::tuple<size_t, std::vector<size_t>>, size_t>>& bestEndIPerPathway, 
    bool vertical, std::shared_ptr<spdlog::logger> logger)
{
    unsigned int bestMovableAtoms = 0;
    std::optional<std::tuple<unsigned int, unsigned int>> bestPathways = std::nullopt;
    std::set<unsigned int> bestIntersection;

    for(const auto& [pathway, startIAndJsAndOwnI] : bestStartIPerPathway)
    {
        const auto& [startIAndJs, ownI] = startIAndJsAndOwnI;
        const auto& [startI, startJs] = startIAndJs;
        if(startJs.size() > bestMovableAtoms)
        {
            for(const auto& [otherPathway, endIAndJsAndOwnI] : bestEndIPerPathway)
            {
                if(pathway != otherPathway)
                {
                    const auto& [otherEndIAndJs, otherOwnI] = endIAndJsAndOwnI;
                    const auto& [endI, endJs] = otherEndIAndJs;
                    if(endJs.size() > bestMovableAtoms)
                    {
                        std::set<unsigned int> intersection;
                        std::set_intersection(intersectingPathways[pathway].begin(), intersectingPathways[pathway].end(),
                            intersectingPathways[otherPathway].begin(), intersectingPathways[otherPathway].end(),
                            std::inserter(intersection, intersection.begin()));
                        if(intersection.size() > bestMovableAtoms)
                        {
                            unsigned int movableAtoms = startJs.size();
                            if(endJs.size() < movableAtoms)
                            {
                                movableAtoms = endJs.size();
                            }
                            if(intersection.size() < movableAtoms)
                            {
                                movableAtoms = intersection.size();
                            }
                            #pragma omp critical
                            {
                            if(!bestPathways.has_value() || movableAtoms > bestMovableAtoms)
                            {
                                bestMovableAtoms = movableAtoms;
                                bestPathways = std::tuple(pathway, otherPathway);
                                bestIntersection = intersection;
                            }
                            }
                        }
                    }
                }
            }
        }
    }
    if(bestPathways.has_value())
    {
        if(vertical)
        {
            if(bestMovableAtoms > Config::getInstance().aodRowLimit)
            {
                bestMovableAtoms = Config::getInstance().aodRowLimit;
            }
        }
        else
        {
            if(bestMovableAtoms > Config::getInstance().aodColLimit)
            {
                bestMovableAtoms = Config::getInstance().aodColLimit;
            }
        }
        if(bestMovableAtoms > Config::getInstance().aodTotalLimit)
        {
            bestMovableAtoms = Config::getInstance().aodTotalLimit;
        }
        const auto& [pathway, otherPathway] = bestPathways.value();
        ParallelMove move;
        ParallelMove::Step start, ontoFirstPathway, fromFirstPathway, ontoSecondPathway, fromSecondPathway, end;
        const auto& [startI, startJs] = std::get<0>(bestStartIPerPathway.at(pathway));
        auto firstPathwayI = std::get<1>(bestStartIPerPathway.at(pathway));
        const auto& [endI, endJs] = std::get<0>(bestEndIPerPathway.at(otherPathway));
        auto secondPathwayI = std::get<1>(bestEndIPerPathway.at(otherPathway));
        if(vertical)
        {
            start.colSelection.push_back(startI);
            for(size_t i = 0; i < bestMovableAtoms; i++)
            {
                start.rowSelection.insert(std::upper_bound(start.rowSelection.begin(), 
                    start.rowSelection.end(), startJs[i]), startJs[i]);
            }
            ontoFirstPathway.colSelection.push_back((double)firstPathwayI / 2.);
            ontoFirstPathway.rowSelection = start.rowSelection;
            fromFirstPathway.colSelection.push_back((double)firstPathwayI / 2.);
            for(const auto& intersectionIndex : bestIntersection)
            {
                fromFirstPathway.rowSelection.push_back((double)intersectionIndex / 2.);
                if(fromFirstPathway.rowSelection.size() == bestMovableAtoms)
                {
                    break;
                }
            }
            ontoSecondPathway.colSelection.push_back((double)secondPathwayI / 2.);
            ontoSecondPathway.rowSelection = fromFirstPathway.rowSelection;
            fromSecondPathway.colSelection.push_back((double)secondPathwayI / 2.);
            for(size_t i = 0; i < bestMovableAtoms; i++)
            {
                fromSecondPathway.rowSelection.insert(std::upper_bound(fromSecondPathway.rowSelection.begin(), 
                    fromSecondPathway.rowSelection.end(), endJs[i]), endJs[i]);
            }
            end.colSelection.push_back(endI);
            end.rowSelection = fromSecondPathway.rowSelection;
        }
        else
        {
            start.rowSelection.push_back(startI);
            for(size_t i = 0; i < bestMovableAtoms; i++)
            {
                start.colSelection.insert(std::upper_bound(start.colSelection.begin(), 
                    start.colSelection.end(), startJs[i]), startJs[i]);
            }
            ontoFirstPathway.rowSelection.push_back((double)firstPathwayI / 2.);
            ontoFirstPathway.colSelection = start.colSelection;
            fromFirstPathway.rowSelection.push_back((double)firstPathwayI / 2.);
            for(const auto& intersectionIndex : bestIntersection)
            {
                fromFirstPathway.colSelection.push_back((double)intersectionIndex / 2.);
                if(fromFirstPathway.colSelection.size() == bestMovableAtoms)
                {
                    break;
                }
            }
            ontoSecondPathway.rowSelection.push_back((double)secondPathwayI / 2.);
            ontoSecondPathway.colSelection = fromFirstPathway.colSelection;
            fromSecondPathway.rowSelection.push_back((double)secondPathwayI / 2.);
            for(size_t i = 0; i < bestMovableAtoms; i++)
            {
                fromSecondPathway.colSelection.insert(std::upper_bound(fromSecondPathway.colSelection.begin(), 
                    fromSecondPathway.colSelection.end(), endJs[i]), endJs[i]);
            }
            end.rowSelection.push_back(endI);
            end.colSelection = fromSecondPathway.colSelection;
        }

        move.steps.push_back(std::move(start));
        move.steps.push_back(std::move(ontoFirstPathway));
        move.steps.push_back(std::move(fromFirstPathway));
        move.steps.push_back(std::move(ontoSecondPathway));
        move.steps.push_back(std::move(fromSecondPathway));
        move.steps.push_back(std::move(end));
        logger->debug("Best intersecting pathway movable atom count: {}", bestMovableAtoms);
        return move;
    }
    logger->debug("No intersecting pathway move found");
    return std::nullopt;
}

std::optional<std::tuple<ParallelMove, double>> findPathwayMove(
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> penalizedPathway, 
    const Eigen::Ref<const Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>>& pathway,
    const Eigen::Ref<const Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>>& labelledPathway,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry,
    std::set<std::tuple<size_t,size_t>>& unusableAtoms, std::set<std::tuple<size_t,size_t>>& usableAtoms, 
    std::set<std::tuple<size_t,size_t>>& usableTargetSites,
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd,
    size_t borderRows, size_t borderCols, double bestIntPerCost, std::shared_ptr<spdlog::logger> logger)
{
    double halfRowSpacing = Config::getInstance().rowSpacing/ 2.;
    double halfColSpacing = Config::getInstance().columnSpacing / 2.;

    double bestLocalBenefitPerCost = 0;
    std::optional<ParallelMove> bestMove = std::nullopt;
    unsigned int bestToneCount = 0;

    Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic> byIPathways[2] = {
        Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>::Zero(pathway.rows(), pathway.cols()),
        Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>::Zero(pathway.rows(), pathway.cols())
    };
    unsigned int pathwayCounts[2] = {0, 0};
    std::vector<std::vector<unsigned int>> pathwaysPerIndex[2];

    auto startTime = std::chrono::steady_clock::now();

    for(bool vertical : {true,false})
    {
        Eigen::Index outerDimSize = vertical ? stateArray.cols() : stateArray.rows();
        size_t outerMaxIndex = 2 * outerDimSize - 1;
        size_t innerMaxIndex = vertical ? (2 * stateArray.rows() - 1) : (2 * stateArray.cols() - 1);
        pathwaysPerIndex[vertical].resize(outerMaxIndex);

        for(size_t i = 0; i < outerMaxIndex; i++)
        {
            std::vector<std::tuple<size_t,size_t>> startsAndEnds;
            bool inPathway = false;
            unsigned int pathwayLength = 0;
            for(size_t j = 0; j < innerMaxIndex; j++)
            {
                Eigen::Index pathwayRow = borderRows + (vertical ? j : i);
                Eigen::Index pathwayCol = borderCols + (vertical ? i : j);
                if(pathway(pathwayRow, pathwayCol) == 0)
                {
                    if(!inPathway)
                    {
                        pathwayLength = 1;
                        inPathway = true;
                    }
                    else
                    {
                        pathwayLength++;
                    }
                }
                else
                {
                    if(inPathway)
                    {
                        if(pathwayLength > 1)
                        {
                            pathwayCounts[vertical]++;
                            for(size_t jBackwards = 1; jBackwards <= pathwayLength; jBackwards++)
                            {
                                Eigen::Index pathwayRowBackwards = borderRows + (vertical ? j - jBackwards : i);
                                Eigen::Index pathwayColBackwards = borderCols + (vertical ? i : j - jBackwards);
                                byIPathways[vertical](pathwayRowBackwards, pathwayColBackwards) = pathwayCounts[vertical];
                            }
                            pathwaysPerIndex[vertical][i].push_back(pathwayCounts[vertical]);
                        }
                        inPathway = false;
                    }
                }
            }
            if(inPathway)
            {
                if(pathwayLength > 1)
                {
                    pathwayCounts[vertical]++;
                    for(size_t jBackwards = 1; jBackwards <= pathwayLength; jBackwards++)
                    {
                        Eigen::Index pathwayRowBackwards = borderRows + (vertical ? innerMaxIndex - jBackwards : i);
                        Eigen::Index pathwayColBackwards = borderCols + (vertical ? i : innerMaxIndex - jBackwards);
                        byIPathways[vertical](pathwayRowBackwards, pathwayColBackwards) = pathwayCounts[vertical];
                    }
                    pathwaysPerIndex[vertical][i].push_back(pathwayCounts[vertical]);
                }
            }
        }
    }
    logger->debug("Time for constructing pathways: {}us", (double)((std::chrono::steady_clock::now() - startTime).count()) / 1000.);
    startTime = std::chrono::steady_clock::now();


    std::vector<std::set<unsigned int>> intersectingPathways[2];
    intersectingPathways[true].resize(pathwayCounts[true] + 1);
    intersectingPathways[false].resize(pathwayCounts[false] + 1);
    for(Eigen::Index row = 0; row < 2 * stateArray.rows() - 1; row++)
    {
        for(Eigen::Index col = 0; col < 2 * stateArray.cols() - 1; col++)
        {
            const auto& pathwayIndex = byIPathways[true](borderRows + row, borderCols + col);
            if(pathwayIndex > 0)
            {
                const auto& intersectingPathwayIndex = byIPathways[false](borderRows + row, borderCols + col);
                if(intersectingPathwayIndex > 0)
                {
                    intersectingPathways[true][pathwayIndex].insert(intersectingPathwayIndex);
                    intersectingPathways[false][intersectingPathwayIndex].insert(pathwayIndex);
                }
            }
        }
    }
    logger->debug("Time for finding intersecting pathways: {}us", (double)((std::chrono::steady_clock::now() - startTime).count()) / 1000.);

    for(bool vertical : {true,false})
    {
        startTime = std::chrono::steady_clock::now();
        Eigen::Index outerDimSize = vertical ? stateArray.cols() : stateArray.rows();
        size_t outerMaxIndex = 2 * outerDimSize - 1;

        std::map<unsigned int, std::tuple<std::vector<std::tuple<size_t,size_t>>,std::vector<std::tuple<size_t,size_t>>>> 
            reachableTargetSitesAndValidSourceSitesPerPathway;

        for(const auto& [row, col] : usableTargetSites)
        {
            for(int dir : {-1, 1})
            {
                for(int dist = 1;; dist++)
                {
                    int newRow = 2 * row;
                    int newCol = 2 * col;
                    if(vertical)
                    {
                        newCol += dir * dist;
                        if(newCol < 0 || newCol >= 2 * stateArray.cols() - 1)
                        {
                            break;
                        }
                    }
                    else
                    {
                        newRow += dir * dist;
                        if(newRow < 0 || newRow >= 2 * stateArray.rows() - 1)
                        {
                            break;
                        }
                    }

                    unsigned int iPathway = byIPathways[vertical](borderRows + newRow, borderCols + newCol);
                    if(iPathway != 0)
                    {
                        std::get<0>(reachableTargetSitesAndValidSourceSitesPerPathway[iPathway]).push_back(std::tuple(row, col));
                    }
                    else if(penalizedPathway(borderRows + newRow, borderCols + newCol) > 0)
                    {
                        break;
                    }
                }
            }
        }
        logger->debug("Time for finding target sites: {}us", (double)((std::chrono::steady_clock::now() - startTime).count()) / 1000.);
        startTime = std::chrono::steady_clock::now();

        std::vector<std::vector<std::tuple<size_t,size_t,unsigned int,unsigned int>>> 
            validSourceSitesThatCanServeTwoPathwayPerI;
        validSourceSitesThatCanServeTwoPathwayPerI.resize(outerMaxIndex);

        for(const auto& [row, col] : usableAtoms)
        {
            for(int perpendicularDir : {-1, 1})
            {
                for(int pDist = (perpendicularDir + 1) / 2;; pDist++)
                {
                    int shiftedRow = 2 * row;
                    int shiftedCol = 2 * col;
                    if(vertical)
                    {
                        shiftedCol += perpendicularDir * pDist;
                        if(shiftedCol < 0 || shiftedCol >= 2 * stateArray.cols() - 1)
                        {
                            break;
                        }
                    }
                    else
                    {
                        shiftedRow += perpendicularDir * pDist;
                        if(shiftedRow < 0 || shiftedRow >= 2 * stateArray.rows() - 1)
                        {
                            break;
                        }
                    }
                    if((vertical ? halfColSpacing : halfRowSpacing) * pDist < Config::getInstance().minDistFromOccSites)
                    {
                        if(penalizedPathway(borderRows + shiftedRow, borderCols + shiftedCol) > 1)
                        {
                            break;
                        }
                        else
                        {
                            unsigned int pathwayNegative = 0;
                            unsigned int pathwayPositive = 0;
                            for(bool alongPathwayDirPositive : {true,false})
                            {
                                for(int alongPDirDist = 1;; alongPDirDist++)
                                {
                                    int twiceShiftedRow = shiftedRow;
                                    int twiceShiftedCol = shiftedCol;
                                    if(vertical)
                                    {
                                        twiceShiftedRow += alongPDirDist * (alongPathwayDirPositive ? 1 : -1);
                                        if(twiceShiftedRow < 0 || twiceShiftedRow >= 2 * stateArray.rows() - 1)
                                        {
                                            break;
                                        }
                                    }
                                    else
                                    {
                                        twiceShiftedCol += alongPDirDist * (alongPathwayDirPositive ? 1 : -1);
                                        if(twiceShiftedCol < 0 || twiceShiftedCol >= 2 * stateArray.cols() - 1)
                                        {
                                            break;
                                        }
                                    }
                                    if(pythagorasDist((double)(twiceShiftedRow - 2 * row) * halfRowSpacing, 
                                        (double)(twiceShiftedCol - 2 * col) * halfColSpacing) < Config::getInstance().minDistFromOccSites)
                                    {
                                        if(penalizedPathway(borderRows + twiceShiftedRow, borderCols + twiceShiftedCol) > 1)
                                        {
                                            break;
                                        }
                                    }
                                    else
                                    {
                                        if(alongPathwayDirPositive)
                                        {
                                            pathwayPositive = byIPathways[vertical](borderRows + twiceShiftedRow, borderCols + twiceShiftedCol);
                                        }
                                        else
                                        {
                                            pathwayNegative = byIPathways[vertical](borderRows + twiceShiftedRow, borderCols + twiceShiftedCol);
                                        }
                                        break;
                                    }
                                }
                            }

                            if(pathwayNegative == 0 || pathwayPositive == 0)
                            {
                                if(pathwayNegative != 0)
                                {
                                    std::get<1>(reachableTargetSitesAndValidSourceSitesPerPathway[pathwayNegative]).push_back(std::tuple(row, col));
                                }
                                if(pathwayPositive != 0)
                                {
                                    std::get<1>(reachableTargetSitesAndValidSourceSitesPerPathway[pathwayPositive]).push_back(std::tuple(row, col));
                                }
                            }
                            else if(vertical)
                            {
                                validSourceSitesThatCanServeTwoPathwayPerI[shiftedCol].push_back(
                                    std::tuple(row, col, pathwayNegative, pathwayPositive));
                            }
                            else
                            {
                                validSourceSitesThatCanServeTwoPathwayPerI[shiftedRow].push_back(
                                    std::tuple(row, col, pathwayNegative, pathwayPositive));
                            }
                        }
                    }
                    else
                    {
                        unsigned int iPathwayLabel = byIPathways[vertical](borderRows + shiftedRow, borderCols + shiftedCol);

                        if(iPathwayLabel > 0)
                        {
                            std::get<1>(reachableTargetSitesAndValidSourceSitesPerPathway[iPathwayLabel]).push_back(std::tuple(row, col));
                        }
                        else if(penalizedPathway(borderRows + shiftedRow, borderCols + shiftedCol) > 0)
                        {
                            break;
                        }
                    }
                }
            }
        }

        logger->debug("Time for finding source sites: {}us", (double)((std::chrono::steady_clock::now() - startTime).count()) / 1000.);
        startTime = std::chrono::steady_clock::now();

        std::map<unsigned int, std::tuple<std::tuple<size_t, std::vector<size_t>>, size_t>> bestStartIAndOwnIPerPathway;
        std::map<unsigned int, std::tuple<std::tuple<size_t, std::vector<size_t>>, size_t>> bestEndIAndOwnIPerPathway;

        for(size_t i = 0; i < outerMaxIndex; i++)
        {
            std::map<unsigned int,std::tuple<std::map<size_t,std::vector<size_t>>,std::map<size_t,std::vector<size_t>>>> 
                startAndEndIndicesPerPathway;
            for(const auto& pathway : pathwaysPerIndex[vertical][i])
            {
                if(reachableTargetSitesAndValidSourceSitesPerPathway.contains(pathway))
                {
                    const auto& [reachableTargetSites, validSourceSites] = reachableTargetSitesAndValidSourceSitesPerPathway[pathway];
                    std::map<size_t,std::vector<size_t>> startJsPerI;
                    if(validSourceSites.size() > 0)
                    {
                        for(const auto& [row,col] : validSourceSites)
                        {
                            if(vertical)
                            {
                                startJsPerI[col].push_back(row);
                            }
                            else
                            {
                                startJsPerI[row].push_back(col);
                            }
                        }
                        const auto& maxStartIAndJs = std::max_element(startJsPerI.begin(),
                            startJsPerI.end(), [](const auto& lhs, const auto& rhs) {
                                return std::get<1>(lhs).size() < std::get<1>(rhs).size();});
                        if(maxStartIAndJs != startJsPerI.end())
                        {
                            bestStartIAndOwnIPerPathway[pathway] = std::tuple(*maxStartIAndJs, i);
                        }
                    }
                    if(reachableTargetSites.size() > 0)
                    {
                        std::map<size_t,std::vector<size_t>> endJsPerI;
                        for(const auto& [row,col] : reachableTargetSites)
                        {
                            if(vertical)
                            {
                                endJsPerI[col].push_back(row);
                            }
                            else
                            {
                                endJsPerI[row].push_back(col);
                            }
                        }
                        const auto& maxEndIAndJs = std::max_element(endJsPerI.begin(),
                            endJsPerI.end(), [](const auto& lhs, const auto& rhs) {
                                return std::get<1>(lhs).size() < std::get<1>(rhs).size();});
                        if(maxEndIAndJs != endJsPerI.end())
                        {
                            bestEndIAndOwnIPerPathway[pathway] = std::tuple(*maxEndIAndJs, i);
                        }
                        startAndEndIndicesPerPathway[pathway] = std::tuple(std::move(startJsPerI), std::move(endJsPerI));
                    }
                }
            }

            for(const auto& [row, col, pathwayNegative, pathwayPositive] : validSourceSitesThatCanServeTwoPathwayPerI[i])
            {
                if(!bestStartIAndOwnIPerPathway.contains(pathwayNegative))
                {
                    bestStartIAndOwnIPerPathway[pathwayNegative] = std::tuple(std::tuple(vertical ? col : row, std::vector({vertical ? row : col})), i);
                }
                else if(std::get<0>(std::get<0>(bestStartIAndOwnIPerPathway[pathwayNegative])) == 
                    (vertical ? col : row))
                {
                    std::get<1>(std::get<0>(bestStartIAndOwnIPerPathway[pathwayNegative])).
                        push_back(vertical ? row : col);
                }
                if(!bestStartIAndOwnIPerPathway.contains(pathwayPositive))
                {
                    bestStartIAndOwnIPerPathway[pathwayPositive] = std::tuple(std::tuple(vertical ? col : row, std::vector({vertical ? row : col})), i);
                }
                else if(std::get<0>(std::get<0>(bestStartIAndOwnIPerPathway[pathwayPositive])) == 
                    (vertical ? col : row))
                {
                    std::get<1>(std::get<0>(bestStartIAndOwnIPerPathway[pathwayPositive])).
                        push_back(vertical ? row : col);
                }

                auto& [negativeSideStartJsPerI, negativeSideEndJsPerI] = startAndEndIndicesPerPathway[pathwayNegative];
                auto& [positiveSideStartJsPerI, positiveSideEndJsPerI] = startAndEndIndicesPerPathway[pathwayPositive];
                size_t sourceI;
                size_t sourceJ;
                if(vertical)
                {
                    sourceI = col;
                    sourceJ = row;
                }
                else
                {
                    sourceI = row;
                    sourceJ = col;
                }
                
                int negativeSideDiff = 0;
                int positiveSideDiff = 0;
                for(const auto& [i,negativeSideStartJs] : negativeSideStartJsPerI)
                {
                    negativeSideDiff += negativeSideStartJs.size();
                }
                for(const auto& [i,negativeSideEndJs] : negativeSideEndJsPerI)
                {
                    negativeSideDiff -= negativeSideEndJs.size();
                }
                for(const auto& [i,positiveSideStartJs] : positiveSideStartJsPerI)
                {
                    positiveSideDiff += positiveSideStartJs.size();
                }
                for(const auto& [i,positiveSideEndJs] : positiveSideEndJsPerI)
                {
                    positiveSideDiff -= positiveSideEndJs.size();
                }

                if(negativeSideDiff < positiveSideDiff)
                {
                    logger->debug("Atom at {}/{} could access {} and {}, diffs: {}; {}; adding to negative", row, 
                        col, pathwayNegative, pathwayPositive, negativeSideDiff, positiveSideDiff);
                    negativeSideStartJsPerI[sourceI].push_back(sourceJ);
                }
                else
                {
                    logger->debug("Atom at {}/{} could access {} and {}, diffs: {}; {}; adding to positive", row, 
                        col, pathwayNegative, pathwayPositive, negativeSideDiff, positiveSideDiff);
                    positiveSideStartJsPerI[sourceI].push_back(sourceJ);
                }
            }
            std::map<std::tuple<Eigen::Index,Eigen::Index>, double> movablePerStartAndEndI;
            unsigned int bestLocalToneCount = 0;
            Eigen::Index bestStartI = 0, bestEndI = 0;
            for(const auto& [pathway, startAndEndJsPerI] : startAndEndIndicesPerPathway)
            {
                const auto& [startJsPerI, endJsPerI] = startAndEndJsPerI;
                if(!startJsPerI.empty() && !endJsPerI.empty())
                {
                    for(const auto& [startI, startJs] : startJsPerI)
                    {
                        if(startJs.size() > 0)
                        {
                            for(const auto& [endI, endJs] : endJsPerI)
                            {
                                if(endJs.size() > 0)
                                {
                                    double newToneCount = movablePerStartAndEndI[std::tuple(startI, endI)] + (startJs.size() < endJs.size() ? 
                                        startJs.size() : endJs.size());
                                    movablePerStartAndEndI[std::tuple(startI, endI)] = newToneCount;
                                    if(newToneCount > bestLocalToneCount)
                                    {
                                        bestStartI = startI;
                                        bestEndI = endI;
                                        bestLocalToneCount = newToneCount;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if(bestLocalToneCount > 0)
            {
                unsigned int maxTones = Config::getInstance().aodTotalLimit;
                if(vertical && maxTones > Config::getInstance().aodRowLimit)
                {
                    maxTones = Config::getInstance().aodRowLimit;
                }
                else if(!vertical && maxTones > Config::getInstance().aodColLimit)
                {
                    maxTones = Config::getInstance().aodColLimit;
                }
                if(bestLocalToneCount > maxTones)
                {
                    bestLocalToneCount = maxTones;
                }
                bool withinCompZone = bestStartI >= (int)(vertical ? compZoneColStart : compZoneRowStart) && bestStartI < (int)(vertical ? compZoneColEnd : compZoneRowEnd);
                double maxBenefitPerCost = bestLocalToneCount * (VALUE_FILLED_DESIRED + (withinCompZone ? VALUE_USED_UNDESIRED : 0)) / 
                    (Config::getInstance().moveCostOffset + costPerSubMove(abs((int)i - bestStartI)) + costPerSubMove(abs((int)i - bestEndI)));
                if(maxBenefitPerCost > bestLocalBenefitPerCost && maxBenefitPerCost > bestIntPerCost && bestLocalToneCount > 0)
                {
                    ParallelMove move;
                    ParallelMove::Step start;
                    ParallelMove::Step step1;
                    ParallelMove::Step step2;
                    ParallelMove::Step end;
                    std::vector<double> *startIndices;
                    std::vector<double> *endIndices;
                    if(vertical)
                    {
                        start.colSelection.push_back(bestStartI);
                        step1.colSelection.push_back((double)i / 2);
                        step2.colSelection.push_back((double)i / 2);
                        end.colSelection.push_back(bestEndI);
                        startIndices = &start.rowSelection;
                        endIndices = &end.rowSelection;
                    }
                    else
                    {
                        start.rowSelection.push_back(bestStartI);
                        step1.rowSelection.push_back((double)i / 2);
                        step2.rowSelection.push_back((double)i / 2);
                        end.rowSelection.push_back(bestEndI);
                        startIndices = &start.colSelection;
                        endIndices = &end.colSelection;
                    }
                    for(const auto& [pathway, startAndEndJsPerI] : startAndEndIndicesPerPathway)
                    {
                        const auto& [startJsPerI, endJsPerI] = startAndEndJsPerI;
                        if(startJsPerI.contains((size_t)bestStartI) && endJsPerI.contains((size_t)bestEndI))
                        {
                            const auto& startJs = startJsPerI.at((size_t)bestStartI);
                            const auto& endJs = endJsPerI.at((size_t)bestEndI);
                            size_t sitesRequired = startJs.size() < endJs.size() ? startJs.size() : endJs.size();
                            if(sitesRequired + startIndices->size() > maxTones)
                            {
                                sitesRequired = maxTones - startIndices->size();
                            }
                            if(sitesRequired > 0)
                            {
                                for(size_t siteIndex = 0; siteIndex < sitesRequired; siteIndex++)
                                {
                                    startIndices->insert(std::upper_bound(startIndices->begin(), startIndices->end(), startJs[siteIndex]), startJs[siteIndex]);
                                    endIndices->insert(std::upper_bound(endIndices->begin(), endIndices->end(), endJs[siteIndex]), endJs[siteIndex]);
                                }
                            }
                        }
                    }

                    double moveBenefit = start.rowSelection.size() * start.colSelection.size() * VALUE_FILLED_DESIRED;
                    for(const auto& row : start.rowSelection)
                    {
                        if(row >= compZoneRowStart && row < compZoneRowEnd)
                        {
                            for(const auto& col : start.colSelection)
                            {
                                if(col >= compZoneColStart && col < compZoneColEnd)
                                {
                                    moveBenefit += VALUE_USED_UNDESIRED;
                                }
                            }
                        }
                    }
                    move.steps.push_back(start);
                    if((int)i != bestStartI * 2)
                    {
                        if(vertical)
                        {
                            step1.rowSelection = start.rowSelection;
                        }
                        else
                        {
                            step1.colSelection = start.colSelection;
                        }
                        move.steps.push_back(std::move(step1));
                    }
                    if((int)i != bestEndI * 2)
                    {
                        if(vertical)
                        {
                            step2.rowSelection = end.rowSelection;
                        }
                        else
                        {
                            step2.colSelection = end.colSelection;
                        }
                        move.steps.push_back(std::move(step2));
                    }
                    move.steps.push_back(std::move(end));
                    double actualBenefitPerCost = moveBenefit / move.cost();
                    if(actualBenefitPerCost > bestLocalBenefitPerCost)
                    {
                        bestLocalBenefitPerCost = actualBenefitPerCost;
                        bestMove = std::move(move);
                        bestToneCount = bestLocalToneCount;
                    }
                }
            }
        }
        logger->debug("Time for iterating over all indices: {}us", (double)((std::chrono::steady_clock::now() - startTime).count()) / 1000.);
        startTime = std::chrono::steady_clock::now();

        // Try to find move that uses two pathways that are connected by a number of perpendicular pathways
        // Since this is generally longer, only consider pathways that have more source/target sites and connecting pathways than the best currently known move uses
        std::erase_if(bestStartIAndOwnIPerPathway, [bestToneCount, &intersectingPathways, vertical](const auto& item)
        {
            return std::get<1>(std::get<0>(std::get<1>(item))).size() <= bestToneCount || 
                intersectingPathways[vertical][std::get<0>(item)].size() <= bestToneCount;
        });
        std::erase_if(bestEndIAndOwnIPerPathway, [bestToneCount, &intersectingPathways, vertical](const auto& item)
        {
            return std::get<1>(std::get<0>(std::get<1>(item))).size() <= bestToneCount || 
                intersectingPathways[vertical][std::get<0>(item)].size() <= bestToneCount;
        });
        
        auto move = checkIntersectingPathwayMove(intersectingPathways[vertical], 
            bestStartIAndOwnIPerPathway, bestEndIAndOwnIPerPathway, vertical, logger);
        if(move.has_value())
        {
            double moveBenefit = move.value().steps[0].rowSelection.size() * move.value().steps[0].colSelection.size() * VALUE_FILLED_DESIRED;
            for(const auto& row : move.value().steps[0].rowSelection)
            {
                if(row >= compZoneRowStart && row < compZoneRowEnd)
                {
                    for(const auto& col : move.value().steps[0].colSelection)
                    {
                        if(col >= compZoneColStart && col < compZoneColEnd)
                        {
                            moveBenefit += VALUE_USED_UNDESIRED;
                        }
                    }
                }
            }
            moveBenefit /= move.value().cost();
            if(moveBenefit > bestLocalBenefitPerCost)
            {
                bestMove = move.value();
                bestLocalBenefitPerCost = moveBenefit;
            }
        }
        logger->debug("Time for finding intersecting pathway move: {}us", (double)((std::chrono::steady_clock::now() - startTime).count()) / 1000.);
        startTime = std::chrono::steady_clock::now();
    }

    if(bestMove.has_value())
    {
        logger->info("Returning pathway move that fills {}, filledPerCost: {}", 
            bestMove.value().steps[0].rowSelection.size() * bestMove.value().steps[0].colSelection.size(), 
            bestLocalBenefitPerCost);
        return std::tuple(bestMove.value(), bestLocalBenefitPerCost);
    }
    else
    {
        logger->info("No better pathway move could be found");
        return std::nullopt;
    }
}

bool handleUnusableSection(std::vector<std::tuple<bool,int>>::iterator unusableSectionStart,
    std::vector<std::tuple<bool,int>>::iterator sectionEndExclusive, int firstFinalLocation, int lastFinalLocation, int blockingDist,
    std::vector<std::tuple<int,int>>& newShifts, std::shared_ptr<spdlog::logger> logger)
{
    bool first = true;
    bool lastAtomUsable = false;
    int lastShiftedAtomLocation = firstFinalLocation - 1;
    for(auto unusableAtom = unusableSectionStart; unusableAtom != sectionEndExclusive; unusableAtom++)
    {
        const auto& [usable, sourceLocation] = *unusableAtom;
        if(!first)
        {
            firstFinalLocation++;
            if(usable || lastAtomUsable)
            {
                firstFinalLocation += (int)blockingDist;
            }
        }
        first = false;
    
        if(usable && !Config::getInstance().allowMultipleMovesPerAtom && sourceLocation < firstFinalLocation)
        {
            logger->warn("Atom may not be moved to non-target position as another move would not be allowed");
            return false;
        }
        else if(firstFinalLocation > lastFinalLocation)
        {
            logger->warn("No more parking position for unusable atom in linear move found");
            return false;
        }
        else if(sourceLocation >= firstFinalLocation)
        {
            firstFinalLocation = sourceLocation;
            break;
        }

        lastShiftedAtomLocation = firstFinalLocation;
        newShifts.push_back(std::tuple(sourceLocation, firstFinalLocation));
        lastAtomUsable = usable;
    }
    first = true;
    for(auto unusableAtom = std::make_reverse_iterator(sectionEndExclusive); unusableAtom != std::make_reverse_iterator(unusableSectionStart); unusableAtom++)
    {
        const auto& [usable, sourceLocation] = *unusableAtom;
        if(!first)
        {
            lastFinalLocation--;
            if(usable || lastAtomUsable)
            {
                lastFinalLocation -= (int)blockingDist;
            }
        }
        first = false;
    
        if(usable && !Config::getInstance().allowMultipleMovesPerAtom && sourceLocation > lastFinalLocation)
        {
            logger->warn("Atom may not be moved to non-target position as another move would not be allowed");
            return false;
        }
        else if(lastFinalLocation <= lastShiftedAtomLocation)
        {
            logger->warn("Left-shifting part of linearly moving unusable atoms overlaps with right-shifting one");
            return false;
        }
        else if(sourceLocation <= lastFinalLocation)
        {
            firstFinalLocation = sourceLocation;
            break;
        }

        newShifts.push_back(std::tuple(sourceLocation, lastFinalLocation));
        lastAtomUsable = usable;
    }
    return true;
}

std::optional<std::tuple<std::vector<std::tuple<int,int>>, double>> analyzeLinearMovementStretch(
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry,
    bool vertical, int outerIndex, std::vector<std::tuple<bool,int>>& sourceAtoms, int& sectionStart, int sectionEnd, 
    size_t innerDimMax, bool sectionOverlapsCompZone,
    unsigned int blockingDist, size_t compZoneStart, size_t compZoneEnd, size_t outerCompZoneStart, 
    std::shared_ptr<spdlog::logger> logger)
{
    std::vector<std::tuple<int,int>> newShifts;
    double shiftBenefit = 0;

    std::vector<std::tuple<bool,int>> compZoneOrder; // 1 means requires usable atom, 0 means parking spot
    std::vector<size_t> targetSitesInCompZoneOrder;
    unsigned int nonTargetLength = 0;
    bool firstSection = true;
    for(int index = sectionStart; index < sectionEnd; index++)
    {
        if(index < (int)compZoneStart || index >= (int)compZoneEnd || 
            ((vertical && !targetGeometry(index - compZoneStart, outerIndex - outerCompZoneStart)) ||
            (!vertical && !targetGeometry(outerIndex - outerCompZoneStart, index - compZoneStart))))
        {
            nonTargetLength++;
        }
        else
        {
            if(firstSection)
            {
                firstSection = false;
            }
            int gapBefore = firstSection ? 0 : blockingDist;
            if((int)nonTargetLength >= (int)blockingDist + 1 + gapBefore)
            {
                for(int backIndex = index - nonTargetLength + gapBefore; 
                    backIndex < index - (int)blockingDist; backIndex++)
                {
                    if(backIndex >= (int)compZoneStart && backIndex < (int)compZoneEnd)
                    {
                        compZoneOrder.push_back(std::tuple(false, backIndex));
                    }
                }
            }
            compZoneOrder.push_back(std::tuple(true, index));
            targetSitesInCompZoneOrder.push_back(compZoneOrder.size() - 1);
            firstSection = false;
            nonTargetLength = 0;
        }
    }
    int gapBefore = firstSection ? 0 : blockingDist;
    if((int)nonTargetLength >= gapBefore + 1)
    {
        for(int backIndex = sectionEnd - nonTargetLength + gapBefore; 
            backIndex < sectionEnd; backIndex++)
        {
            if(backIndex >= (int)compZoneStart && backIndex < (int)compZoneEnd)
            {
                compZoneOrder.push_back(std::tuple(false, backIndex));
            }
        }
    }

    if(sectionStart == 0)
    {
        for(auto iter = sourceAtoms.begin(); iter != sourceAtoms.end();)
        {
            const auto& [usable, atomIndex] = *iter;
            if(usable)
            {
                break;
            }
            else
            {
                newShifts.push_back(std::tuple(atomIndex, -1));
                shiftBenefit += ((atomIndex >= (int)compZoneStart && atomIndex < (int)compZoneEnd) ? 
                    VALUE_CLEARED_UNDESIRED_UNUSABLE : VALUE_CLEARED_OUTSIDE_UNUSABLE);
                iter = sourceAtoms.erase(iter);
            }
        }
    }
    if(sectionEnd == (int)innerDimMax)
    {
        for(auto iter = sourceAtoms.rbegin(); iter != sourceAtoms.rend();)
        {
            const auto& [usable, atomIndex] = *iter;
            if(usable)
            {
                break;
            }
            else
            {
                newShifts.push_back(std::tuple(atomIndex, innerDimMax));
                shiftBenefit += ((atomIndex >= (int)compZoneStart && atomIndex < (int)compZoneEnd) ? 
                    VALUE_CLEARED_UNDESIRED_UNUSABLE : VALUE_CLEARED_OUTSIDE_UNUSABLE);
                sourceAtoms.erase((++iter).base());
            }
        }
    }

    // Only try to shift atoms if there are valid source and target sites
    if(targetSitesInCompZoneOrder.size() > 0 && std::find_if(sourceAtoms.begin(), sourceAtoms.end(), 
        [](const auto& val)
        {
            return std::get<0>(val);
        }) != sourceAtoms.end())
    {
        std::vector<std::tuple<std::vector<std::tuple<bool,int>>::iterator,size_t>> 
            bestUsedAtomsAndTargetSites;

        int minimumTargetSpot = sectionStart - (int)blockingDist - 1;
        bool lastAtomUsable = true;
        const int lastTargetSite = std::get<1>(compZoneOrder[targetSitesInCompZoneOrder.back()]);
        for(auto iter = sourceAtoms.begin(); iter != sourceAtoms.end(); iter++)
        {
            const auto& [firstAtomUsable, firstAtomLocation] = *iter;

            bool requiresGap = iter != sourceAtoms.begin() || (!firstAtomUsable && !lastAtomUsable);

            int shiftedAtomLocation = minimumTargetSpot;
            bool lastShiftedAtomUsable = lastAtomUsable;
            bool targetSitesRemaining = true;
            bool thisAtomAsStartAllowed = true;

            std::vector<std::tuple<std::vector<std::tuple<bool,int>>::iterator,size_t>> 
                usedAtomsAndTargetSites;

            for(auto compZoneIter = iter; compZoneIter != sourceAtoms.end(); compZoneIter++)
            {
                const auto& [usable, atomLocation] = *compZoneIter;
                if(usable && targetSitesRemaining)
                {
                    auto shiftIndex = std::find_if(targetSitesInCompZoneOrder.begin(), targetSitesInCompZoneOrder.end(),
                        [&compZoneOrder, &shiftedAtomLocation, &blockingDist](const auto& targetSiteIndex)
                        {
                            return std::get<1>(compZoneOrder[targetSiteIndex]) > shiftedAtomLocation + (int)blockingDist;
                        });
                    if(shiftIndex != targetSitesInCompZoneOrder.end())
                    {
                        shiftedAtomLocation = std::get<1>(compZoneOrder[*shiftIndex]);
                        usedAtomsAndTargetSites.push_back(std::tuple(compZoneIter, shiftedAtomLocation));
                    }
                    else
                    {
                        targetSitesRemaining = false;
                    }
                }
                if(!usable || !targetSitesRemaining)
                {
                    shiftedAtomLocation++;
                    if(lastShiftedAtomUsable || usable)
                    {
                        shiftedAtomLocation += blockingDist;
                    }
                }
                if(shiftedAtomLocation >= lastTargetSite)
                {
                    targetSitesRemaining = false;
                }
                if(!targetSitesRemaining && shiftedAtomLocation <= atomLocation)
                {
                    break;
                }
                if(shiftedAtomLocation >= sectionEnd)
                {
                    thisAtomAsStartAllowed = false;
                    break;
                }
            }
            if(thisAtomAsStartAllowed && usedAtomsAndTargetSites.size() > bestUsedAtomsAndTargetSites.size())
            {
                bestUsedAtomsAndTargetSites = usedAtomsAndTargetSites;
            }

            if(firstAtomUsable && !Config::getInstance().allowMultipleMovesPerAtom)
            {
                minimumTargetSpot = firstAtomLocation;
            }
            else
            {
                minimumTargetSpot++;
                if(requiresGap)
                {
                    minimumTargetSpot += blockingDist;
                }
            }

            lastAtomUsable = firstAtomUsable;
        }

        if(bestUsedAtomsAndTargetSites.size() > 0)
        {
            std::vector<std::tuple<bool,int>>::iterator unusableSectionStart = sourceAtoms.begin();
            int firstUsableTargetLocation = sectionStart;

            for(const auto& [usedAtom, targetSite] : bestUsedAtomsAndTargetSites)
            {
                if(!handleUnusableSection(unusableSectionStart, usedAtom, firstUsableTargetLocation, targetSite - (int)blockingDist - 1, blockingDist, newShifts, logger))
                {
                    return std::nullopt;
                }
                int usedSourceLocation = std::get<1>(*usedAtom);
                newShifts.push_back(std::tuple(usedSourceLocation, targetSite));
                shiftBenefit += VALUE_FILLED_DESIRED;
                if(usedSourceLocation >= (int)compZoneStart && usedSourceLocation < (int)compZoneEnd &&
                    ((vertical && !targetGeometry(usedSourceLocation - compZoneStart, outerIndex - outerCompZoneStart)) ||
                    (!vertical && !targetGeometry(outerIndex - outerCompZoneStart, usedSourceLocation - compZoneStart))))
                {
                    shiftBenefit += VALUE_USED_UNDESIRED;
                }
                unusableSectionStart = usedAtom + 1;
                firstUsableTargetLocation = targetSite + (int)blockingDist + 1;
            }
            if(!handleUnusableSection(unusableSectionStart, sourceAtoms.end(), firstUsableTargetLocation, sectionEnd - 1, blockingDist, newShifts, logger))
            {
                return std::nullopt;
            }

            logger->debug("Best starting atom can fill {} positions", bestUsedAtomsAndTargetSites.size());
            return std::tuple(newShifts, shiftBenefit);
        }
    }

    return std::nullopt;
}

std::optional<std::tuple<ParallelMove, double>> findLinearMove(
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> penalizedPathway, 
    const Eigen::Ref<const Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>>& pathway,
    const Eigen::Ref<const Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>>& labelledPathway,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry,
    std::set<std::tuple<size_t,size_t>>& unusableAtoms, std::set<std::tuple<size_t,size_t>>& usableAtoms, 
    std::set<std::tuple<size_t,size_t>>& usableTargetSites,
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd,
    size_t borderRows, size_t borderCols, double bestIntPerCost, std::shared_ptr<spdlog::logger> logger)
{
    std::optional<ParallelMove> bestMove;
    double bestBenefitPerCost = 0;
    for(bool vertical : {true, false})
    {
        unsigned int maxTones = Config::getInstance().aodTotalLimit;
        if(vertical && maxTones > Config::getInstance().aodRowLimit)
        {
            maxTones = Config::getInstance().aodRowLimit;
        }
        else if(!vertical && maxTones > Config::getInstance().aodColLimit)
        {
            maxTones = Config::getInstance().aodColLimit;
        }

        size_t outerStartIndex, outerEndIndex, innerCompZoneStart, innerCompZoneEnd, innerEndIndex;
        if(vertical)
        {
            outerStartIndex = compZoneColStart;
            outerEndIndex = compZoneColEnd;
            innerCompZoneStart = compZoneRowStart;
            innerCompZoneEnd = compZoneRowEnd;
            innerEndIndex = stateArray.rows();
        }
        else
        {
            outerStartIndex = compZoneRowStart;
            outerEndIndex = compZoneRowEnd;
            innerCompZoneStart = compZoneColStart;
            innerCompZoneEnd = compZoneColEnd;
            innerEndIndex = stateArray.cols();
        }

        int blockingDistAlongIndex = ceil(Config::getInstance().minDistFromOccSites / 
            (vertical ? Config::getInstance().rowSpacing : Config::getInstance().columnSpacing)) - 1;
        size_t innerDimMax = vertical ? stateArray.rows() : stateArray.cols();
        for(size_t outerIndex = outerStartIndex; outerIndex < outerEndIndex; outerIndex++)
        {
            Eigen::Array<unsigned int, -1, 1> pathwayAlongIndex;
            if(vertical)
            {
                pathwayAlongIndex = penalizedPathway.col(borderCols + 2 * outerIndex)(Eigen::seqN(borderRows, stateArray.rows(), 2));
            }
            else
            {
                pathwayAlongIndex = penalizedPathway.row(borderRows + 2 * outerIndex)(Eigen::seqN(borderCols, stateArray.cols(), 2));
            }

            for(size_t innerIndex = 0; innerIndex < innerEndIndex; innerIndex++)
            {
                Eigen::Index row = vertical ? innerIndex : outerIndex;
                Eigen::Index col = vertical ? outerIndex : innerIndex;
                if(stateArray(row, col))
                {
                    Eigen::Index sliceStart = innerIndex - blockingDistAlongIndex;
                    if(sliceStart < 0)
                    {
                        sliceStart = 0;
                    }
                    size_t sliceEnd = innerIndex + blockingDistAlongIndex;
                    if(sliceEnd >= innerEndIndex)
                    {
                        sliceEnd = innerEndIndex - 1;
                    }
                    pathwayAlongIndex(Eigen::seq(sliceStart, sliceEnd)) -= 1;
                }
            }

            int sectionStart = 0;
            std::vector<std::tuple<bool,int>> sourceAtoms;
            bool sectionOverlapsCompZone = false;
            bool inSection = false;
            bool sectionContainsUsableAtoms = false;
            bool sectionContainsTargetSites = false;
            std::vector<std::tuple<std::vector<std::tuple<int,int>>, double>> shiftsPerSection;
            for(size_t innerIndex = 0; innerIndex < innerDimMax; innerIndex++)
            {
                bool terminateSection = false;
                Eigen::Index row = vertical ? innerIndex : outerIndex;
                Eigen::Index col = vertical ? outerIndex : innerIndex;
                bool inCompZone = innerIndex >= innerCompZoneStart && innerIndex < innerCompZoneEnd;
                // Don't move past a point that is a non-pathway due to external atoms
                if(pathwayAlongIndex(innerIndex) > 0)
                {
                    terminateSection = true;
                    if(inSection)
                    {
                        if(sectionStart == 0 || (sectionContainsTargetSites && sectionContainsUsableAtoms))
                        {
                            auto localShifts = analyzeLinearMovementStretch(targetGeometry, vertical, outerIndex, sourceAtoms, sectionStart, 
                                innerIndex, innerDimMax, sectionOverlapsCompZone, blockingDistAlongIndex, innerCompZoneStart,
                                innerCompZoneEnd, outerStartIndex, logger);
                            if(localShifts.has_value())
                            {
                                shiftsPerSection.push_back(std::move(localShifts.value()));
                            }
                        }
                    }
                }
                else
                {
                    if(stateArray(row, col))
                    {
                        bool usable = !unusableAtoms.contains(std::tuple(row, col));
                        if(usable)
                        {
                            if(inCompZone && targetGeometry(row - compZoneRowStart, col - compZoneColStart) && 
                                !Config::getInstance().allowMultipleMovesPerAtom)
                            {
                                terminateSection = true;
                                if(inSection)
                                {
                                    if(sectionStart == 0 || (sectionContainsTargetSites && sectionContainsUsableAtoms))
                                    {
                                        auto localShifts = analyzeLinearMovementStretch(targetGeometry, vertical, outerIndex, sourceAtoms, sectionStart, 
                                            innerIndex - blockingDistAlongIndex, innerDimMax, sectionOverlapsCompZone, blockingDistAlongIndex, innerCompZoneStart,
                                            innerCompZoneEnd, outerStartIndex, logger);
                                        if(localShifts.has_value())
                                        {
                                            shiftsPerSection.push_back(std::move(localShifts.value()));
                                        }
                                    }
                                    innerIndex += blockingDistAlongIndex;
                                }
                            }
                            else
                            {
                                sectionContainsUsableAtoms = true;
                                sourceAtoms.push_back(std::tuple(true, innerIndex));
                            }
                        }
                        else
                        {
                            sourceAtoms.push_back(std::tuple(false, innerIndex));
                        }
                    }
                }

                if(terminateSection)
                {
                    sourceAtoms.clear();
                    sectionOverlapsCompZone = false;
                    inSection = false;
                    sectionContainsUsableAtoms = false;
                    sectionContainsTargetSites = false;
                }
                else
                {
                    sectionOverlapsCompZone = sectionOverlapsCompZone || inCompZone;
                    if(inCompZone)
                    {
                        sectionContainsTargetSites = sectionContainsTargetSites || targetGeometry(row - compZoneRowStart, col - compZoneColStart);
                    }
                    if(!inSection)
                    {
                        sectionStart = innerIndex;
                    }
                    inSection = true;
                }
            }
            if(inSection)
            {
                auto localShifts = analyzeLinearMovementStretch(targetGeometry, vertical, outerIndex, sourceAtoms, sectionStart, 
                    innerDimMax, innerDimMax, sectionOverlapsCompZone, blockingDistAlongIndex, innerCompZoneStart,
                    innerCompZoneEnd, outerStartIndex, logger);
                if(localShifts.has_value())
                {
                    shiftsPerSection.push_back(std::move(localShifts.value()));
                }
            }

            double totalBenefit = 0;
            unsigned int sumOfRequiredMoves = 0;
            for(const auto& [shifts, benefit] : shiftsPerSection)
            {
                totalBenefit += benefit;
                sumOfRequiredMoves += shifts.size();
            }
            std::vector<std::tuple<int,int>> plannedShifts;
            if(sumOfRequiredMoves > maxTones)
            {
                std::sort(shiftsPerSection.begin(), shiftsPerSection.end(), [](const auto& lhs, const auto& rhs)
                    {
                        return std::get<1>(lhs) / std::get<0>(lhs).size() > std::get<1>(rhs) / std::get<0>(rhs).size();
                    });
                totalBenefit = 0;
                sumOfRequiredMoves = 0;
                for(auto iter = shiftsPerSection.begin(); iter != shiftsPerSection.end();)
                {
                    const auto& [shifts, benefit] = *iter;
                    if(sumOfRequiredMoves + shifts.size() > maxTones)
                    {
                        iter = shiftsPerSection.erase(iter);
                    }
                    else
                    {
                        sumOfRequiredMoves += shifts.size();
                        totalBenefit += benefit;
                        iter++;
                    }
                }
            }

            int maxShiftDist = 0;
            for(const auto& [shifts, benefit] : shiftsPerSection)
            {
                for(const auto& [shiftStart, shiftEnd] : shifts)
                {
                    int dist = abs(shiftEnd - shiftStart);
                    if(dist > maxShiftDist)
                    {
                        maxShiftDist = dist;
                    }
                }
            }

            double benefitPerCost = totalBenefit / (Config::getInstance().moveCostOffset + costPerSubMove(maxShiftDist));
            if(benefitPerCost > bestBenefitPerCost)
            {
                ParallelMove move;
                ParallelMove::Step start, end;
                std::vector<double> *startInnerSelection, *endInnerSelection;
                if(vertical)
                {
                    start.colSelection.push_back(outerIndex);
                    end.colSelection.push_back(outerIndex);
                    startInnerSelection = &start.rowSelection;
                    endInnerSelection = &end.rowSelection;
                }
                else
                {
                    start.rowSelection.push_back(outerIndex);
                    end.rowSelection.push_back(outerIndex);
                    startInnerSelection = &start.colSelection;
                    endInnerSelection = &end.colSelection;
                }
                for(const auto& [shifts, benefit] : shiftsPerSection)
                {
                    for(const auto& [shiftStart, shiftEnd] : shifts)
                    {
                        startInnerSelection->push_back(shiftStart);
                        endInnerSelection->push_back(shiftEnd);
                    }
                }
                std::sort(startInnerSelection->begin(), startInnerSelection->end());
                std::sort(endInnerSelection->begin(), endInnerSelection->end());
                int count = 0;
                for(double tone : *endInnerSelection)
                {
                    if(tone < 0)
                    {
                        count++;
                    }
                    else
                    {
                        break;
                    }
                }
                for(int i = 0; i < count; i++)
                {
                    (*endInnerSelection)[i] = -count + i;
                }
                count = 0;
                for(double tone : *endInnerSelection | std::views::reverse)
                {
                    if(tone > innerDimMax - 1)
                    {
                        count++;
                    }
                    else
                    {
                        break;
                    }
                }
                for(int i = 0; i < count; i++)
                {
                    (*endInnerSelection)[endInnerSelection->size() - count + i] = innerDimMax + i;
                }

                move.steps.push_back(std::move(start));
                move.steps.push_back(std::move(end));
                bestMove = move;
                bestBenefitPerCost = benefitPerCost;
            }
        }
    }

    if(bestMove.has_value())
    {
        logger->info("Returning linear move with {} tones and benefitPerCost: {}", 
            bestMove.value().steps[0].rowSelection.size() * bestMove.value().steps[0].colSelection.size(), bestBenefitPerCost);
        return std::tuple(bestMove.value(), bestBenefitPerCost);
    }
    else
    {
        logger->info("No better linear move could be found");
        return std::nullopt;
    }
}

bool canAtomsBeMovedSimultaneously(
    Eigen::Ref<Eigen::Array<std::optional<std::tuple<int,int,double,unsigned int>>, Eigen::Dynamic, Eigen::Dynamic>> sitesPath,
    const std::set<std::tuple<int,int>>& currentRemovalSet, const std::tuple<int,int>& otherMovableAtom, 
    size_t moveDist, bool lastMoveVertical, int pathwayRowCount)
{
    for(const auto& existingNode : currentRemovalSet)
    {
        std::tuple<int,int> pathNode = existingNode;
        std::tuple<int,int> otherNode = otherMovableAtom;
        for(size_t i = 0; i <= moveDist; i++)
        {
            if(pathNode == otherNode)
            {
                return false;
            }
            const auto& nextPathNode = sitesPath(std::get<0>(pathNode), std::get<1>(pathNode));
            const auto& nextOtherNode = sitesPath(std::get<0>(otherNode), std::get<1>(otherNode));
            if(!nextPathNode.has_value() || !nextOtherNode.has_value())
            {
                return false;
            }
            else
            {
                if(lastMoveVertical)
                {
                    if(std::get<1>(pathNode) != std::get<1>(otherNode) || 
                        (i < moveDist && ((std::get<0>(pathNode) < std::get<0>(otherNode)) != 
                        (std::get<0>(nextPathNode.value()) < std::get<0>(nextOtherNode.value())))))
                    {
                        return false;
                    }
                }
                else
                {
                    if(std::get<0>(pathNode) != std::get<0>(otherNode) || 
                        (i < moveDist && ((std::get<1>(pathNode) < std::get<1>(otherNode)) != 
                        (std::get<1>(nextPathNode.value()) < std::get<1>(nextOtherNode.value())))))
                    {
                        return false;
                    }
                }
                pathNode = std::tuple(std::get<0>(nextPathNode.value()), std::get<1>(nextPathNode.value()));
                otherNode = std::tuple(std::get<0>(nextOtherNode.value()), std::get<1>(nextOtherNode.value()));
            }
        }
        bool otherLastVertical = std::get<0>(otherNode) == -1 ||
            std::get<0>(otherNode) == pathwayRowCount;
        if(otherLastVertical != lastMoveVertical)
        {
            return false;
        }
    }
    return true;
}

std::optional<std::tuple<ParallelMove,double>> removeAtomsInBorderPathway(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    std::set<std::tuple<size_t,size_t>>& unusableAtoms, size_t compZoneRowStart, size_t compZoneRowEnd, 
    size_t compZoneColStart, size_t compZoneColEnd, size_t borderRows, size_t borderCols,
    std::vector<ParallelMove>& moveList, std::shared_ptr<spdlog::logger> logger)
{
    double rowSpacing = Config::getInstance().rowSpacing;
    double colSpacing = Config::getInstance().columnSpacing;

    auto startTime = std::chrono::steady_clock::now();
    // Generate pathway only for usable atoms. Since removed atoms are discarded, we don't care about heating -> Penalized pathway
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> stateArrayCopy = stateArray;
    for(const auto& [r, c] : unusableAtoms)
    {
        stateArrayCopy(r,c) = false;
    }

    auto pathway = generatePathway(borderRows, borderCols, stateArrayCopy, Config::getInstance().minDistFromOccSites, 0);
    auto [labelledPathway, labelCount] = labelPathway(pathway);
    
    // Find straight pathways to the outside. Removed atoms will be extracted through here
    std::vector<std::tuple<size_t, bool, bool>> outsidePathwayVerticalAtLowIndex;
    // Vertical pathways
    for(size_t col = 2 * compZoneColStart; col <= 2 * (compZoneColEnd - 1); col++)
    {
        bool isPathway = true;
        for(size_t row = 0; row < 2 * compZoneRowStart; row++)
        {
            if(pathway(row + borderRows, col + borderCols) > 0)
            {
                isPathway = false;
                break;
            }
        }
        if(isPathway)
        {
            outsidePathwayVerticalAtLowIndex.push_back(std::tuple(col, true, true));
        }

        isPathway = true;
        for(size_t row = 2 * (stateArray.rows() - 1); row > 2 * (compZoneRowEnd - 1); row--)
        {
            if(pathway(row + borderRows, col + borderCols) > 0)
            {
                isPathway = false;
                break;
            }
        }
        if(isPathway)
        {
            outsidePathwayVerticalAtLowIndex.push_back(std::tuple(col, true, false));
        }
    }
    // Horizontal pathways
    for(size_t row = 2 * compZoneRowStart; row <= 2 * (compZoneRowEnd - 1); row++)
    {
        bool isPathway = true;
        for(size_t col = 0; col < 2 * compZoneColStart; col++)
        {
            if(pathway(row + borderRows, col + borderCols) > 0)
            {
                isPathway = false;
                break;
            }
        }
        if(isPathway)
        {
            outsidePathwayVerticalAtLowIndex.push_back(std::tuple(row, false, true));
        }

        isPathway = true;
        for(size_t col = 2 * (stateArray.cols() - 1); col > 2 * (compZoneColEnd - 1); col--)
        {
            if(pathway(row + borderRows, col + borderCols) > 0)
            {
                isPathway = false;
                break;
            }
        }
        if(isPathway)
        {
            outsidePathwayVerticalAtLowIndex.push_back(std::tuple(row, false, false));
        }
    }

    int pathwayRowCount = 2 * stateArray.rows() - 1;
    int pathwayColCount = 2 * stateArray.cols() - 1;
    Eigen::Array<std::optional<std::tuple<int,int,double,unsigned int>>, Eigen::Dynamic, Eigen::Dynamic> sitesPath = 
        Eigen::Array<std::optional<std::tuple<int,int,double,unsigned int>>, Eigen::Dynamic, Eigen::Dynamic>::Constant(
            pathwayRowCount, pathwayColCount, std::nullopt);

    // Insert starting points for inward search for atoms to be removed
    // Start just outside comp zone
    std::vector<std::vector<std::tuple<int,int>>> reachableSitesAfterMove;
    reachableSitesAfterMove.push_back(std::vector<std::tuple<int,int>>());
    for(const auto& [index, vertical, atLowIndex] : outsidePathwayVerticalAtLowIndex)
    {
        if(vertical)
        {
            if(atLowIndex)
            {
                reachableSitesAfterMove[0].push_back(std::tuple(-1, index));
            }
            else
            {
                reachableSitesAfterMove[0].push_back(std::tuple(pathwayRowCount, index));
            }
        }
        else
        {
            if(atLowIndex)
            {
                reachableSitesAfterMove[0].push_back(std::tuple(index, -1));
            }
            else
            {
                reachableSitesAfterMove[0].push_back(std::tuple(index, pathwayColCount));
            }
        }
    }

    logger->debug("Time for finding outside path and setup: {}us", (double)((std::chrono::steady_clock::now() - startTime).count()) / 1000.);
    startTime = std::chrono::steady_clock::now();

    for(size_t moveDist = 0; !reachableSitesAfterMove[moveDist].empty(); moveDist++)
    {
        startTime = std::chrono::steady_clock::now();
        reachableSitesAfterMove.push_back(std::vector<std::tuple<int,int>>());
        std::vector<std::tuple<int,int>> currentlyMovableAtoms;
        std::vector<std::tuple<int,int>> currentlyMovableNonTargetAtoms;
        logger->debug("{} sites reachable at dist {}", reachableSitesAfterMove[moveDist].size(), moveDist);
        #pragma omp for schedule(static, 8)
        for(const auto& [row,col] : reachableSitesAfterMove[moveDist])
        {
            bool nextMoveVertical;
            double cost = 0;
            if(moveDist == 0)
            {
                nextMoveVertical = row == -1 || row == pathwayRowCount;
            }
            else
            {
                auto lastLoc = sitesPath(row,col);
                if(!lastLoc.has_value())
                {
                    logger->warn("Last site in pathway path not found");
                    continue;
                }
                cost = std::get<2>(lastLoc.value());
                nextMoveVertical = std::get<0>(lastLoc.value()) == row;
            }
            for(int dir : {-1,1})
            {
                int rowDir = nextMoveVertical ? dir : 0;
                int colDir = nextMoveVertical ? 0 : dir;
                bool hasCrossedAtom = false;
                for(int dist = 1;; dist++)
                {
                    int currRow = row + dist * rowDir;
                    int currCol = col + dist * colDir;
                    if(currRow < 0 || currRow >= pathwayRowCount || 
                        currCol < 0 || currCol >= pathwayColCount || 
                        pathway(borderRows + currRow, borderCols + currCol) > 0)
                    {
                        break;
                    }
                    else if(currRow % 2 == 0 && currCol % 2 == 0 && stateArray(currRow / 2, currCol / 2))
                    {
                        hasCrossedAtom = true;
                        double newCost = cost + costPerSubMove(dist * (nextMoveVertical ? rowSpacing : colSpacing));
                        if(sitesPath(currRow, currCol).has_value())
                        {
                            if(newCost < std::get<2>(sitesPath(currRow, currCol).value()) && std::get<3>(sitesPath(currRow, currCol).value()) == moveDist)
                            {
                                #pragma omp critical (sitesPath)
                                sitesPath(currRow, currCol) = std::tuple(row, col, newCost, moveDist);
                            }
                        }
                        else
                        {
                            #pragma omp critical (movable)
                            {
                            if(isInCompZone(currRow / 2, currCol / 2, compZoneRowStart, 
                                compZoneRowEnd, compZoneColStart, compZoneColEnd))
                            {
                                currentlyMovableAtoms.push_back(std::tuple(currRow, currCol));
                            }
                            else
                            {
                                currentlyMovableNonTargetAtoms.push_back(std::tuple(currRow, currCol));
                            }
                            }
                            #pragma omp critical (sitesPath)
                            sitesPath(currRow, currCol) = std::tuple(row, col, newCost, moveDist);
                        }
                    }
                    else if(!hasCrossedAtom && currentlyMovableAtoms.empty())
                    {
                        double newCost = cost + costPerSubMove(dist * (nextMoveVertical ? rowSpacing : colSpacing));
                        if(sitesPath(currRow, currCol).has_value())
                        {
                            if(newCost < std::get<2>(sitesPath(currRow, currCol).value()) && std::get<3>(sitesPath(currRow, currCol).value()) == moveDist)
                            {
                                #pragma omp critical (sitesPath)
                                sitesPath(currRow, currCol) = std::tuple(row, col, newCost, moveDist);
                            }
                        }
                        else
                        {
                            #pragma omp critical (reachable)
                            reachableSitesAfterMove[moveDist + 1].push_back(std::tuple(currRow, currCol));
                            #pragma omp critical (sitesPath)
                            sitesPath(currRow, currCol) = std::tuple(row, col, newCost, moveDist);
                        }
                    }
                }
            }
        }

        logger->debug("Time for finding locations after submove {}: {}us", moveDist, (double)((std::chrono::steady_clock::now() - startTime).count()) / 1000.);
        startTime = std::chrono::steady_clock::now();

        logger->debug("{} removable atoms at move dist {}", currentlyMovableAtoms.size(), moveDist);
        if(currentlyMovableAtoms.size() > 0)
        {
            std::vector<std::tuple<std::set<std::tuple<int,int>>, bool, double>> currentRemovalSets;
            for(const auto& movableAtom : currentlyMovableAtoms)
            {
                bool currentLastMoveVertical = false;
                std::tuple<int,int> pathNode = movableAtom;
                for(size_t i = 0; i <= moveDist; i++)
                {
                    auto nextNode = sitesPath(std::get<0>(pathNode), std::get<1>(pathNode));
                    if(!nextNode.has_value())
                    {
                        logger->warn("Pathway move could not be reconstructed (Node does not exist)");
                        continue;
                    }
                    else
                    {
                        pathNode = std::tuple(std::get<0>(nextNode.value()), std::get<1>(nextNode.value()));
                    }
                }
                if(std::get<0>(pathNode) == -1 || std::get<0>(pathNode) == pathwayRowCount)
                {
                    currentLastMoveVertical = true;
                }
                else if(std::get<1>(pathNode) == -1 || std::get<1>(pathNode) == pathwayColCount)
                {
                    currentLastMoveVertical = false;
                }
                else
                {
                    logger->warn("Pathway move could not be reconstructed (End not found in move limit)");
                    continue;
                }

                if(currentRemovalSets.empty())
                {
                    currentRemovalSets.push_back(std::tuple(std::set({movableAtom}), currentLastMoveVertical, VALUE_CLEARED_UNDESIRED_UNUSABLE));
                }
                else
                {
                    bool inserted = false;
                    for(auto& [removalSet, lastMoveVertical, removalSetBenefit] : currentRemovalSets)
                    {
                        if(lastMoveVertical == currentLastMoveVertical && 
                            canAtomsBeMovedSimultaneously(sitesPath, removalSet, movableAtom, moveDist, lastMoveVertical, pathwayRowCount))
                        {
                            inserted = true;
                            removalSet.insert(movableAtom);
                            removalSetBenefit += VALUE_CLEARED_UNDESIRED_UNUSABLE;
                        }
                    }
                    if(!inserted)
                    {
                        currentRemovalSets.push_back(std::tuple(std::set({movableAtom}), currentLastMoveVertical, VALUE_CLEARED_UNDESIRED_UNUSABLE));
                    }
                }
            }

            if(currentRemovalSets.empty())
            {
                logger->warn("No removal sets found even though there should have been removable atoms");
                return std::nullopt;
            }
            else
            {
                for(const auto& moveableNonTargetAtom : currentlyMovableNonTargetAtoms)
                {
                    for(auto& [removalSet, lastMoveVertical, removalSetBenefit] : currentRemovalSets)
                    {
                        if(canAtomsBeMovedSimultaneously(sitesPath, removalSet, moveableNonTargetAtom, moveDist, lastMoveVertical, pathwayRowCount))
                        {
                            removalSet.insert(moveableNonTargetAtom);
                            removalSetBenefit += VALUE_CLEARED_OUTSIDE_UNUSABLE;
                        }
                    }
                }

                auto removalSetWithBestBenefit = std::max_element(currentRemovalSets.begin(), currentRemovalSets.end(), [](const auto& lhs, const auto& rhs)
                {
                    return std::get<2>(lhs) < std::get<2>(rhs);
                });
                if(removalSetWithBestBenefit != currentRemovalSets.end())
                {
                    logger->debug("Time for finding best removal set: {}us", (double)((std::chrono::steady_clock::now() - startTime).count()) / 1000.);
                    startTime = std::chrono::steady_clock::now();
                    
                    const auto& [bestRemovalSet, bestLastMoveVertical, bestMoveBenefit] = *removalSetWithBestBenefit;
                    std::vector<std::tuple<int,int>> currentSites(bestRemovalSet.begin(), bestRemovalSet.end());
                    std::vector<std::tuple<int,int>> nextStepSites;

                    ParallelMove move;
                    for(size_t pathIndex = 0; pathIndex <= moveDist; pathIndex++)
                    {
                        ParallelMove::Step step;
                        if(bestLastMoveVertical)
                        {
                            step.colSelection.push_back((double)(std::get<1>(
                                *currentSites.begin())) / 2.);
                        }
                        else
                        {
                            step.rowSelection.push_back((double)(std::get<0>(
                                *currentSites.begin())) / 2.);
                        }
                        for(const auto& [row,col] : currentSites)
                        {
                            if(bestLastMoveVertical)
                            {
                                step.rowSelection.insert(std::upper_bound(step.rowSelection.begin(), 
                                    step.rowSelection.end(), (double)(row) / 2.), (double)(row) / 2.);
                            }
                            else
                            {
                                step.colSelection.insert(std::upper_bound(step.colSelection.begin(), 
                                    step.colSelection.end(), (double)(col) / 2.), (double)(col) / 2.);
                            }
                            const auto& nextSite = sitesPath(row, col);
                            if(!nextSite.has_value())
                            {
                                logger->warn("Pathway move could not be reconstructed during move creation (Node does not exist)");
                                return std::nullopt;
                            }
                            else
                            {
                                nextStepSites.push_back(std::tuple(std::get<0>(nextSite.value()), std::get<1>(nextSite.value())));
                            }
                        }
                        currentSites = nextStepSites;
                        nextStepSites.clear();
                        move.steps.push_back(std::move(step));
                    }
                    ParallelMove::Step end;
                    if(bestLastMoveVertical)
                    {
                        end.colSelection.push_back((double)(std::get<1>(*currentSites.begin())) / 2.);
                        int lowEndCount = 0, highEndCount = 0;
                        for(const auto& [row, col] : currentSites)
                        {
                            if(row == -1)
                            {
                                lowEndCount++;
                            }
                            else
                            {
                                highEndCount++;
                            }
                        }
                        for(int i = 0; i < lowEndCount; i++)
                        {
                            end.rowSelection.push_back(-lowEndCount + i);
                        }
                        for(int i = 0; i < highEndCount; i++)
                        {
                            end.rowSelection.push_back(stateArray.rows() + i);
                        }
                    }
                    else
                    {
                        end.rowSelection.push_back((double)(std::get<0>(*currentSites.begin())) / 2.);
                        int lowEndCount = 0, highEndCount = 0;
                        for(const auto& [row, col] : currentSites)
                        {
                            if(col == -1)
                            {
                                lowEndCount++;
                            }
                            else
                            {
                                highEndCount++;
                            }
                        }
                        for(int i = 0; i < lowEndCount; i++)
                        {
                            end.colSelection.push_back(-lowEndCount + i);
                        }
                        for(int i = 0; i < highEndCount; i++)
                        {
                            end.colSelection.push_back(stateArray.cols() + i);
                        }
                    }
                    move.steps.push_back(std::move(end));

                    logger->debug("Time for constructing move: {}us", (double)((std::chrono::steady_clock::now() - startTime).count()) / 1000.);
                    logger->info("Returning removal move that eliminates {} unusable atoms", 
                        move.steps[0].rowSelection.size() * move.steps[0].colSelection.size());
                    return std::tuple(move, bestMoveBenefit / move.cost());
                }
                else
                {
                    logger->warn("No best removal sets found even though there should have been removable atoms");
                    return std::nullopt;    
                }
            }
        }
    }

    logger->info("No better removal move could be found");
    return std::nullopt;
}

bool updateDataStructuresAfterFindingMove(
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> penalizedPathway,
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> pathway,
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> labelledPathway, 
    unsigned int& labelCount, size_t borderRows, size_t borderCols,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry, 
    const ParallelMove &move, std::set<std::tuple<size_t,size_t>>& unusableAtoms, std::set<std::tuple<size_t,size_t>>& usableAtoms,
    std::set<std::tuple<size_t,size_t>>& usableTargetSites,
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    std::shared_ptr<spdlog::logger> logger)
{
    auto occMask = generateMask(Config::getInstance().recommendedDistFromOccSites, 0.5);
    Eigen::Index halfOccRows = occMask.rows() / 2;
    Eigen::Index halfOccCols = occMask.cols() / 2;
    auto emptyMask = generateMask(Config::getInstance().recommendedDistFromEmptySites, 0.5);
    Eigen::Index halfEmptyRows = emptyMask.rows() / 2;
    Eigen::Index halfEmptyCols = emptyMask.cols() / 2;
    auto penalizedOccMask = generateMask(Config::getInstance().minDistFromOccSites, 0.5);
    Eigen::Index halfPOccRows = penalizedOccMask.rows() / 2;
    Eigen::Index halfPOccCols = penalizedOccMask.cols() / 2;

    for(size_t rIndex = 0; rIndex < move.steps[0].rowSelection.size(); rIndex++)
    {
        for(size_t cIndex = 0; cIndex < move.steps[0].colSelection.size(); cIndex++)
        {
            int startRow = move.steps[0].rowSelection[rIndex] + DOUBLE_EQUIVALENCE_THRESHOLD; 
            int startCol = move.steps[0].colSelection[cIndex] + DOUBLE_EQUIVALENCE_THRESHOLD;

            if(move.steps[0].rowSelection[rIndex] + DOUBLE_EQUIVALENCE_THRESHOLD - startRow < 0.25 && 
                move.steps[0].colSelection[cIndex] + DOUBLE_EQUIVALENCE_THRESHOLD - startCol < 0.25 && 
                stateArray(startRow, startCol))
            {
                int newRow = move.steps.back().rowSelection[rIndex] + DOUBLE_EQUIVALENCE_THRESHOLD * 
                    (move.steps.back().rowSelection[rIndex] > 0 ? 1 : -1);
                int newCol = move.steps.back().colSelection[cIndex] + DOUBLE_EQUIVALENCE_THRESHOLD * 
                    (move.steps.back().colSelection[cIndex] > 0 ? 1 : -1);
                int pathwayStartRow = 2 * startRow + borderRows;
                int pathwayStartCol = 2 * startCol + borderCols;

                usableAtoms.erase(std::tuple(startRow, startCol));

                if(unusableAtoms.erase(std::tuple(startRow, startCol)))
                {
                    if(newRow >= 0 && newRow < stateArray.rows() && newCol >= 0 && newCol < stateArray.cols())
                    {
                        unusableAtoms.insert(std::tuple(newRow, newCol));
                    }
                }

                if(newRow >= 0 && newRow < stateArray.rows() && 
                    newCol >= 0 && newCol < stateArray.cols())
                {
                    usableTargetSites.erase(std::tuple(newRow, newCol));
                    pathway(Eigen::seqN(2 * newRow + borderRows - halfOccRows, occMask.rows()), 
                        Eigen::seqN(2 * newCol + borderCols - halfOccCols, occMask.cols())) += occMask.cast<unsigned int>();
                    pathway(Eigen::seqN(2 * newRow + borderRows - halfEmptyRows, emptyMask.rows()), 
                        Eigen::seqN(2 * newCol + borderCols - halfEmptyCols, emptyMask.cols())) -= emptyMask.cast<unsigned int>();
                    penalizedPathway(Eigen::seqN(2 * newRow + borderRows - halfPOccRows, penalizedOccMask.rows()), 
                        Eigen::seqN(2 * newCol + borderCols - halfPOccCols, penalizedOccMask.cols())) += penalizedOccMask.cast<unsigned int>();
                }

                pathway(Eigen::seqN(pathwayStartRow - halfOccRows, occMask.rows()), 
                    Eigen::seqN(pathwayStartCol - halfOccCols, occMask.cols())) -= occMask.cast<unsigned int>();
                pathway(Eigen::seqN(pathwayStartRow - halfEmptyRows, emptyMask.rows()), 
                    Eigen::seqN(pathwayStartCol - halfEmptyCols, emptyMask.cols())) += emptyMask.cast<unsigned int>();
                penalizedPathway(Eigen::seqN(pathwayStartRow - halfPOccRows, penalizedOccMask.rows()), 
                    Eigen::seqN(pathwayStartCol - halfPOccCols, penalizedOccMask.cols())) -= penalizedOccMask.cast<unsigned int>();
            }
        }
    }

    for(Eigen::Index row = 0; row < targetGeometry.rows(); row++)
    {
        for(Eigen::Index col = 0; col < targetGeometry.cols(); col++)
        {
            if(targetGeometry(row,col) && penalizedPathway(borderRows + 2 * (row + compZoneRowStart), 
                borderCols + 2 * (col + compZoneColStart)) == 0)
            {
                usableTargetSites.insert(std::tuple(row + compZoneRowStart, col + compZoneColStart));
            }
        }
    }

    std::tie(labelledPathway, labelCount) = labelPathway(pathway);
    return true;
}

std::tuple<std::set<std::tuple<size_t,size_t>>,std::set<std::tuple<size_t,size_t>>,std::set<std::tuple<size_t,size_t>>> analyzeCurrentStateArray(
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    std::shared_ptr<spdlog::logger> logger)
{
    auto usabilityPreventingNeighborhoodMask = generateMask(Config::getInstance().minDistFromOccSites);
    int usabilityPreventingNeighborhoodMaskRowDist = usabilityPreventingNeighborhoodMask.rows() / 2;
    int usabilityPreventingNeighborhoodMaskColDist = usabilityPreventingNeighborhoodMask.cols() / 2;
    usabilityPreventingNeighborhoodMask(usabilityPreventingNeighborhoodMaskRowDist, usabilityPreventingNeighborhoodMaskColDist) = false;

    std::set<std::tuple<size_t,size_t>> unusableAtoms;
    std::set<std::tuple<size_t,size_t>> usableNonTargetAtoms;
    std::set<std::tuple<size_t,size_t>> usableTargetSites;

    bool changesWereMade = true;
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> unusabilityArray(stateArray.rows(), stateArray.cols());
    unusabilityArray.fill(false);

    while(changesWereMade)
    {
        changesWereMade = false;
        for(Eigen::Index row = 0; row < stateArray.rows(); row++)
        {
            for(Eigen::Index col = 0; col < stateArray.cols(); col++)
            {
                if(stateArray(row,col) && (row < (int)compZoneRowStart || row >= (int)compZoneRowEnd || 
                    col < (int)compZoneColStart || col >= (int)compZoneColEnd || 
                    !targetGeometry(row - compZoneRowStart, col - compZoneColStart) ||
                    unusabilityArray(row,col)))
                {
                    for(int rowShift = -usabilityPreventingNeighborhoodMaskRowDist; 
                        rowShift <= usabilityPreventingNeighborhoodMaskRowDist; rowShift++)
                    {
                        int shiftedRow = row + rowShift;
                        if(shiftedRow >= 0 && shiftedRow < stateArray.rows())
                        {
                            for(int colShift = -usabilityPreventingNeighborhoodMaskColDist; 
                                colShift <= usabilityPreventingNeighborhoodMaskColDist; colShift++)
                            {
                                int shiftedCol = col + colShift;
                                if(shiftedCol >= 0 && shiftedCol < stateArray.cols() && usabilityPreventingNeighborhoodMask(
                                    rowShift + usabilityPreventingNeighborhoodMaskRowDist, 
                                    colShift + usabilityPreventingNeighborhoodMaskColDist))
                                {
                                    changesWereMade |= !unusabilityArray(shiftedRow, shiftedCol);
                                    unusabilityArray(shiftedRow, shiftedCol) = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::stringstream strstream;
    strstream << "Unusable sites: \n";
    for(int row = 0; row < stateArray.rows(); row++)
    {
        for(int col = 0; col < stateArray.cols(); col++)
        {
            if(stateArray(row,col))
            {
                if(unusabilityArray(row, col))
                {
                    unusableAtoms.insert(std::tuple(row,col));
                    strstream << "X";
                }
                else
                {
                    if(row < (int)compZoneRowStart || row >= (int)compZoneRowEnd ||
                        col < (int)compZoneColStart || col >= (int)compZoneColEnd ||
                        !targetGeometry(row - compZoneRowStart, col - compZoneColStart))
                    {
                        usableNonTargetAtoms.insert(std::tuple(row,col));
                    }
                    strstream << "█";
                }
            }
            else
            {
                if(row >= (int)compZoneRowStart && row < (int)compZoneRowEnd && 
                    col >= (int)compZoneColStart && col < (int)compZoneColEnd && 
                    targetGeometry(row - compZoneRowStart, col - compZoneColStart) && 
                    !unusabilityArray(row, col))
                {
                    usableTargetSites.insert(std::tuple(row,col));
                    strstream << "A";
                }
                else
                {
                    strstream << " ";
                }
            }
        }
        strstream << "\n";
    }
    logger->debug(strstream.str());
    return std::tuple(unusableAtoms, usableNonTargetAtoms, usableTargetSites);
}

void designateUsableUndesiredAtomsAsUnusable(const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, std::set<std::tuple<size_t,size_t>>& unusableAtoms,
    std::shared_ptr<spdlog::logger> logger)
{
    auto undesiredAtoms = stateArray(Eigen::seq(compZoneRowStart, compZoneRowEnd - 1),Eigen::seq(compZoneColStart, compZoneColEnd - 1)) && !targetGeometry;
    for(Eigen::Index row = 0; row < targetGeometry.rows(); row++)
    {
        for(Eigen::Index col = 0; col < targetGeometry.cols(); col++)
        {
            if(undesiredAtoms(row,col))
            {
                unusableAtoms.insert(std::tuple(row + compZoneRowStart, col + compZoneColStart));
            }
        }
    }
}

std::vector<ParallelMove> removeAllDirectlyRemovableUnusableAtoms(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    std::set<std::tuple<size_t,size_t>>& unusableAtoms, size_t borderRows, 
    size_t borderCols, std::shared_ptr<spdlog::logger> logger)
{
    std::vector<ParallelMove> removalMoves;

    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> stateArrayCopy = stateArray;
    for(const auto& [r, c] : unusableAtoms)
    {
        stateArrayCopy(r,c) = false;
    }

    auto pathway = generatePathway(borderRows, borderCols, stateArrayCopy, Config::getInstance().minDistFromOccSites, 0);

    for(bool traverseRow : {true,false})
    {
        if((traverseRow && Config::getInstance().minDistFromOccSites <= Config::getInstance().columnSpacing) || 
            (!traverseRow && Config::getInstance().minDistFromOccSites <= Config::getInstance().rowSpacing))
        {
            for(bool lowToHigh : {true,false})
            {
                std::vector<int> minDist;
                std::vector<int> maxDist;
                size_t alongBorderMaxIndex = traverseRow ? stateArray.cols() : stateArray.rows();
                RowBitMask indexDone(alongBorderMaxIndex);
                size_t inwardMax = traverseRow ? stateArray.rows() : stateArray.cols();
                size_t startIndex = lowToHigh ? 0 : (inwardMax - 1);
                int dir = lowToHigh ? 1 : -1;
                for(size_t alongBorder = 0; alongBorder < alongBorderMaxIndex; alongBorder++)
                {
                    int minDistV = -1;
                    bool inBlock = false;
                    for(int inwardDist = 0; inwardDist < (int)inwardMax; inwardDist++)
                    {
                        size_t r = traverseRow ? startIndex + inwardDist * dir : alongBorder;
                        size_t c = traverseRow ? alongBorder : (startIndex + inwardDist * dir);
                        if(unusableAtoms.contains(std::tuple(r,c)))
                        {
                            inBlock = true;
                        }
                        else if(inBlock)
                        {
                            inBlock = false;
                            minDistV = inwardDist - 1;
                        }
                        if(!pathway(2 * r + borderRows, 2 * c + borderCols) == 0 || inwardDist == (int)(inwardMax - 1))
                        {
                            minDist.push_back(minDistV);
                            maxDist.push_back(inwardDist - 1);
                            indexDone.set(alongBorder, minDistV == -1);
                            break;
                        }
                    }
                }
                logger->debug("Indices to move directly: {}/{}", alongBorderMaxIndex - indexDone.bitsSet(), alongBorderMaxIndex);

                while(indexDone.bitsSet() < alongBorderMaxIndex)
                {
                    int maxMinDist = -1;
                    for(size_t i = 0; i < minDist.size(); i++)
                    {
                        if(!indexDone[i] && minDist[i] > maxMinDist)
                        {
                            maxMinDist = minDist[i];
                        }
                    }
                    std::vector<size_t> usedIndices;
                    unsigned int selectedIndices = 0;
                    unsigned int maxIndicesAlongBorder = traverseRow ? Config::getInstance().aodColLimit : Config::getInstance().aodRowLimit;
                    if(Config::getInstance().aodTotalLimit < maxIndicesAlongBorder)
                    {
                        maxIndicesAlongBorder = Config::getInstance().aodTotalLimit;
                    }
                    for(size_t alongBorder = 0; alongBorder < alongBorderMaxIndex && selectedIndices < maxIndicesAlongBorder; alongBorder++)
                    {
                        if(!indexDone[alongBorder] && maxDist[alongBorder] >= maxMinDist)
                        {
                            usedIndices.push_back(alongBorder);
                            indexDone.set(alongBorder, true);
                            selectedIndices++;
                        }
                    }
                    unsigned int maxIndicesInward = traverseRow ? Config::getInstance().aodRowLimit : Config::getInstance().aodColLimit;
                    if(maxIndicesInward * selectedIndices > Config::getInstance().aodTotalLimit)
                    {
                        maxIndicesInward = Config::getInstance().aodTotalLimit / selectedIndices;
                    }
                    std::vector<int> inwardIndicesRequired;
                    for(int inwardDist = 0; inwardDist <= maxMinDist; inwardDist++)
                    {
                        bool stepRequired = false;
                        for(const auto& alongBorder : usedIndices)
                        {
                            if(unusableAtoms.erase(traverseRow ? 
                                std::tuple(startIndex + inwardDist * dir,alongBorder) : 
                                std::tuple(alongBorder, startIndex + inwardDist * dir)))
                            {
                                stepRequired = true;
                            }
                        }
                        if(stepRequired)
                        {
                            inwardIndicesRequired.push_back(startIndex + inwardDist * dir);
                        }
                    }

                    auto first = inwardIndicesRequired.begin();
                    unsigned int moveCount = (unsigned int)ceil((double)inwardIndicesRequired.size() / (double)maxIndicesInward);
                    for(unsigned int moveIndex = 0; moveIndex < moveCount; moveIndex++)
                    {
                        ParallelMove move;
                        ParallelMove::Step start;
                        std::vector<double> *alongBorderSelection;
                        std::vector<double> *inwardSelection;
                        if(traverseRow)
                        {
                            alongBorderSelection = &start.colSelection;
                            inwardSelection = &start.rowSelection;
                        }
                        else
                        {
                            alongBorderSelection = &start.rowSelection;
                            inwardSelection = &start.colSelection;
                        }
                        auto last = ((moveIndex + 1) * maxIndicesInward >= inwardIndicesRequired.size()) ? inwardIndicesRequired.end() : std::next(first, maxIndicesInward);
                        inwardSelection->insert(inwardSelection->end(), first, last);
                        if(!lowToHigh)
                        {
                            std::reverse(inwardSelection->begin(), inwardSelection->end());
                        }
                        alongBorderSelection->insert(alongBorderSelection->end(), usedIndices.begin(), usedIndices.end());
                        move.steps.push_back(std::move(start));

                        ParallelMove::Step end;
                        if(traverseRow)
                        {
                            alongBorderSelection = &end.colSelection;
                            inwardSelection = &end.rowSelection;
                        }
                        else
                        {
                            alongBorderSelection = &end.rowSelection;
                            inwardSelection = &end.colSelection;
                        }
                        int count = ((moveIndex + 1) * maxIndicesInward > inwardIndicesRequired.size()) ? inwardIndicesRequired.size() - moveIndex * maxIndicesInward : maxIndicesInward;
                        logger->debug("Moving {}/{} atoms", count, inwardIndicesRequired.size());
                        for(int i = 0; i < count; i++)
                        {
                            inwardSelection->push_back((int)startIndex - dir * (lowToHigh ? (count - i) : (i + 1)));
                        }
                        alongBorderSelection->insert(alongBorderSelection->end(), usedIndices.begin(), usedIndices.end());
                        move.steps.push_back(std::move(end));

                        removalMoves.push_back(std::move(move));

                        if(moveIndex < moveCount - 1)
                        {
                            first = std::next(first, maxIndicesInward);
                        }
                    }
                }
            }
        }
    }
    return removalMoves;
}

bool findAndExecuteMoves(Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> pathway, 
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> labelledPathway, unsigned int labelCount,
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> penalizedPathway,
    std::set<std::tuple<size_t,size_t>>& unusableAtoms, std::set<std::tuple<size_t,size_t>>& usableAtoms, std::set<std::tuple<size_t,size_t>>& usableTargetSites, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd,
    size_t borderRows, size_t borderCols, std::vector<ParallelMove>& moveList, 
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray,
    const py::EigenDRef<const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry, std::shared_ptr<spdlog::logger> logger)
{
    bool alsoRemoveUsableAtoms = false;
    unsigned int movesSinceLastDirectRemoval = 1;
    std::chrono::duration<double, std::milli> distOneMoveTime = std::chrono::duration<double, std::milli>::zero(), 
        removalMoveTime = std::chrono::duration<double, std::milli>::zero(), 
        pathwayMoveTime = std::chrono::duration<double, std::milli>::zero(), 
        complexMoveTime = std::chrono::duration<double, std::milli>::zero(), 
        executeMoveTime = std::chrono::duration<double, std::milli>::zero(),
        linearMoveTime = std::chrono::duration<double, std::milli>::zero();
    while((stateArray(Eigen::seq(compZoneRowStart, compZoneRowEnd - 1),Eigen::seq(compZoneColStart, compZoneColEnd - 1)) != targetGeometry).any())
    {
        if(CHECK_FOR_DIRECT_REMOVAL_EVERY_X_MOVES > 0 && 
            movesSinceLastDirectRemoval >= CHECK_FOR_DIRECT_REMOVAL_EVERY_X_MOVES)
        {
            movesSinceLastDirectRemoval = 1;
            auto removalMoves = removeAllDirectlyRemovableUnusableAtoms(stateArray, unusableAtoms,
                borderRows, borderCols, logger);
            for(const auto& move : removalMoves)
            {
                if(!updateDataStructuresAfterFindingMove(penalizedPathway, pathway, labelledPathway, labelCount, borderRows, borderCols,
                    stateArray, targetGeometry, move, unusableAtoms, usableAtoms, usableTargetSites, 
                    compZoneRowStart, compZoneRowEnd, compZoneColStart, compZoneColEnd, logger))
                {
                    return false;
                }
                EigenArrayAccessor genericAccessor(stateArray);
                if(!move.execute(genericAccessor, logger))
                {
                    return false;
                }
                moveList.push_back(move);
            }
        }
        else
        {
            movesSinceLastDirectRemoval++;
        }

        logger->debug("Usable atoms: {}, usable target sites: {}" , usableAtoms.size(), usableTargetSites.size());

        std::optional<ParallelMove> bestMove = std::nullopt;
        double bestIntPerCost = -1;

        auto startTime = std::chrono::steady_clock::now();
        auto distOneMove = findDistOneMove(penalizedPathway, stateArray, targetGeometry, unusableAtoms, compZoneRowStart, 
            compZoneRowEnd, compZoneColStart, compZoneColEnd, borderRows, borderCols, logger);
        if(distOneMove.has_value())
        {
            std::tie(bestMove, bestIntPerCost) = distOneMove.value();
        }
        distOneMoveTime += std::chrono::steady_clock::now() - startTime;

        startTime = std::chrono::steady_clock::now();
        auto removalMove = removeAtomsInBorderPathway(stateArray, unusableAtoms, compZoneRowStart, compZoneRowEnd, 
            compZoneColStart, compZoneColEnd, borderRows, borderCols, moveList, logger);
        if(removalMove.has_value() && (!bestMove.has_value() || std::get<1>(removalMove.value()) > bestIntPerCost))
        {
            std::tie(bestMove, bestIntPerCost) = removalMove.value();
        }
        removalMoveTime += std::chrono::steady_clock::now() - startTime;

        startTime = std::chrono::steady_clock::now();
        auto pathwayMove = findPathwayMove(penalizedPathway, pathway, labelledPathway, stateArray, targetGeometry, unusableAtoms, usableAtoms, usableTargetSites,
            compZoneRowStart, compZoneRowEnd, compZoneColStart, compZoneColEnd, borderRows, borderCols, bestIntPerCost, logger);
        if(pathwayMove.has_value() && (!bestMove.has_value() || std::get<1>(pathwayMove.value()) > bestIntPerCost))
        {
            std::tie(bestMove, bestIntPerCost) = pathwayMove.value();
        }
        pathwayMoveTime += std::chrono::steady_clock::now() - startTime;

        startTime = std::chrono::steady_clock::now();
        if(bestMove.has_value())
        {
            if(!updateDataStructuresAfterFindingMove(penalizedPathway, pathway, labelledPathway, labelCount, borderRows, borderCols,
                stateArray, targetGeometry, bestMove.value(), unusableAtoms, usableAtoms, usableTargetSites, 
                compZoneRowStart, compZoneRowEnd, compZoneColStart, compZoneColEnd, logger))
            {
                return false;
            }
            EigenArrayAccessor genericAccessor(stateArray);
            if(!bestMove.value().execute(genericAccessor, logger))
            {
                return false;
            }
            moveList.push_back(std::move(bestMove.value()));
        }
        else
        {
            if(!alsoRemoveUsableAtoms)
            {
                alsoRemoveUsableAtoms = true;
                designateUsableUndesiredAtomsAsUnusable(stateArray, targetGeometry, compZoneRowStart, 
                    compZoneRowEnd, compZoneColStart, compZoneColEnd, unusableAtoms, logger);
            }
            else
            {
                logger->error("No more moves could be found!");
                return false;
            }
        }
        executeMoveTime += std::chrono::steady_clock::now() - startTime;
    }

    logger->debug("Time spent on distOne move: {}ms", distOneMoveTime.count());
    logger->debug("Time spent on removal move: {}ms", removalMoveTime.count());
    logger->debug("Time spent on pathway move: {}ms", pathwayMoveTime.count());
    logger->debug("Time spent on complex move: {}ms", complexMoveTime.count());
    logger->debug("Time spent on linear move: {}ms", linearMoveTime.count());
    logger->debug("Time spent on move execution: {}ms", executeMoveTime.count());
    return true;
}

bool checkTargetGeometryFeasibility(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry, 
    std::shared_ptr<spdlog::logger> logger)
{
    auto usabilityPreventingNeighborhoodMask = generateMask(Config::getInstance().minDistFromOccSites);
    int usabilityPreventingNeighborhoodMaskRowDist = usabilityPreventingNeighborhoodMask.rows() / 2;
    int usabilityPreventingNeighborhoodMaskColDist = usabilityPreventingNeighborhoodMask.cols() / 2;
    usabilityPreventingNeighborhoodMask(usabilityPreventingNeighborhoodMaskRowDist, usabilityPreventingNeighborhoodMaskColDist) = false;

    for(Eigen::Index row = 0; row < targetGeometry.rows(); row++)
    {
        for(Eigen::Index col = 0; col < targetGeometry.cols(); col++)
        {
            if(targetGeometry(row,col))
            {
                for(int rowShift = -usabilityPreventingNeighborhoodMaskRowDist; 
                    rowShift <= usabilityPreventingNeighborhoodMaskRowDist; rowShift++)
                {
                    int shiftedRow = row + rowShift;
                    if(shiftedRow >= 0 && shiftedRow < targetGeometry.rows())
                    {
                        for(int colShift = -usabilityPreventingNeighborhoodMaskColDist; 
                            colShift <= usabilityPreventingNeighborhoodMaskColDist; colShift++)
                        {
                            int shiftedCol = col + colShift;
                            if(shiftedCol >= 0 && shiftedCol < targetGeometry.cols() && usabilityPreventingNeighborhoodMask(
                                rowShift + usabilityPreventingNeighborhoodMaskRowDist, 
                                colShift + usabilityPreventingNeighborhoodMaskColDist))
                            {
                                if(targetGeometry(shiftedRow,shiftedCol))
                                {
                                    logger->error("Target atoms too close together to be sorted");
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    size_t borderRows = Config::getInstance().minDistFromOccSites / (Config::getInstance().rowSpacing / 2.) + 2;
    size_t borderCols = Config::getInstance().minDistFromOccSites / (Config::getInstance().columnSpacing / 2.) + 2;
    auto pathway = generatePathway(borderRows, borderCols, targetGeometry, Config::getInstance().minDistFromOccSites, 0);
    auto [labelledPathway, labelCount] = labelPathway(pathway);

    auto occMask = generateMask(Config::getInstance().minDistFromOccSites, 0.5);
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> localAccessibility(occMask.rows() + 2, occMask.cols() + 2);
    for(Eigen::Index r = 0; r < targetGeometry.rows(); r++)
    {
        for(Eigen::Index c = 0; c < targetGeometry.cols(); c++)
        {
            if(targetGeometry(r,c))
            {
                bool directConnectionToPathwayExists = false;
                bool changesWereMade = true;
                localAccessibility.fill(false);
                localAccessibility(localAccessibility.rows() / 2, localAccessibility.cols() / 2) = true;
                while(changesWereMade && !directConnectionToPathwayExists)
                {
                    changesWereMade = false;
                    for(Eigen::Index lRow = 0; lRow < localAccessibility.rows() && !directConnectionToPathwayExists; lRow++)
                    {
                        for(Eigen::Index lCol = 0; lCol < localAccessibility.cols() && !directConnectionToPathwayExists; lCol++)
                        {
                            if(localAccessibility(lRow, lCol))
                            {
                                if(labelledPathway(2 * r + borderRows - localAccessibility.rows() / 2 + lRow,
                                    2 * c + borderCols - localAccessibility.cols() / 2 + lCol) == 1)
                                {
                                    directConnectionToPathwayExists = true;
                                }
                            }
                            else
                            {
                                bool withinOwnAtomsProhibitedArea = lRow > 0 && lRow < localAccessibility.rows() - 1 && 
                                    lCol > 0 && lCol < localAccessibility.cols() - 1 && occMask(lRow - 1, lCol - 1);
                                bool notInOtherPathway = pathway(2 * r + borderRows - localAccessibility.rows() / 2 + lRow,
                                    2 * c + borderCols - localAccessibility.cols() / 2 + lCol) <= (withinOwnAtomsProhibitedArea ? 1 : 0);
                                bool adjacentToAccessibleLoc = 
                                    (lRow > 0 && localAccessibility(lRow - 1, lCol)) || 
                                    (lRow < localAccessibility.rows() - 1 && localAccessibility(lRow + 1, lCol)) || 
                                    (lCol > 0 && localAccessibility(lRow, lCol - 1)) || 
                                    (lCol < localAccessibility.cols() - 1 && localAccessibility(lRow, lCol + 1));
                                if(notInOtherPathway && adjacentToAccessibleLoc)
                                {
                                    localAccessibility(lRow, lCol) = true;
                                    changesWereMade = true;
                                }
                            }
                        }
                    }
                }
                
                if(!directConnectionToPathwayExists)
                {
                    logger->error("No connection to pathway for ({}/{})", r, c);
                    return false;
                }
            }
        }
    }
    std::stringstream strstream;
    strstream << "Target: \n";
    for(size_t r = 0; r < (size_t)targetGeometry.rows(); r++)
    {
        for(size_t c = 0; c < (size_t)targetGeometry.cols(); c++)
        {
            strstream << (targetGeometry(r,c) ? "X" : " ");
        }
        strstream << "\n";
    }
    strstream << "Pathway: \n";
    for(Eigen::Index r = 0; r < pathway.rows(); r++)
    {
        for(Eigen::Index c = 0; c < pathway.cols(); c++)
        {
            strstream << (pathway(r, c) == 0 ? " " : "█");
        }
        strstream << "\n";
    }
    logger->info(strstream.str());

    return true;
}

ParallelMove createRemovalMove(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    bool verticalMove, std::vector<double> unchangingIndices, std::vector<double> movedIndices, bool moveTowardsLowIndices)
{
    ParallelMove move;
    ParallelMove::Step start;
    ParallelMove::Step end;
    if(verticalMove)
    {
        start.colSelection = unchangingIndices;
        end.colSelection = unchangingIndices;
        start.rowSelection = movedIndices;
        for(size_t outsideDist = 0; outsideDist < movedIndices.size(); outsideDist++)
        {
            if(moveTowardsLowIndices)
            {
                end.rowSelection.push_back(-(double)(movedIndices.size() - outsideDist));
            }
            else
            {
                end.rowSelection.push_back(stateArray.rows() + outsideDist);
            }
        }
    }
    else
    {
        start.rowSelection = unchangingIndices;
        end.rowSelection = unchangingIndices;
        start.colSelection = movedIndices;
        for(size_t outsideDist = 0; outsideDist < movedIndices.size(); outsideDist++)
        {
            if(moveTowardsLowIndices)
            {
                end.colSelection.push_back(-(double)(movedIndices.size() - outsideDist));
            }
            else
            {
                end.colSelection.push_back(stateArray.cols() + outsideDist);
            }
        }
    }
    move.steps.push_back(std::move(start));
    move.steps.push_back(std::move(end));

    return move;
}

bool createMinimallyInvasiveAccessPathway(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray,
    std::set<std::tuple<size_t,size_t>>& unusableAtoms, std::set<std::tuple<size_t,size_t>>& usableAtoms, 
    size_t borderRows, size_t borderCols, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd,
    std::vector<ParallelMove>& moveList, unsigned int count, std::shared_ptr<spdlog::logger> logger)
{
    unsigned int aodRowLimit = Config::getInstance().aodRowLimit;
    unsigned int aodColLimit = Config::getInstance().aodColLimit;
    unsigned int aodTotalLimit = Config::getInstance().aodTotalLimit;

    bool pathwayVertical = Config::getInstance().columnSpacing > Config::getInstance().rowSpacing;
    double spacing = pathwayVertical ? Config::getInstance().columnSpacing : Config::getInstance().rowSpacing;
    if(2 * Config::getInstance().minDistFromOccSites < spacing)
    {
        return true;
    }
    unsigned int width = ceil((double)(2 * Config::getInstance().minDistFromOccSites) / spacing) - 1;
    size_t compZoneWidth = pathwayVertical ? (compZoneColEnd - compZoneColStart) : (compZoneRowEnd - compZoneRowStart);
    if(width > compZoneWidth)
    {
        logger->error("Width of outside pathway exceeds size of computation zone");
        return false;
    }
    std::vector<unsigned int> usableAtomsPerIndexLowIndexSide;
    std::vector<unsigned int> usableAtomsPerIndexHighIndexSide;
    RowBitMask lowSideContainsAtoms(compZoneWidth);
    RowBitMask highSideContainsAtoms(compZoneWidth);
    for(size_t alongBorderIndex = 0; alongBorderIndex < compZoneWidth; alongBorderIndex++)
    {
        unsigned int removedUsableAtoms = 0;
        for(size_t inwardIndex = 0; inwardIndex < (pathwayVertical ? compZoneRowStart : compZoneColStart); inwardIndex++)
        {
            size_t row = pathwayVertical ? inwardIndex : (alongBorderIndex + compZoneRowStart);
            size_t col = pathwayVertical ? (alongBorderIndex + compZoneColStart) : inwardIndex;
            if(stateArray(row, col))
            {
                lowSideContainsAtoms.set(alongBorderIndex, true);
                if(!unusableAtoms.contains(std::tuple(row, col)))
                {
                    removedUsableAtoms++;
                }
            }
        }
        usableAtomsPerIndexLowIndexSide.push_back(removedUsableAtoms);
        removedUsableAtoms = 0;
        for(size_t inwardIndex = pathwayVertical ? (stateArray.rows() - 1) : (stateArray.cols() - 1); 
            inwardIndex >= (pathwayVertical ? compZoneRowEnd : compZoneColEnd); inwardIndex--)
        {
            size_t row = pathwayVertical ? inwardIndex : (alongBorderIndex + compZoneRowStart);
            size_t col = pathwayVertical ? (alongBorderIndex + compZoneColStart) : inwardIndex;
            if(stateArray(row, col))
            {
                highSideContainsAtoms.set(alongBorderIndex, true);
                if(!unusableAtoms.contains(std::tuple(row, col)))
                {
                    removedUsableAtoms++;
                }
            }
        }
        usableAtomsPerIndexHighIndexSide.push_back(removedUsableAtoms);
    }

    std::vector<std::tuple<size_t,unsigned int>> bestIndicesLow;
    std::vector<std::tuple<size_t,unsigned int>> bestIndicesHigh;
    for(size_t alongBorderIndex = 0; alongBorderIndex < compZoneWidth - width; alongBorderIndex++)
    {
        unsigned int totalRemovedUsableAtoms = 0;
        for(size_t i = 0; i < width; i++)
        {
            totalRemovedUsableAtoms += usableAtomsPerIndexLowIndexSide[alongBorderIndex + i];
        }
        std::tuple<size_t,unsigned int> newElemLow(alongBorderIndex, totalRemovedUsableAtoms);
        bestIndicesLow.insert(std::upper_bound(bestIndicesLow.begin(), bestIndicesLow.end(), newElemLow, 
            [](const auto& lhs, const auto& rhs) { return std::get<1>(lhs) < std::get<1>(rhs); }), newElemLow);

        totalRemovedUsableAtoms = 0;
        for(size_t i = 0; i < width; i++)
        {
            totalRemovedUsableAtoms += usableAtomsPerIndexHighIndexSide[alongBorderIndex + i];
        }
        std::tuple<size_t,unsigned int> newElemHigh(alongBorderIndex, totalRemovedUsableAtoms);
        bestIndicesHigh.insert(std::upper_bound(bestIndicesHigh.begin(), bestIndicesHigh.end(), newElemHigh, 
            [](const auto& lhs, const auto& rhs) { return std::get<1>(lhs) < std::get<1>(rhs); }), newElemHigh);
    }

    std::set<size_t> usedIndicesLow;
    std::set<size_t> usedIndicesHigh;
    for(unsigned int index = 0; index < count; index++)
    {
        std::optional<unsigned int> lowestRemovedUnusableAtoms = std::nullopt;
        std::optional<size_t> bestNewIndex = std::nullopt;
        std::optional<int> bestMinDist = std::nullopt;
        unsigned int removedAtoms = 0;

        std::vector<std::tuple<size_t,unsigned int>> *bestIndices;
        std::set<size_t> *usedIndices;

        if((usedIndicesLow.size() <= usedIndicesHigh.size() || bestIndicesHigh.size() <= usedIndicesHigh.size()) && 
            bestIndicesLow.size() > usedIndicesLow.size())
        {
            bestIndices = &bestIndicesLow;
            usedIndices = &usedIndicesLow;
        }
        else if(bestIndicesHigh.size() > usedIndicesHigh.size())
        {
            bestIndices = &bestIndicesHigh;
            usedIndices = &usedIndicesHigh;
        }
        else
        {
            logger->error("Not enough indices to provide required number of pathways to outside");
            return false;
        }
        for(const auto& [index, removedUsableAtoms] : *bestIndices)
        {
            if(!usedIndices->contains(index))
            {
                if(!lowestRemovedUnusableAtoms.has_value())
                {
                    lowestRemovedUnusableAtoms = removedUsableAtoms;
                }
                if(!bestNewIndex.has_value())
                {
                    bestNewIndex = index;
                    removedAtoms = removedUsableAtoms;
                    for(const auto& alreadySelectedIndex : *usedIndices)
                    {
                        int dist = abs((int)alreadySelectedIndex - (int)index);
                        if(!bestMinDist.has_value() || dist < bestMinDist.value())
                        {
                            bestMinDist = dist;
                        }
                    }
                    if(usedIndices->empty())
                    {
                        break;
                    }
                }
                else
                {
                    std::optional<int> localMinDist = std::nullopt;
                    for(const auto& alreadySelectedIndex : *usedIndices)
                    {
                        int dist = abs((int)alreadySelectedIndex - (int)index);
                        if(!localMinDist.has_value() || dist < localMinDist.value())
                        {
                            localMinDist = dist;
                        }
                    }
                    if(bestMinDist.has_value() && localMinDist.has_value() && localMinDist.value() > bestMinDist.value())
                    {
                        bestNewIndex = index;
                        removedAtoms = removedUsableAtoms;
                        bestMinDist = localMinDist;
                    }
                }
            }
            if(lowestRemovedUnusableAtoms.has_value() && bestNewIndex.has_value() && 
                removedUsableAtoms > lowestRemovedUnusableAtoms.value() + 2)
            {
                break;
            }
        }

        if(bestNewIndex.has_value())
        {
            logger->info("Sacrificing {} useful atoms to create pathway of width {} from compZone to outside start at index {}", 
                removedAtoms, width, bestNewIndex.value());
            usedIndices->insert(bestNewIndex.value());
        }
        else
        {
            logger->error("No index for creating pathway to outside could be determined");
            return false;
        }
    }

    while(!usedIndicesLow.empty() || !usedIndicesHigh.empty())
    {
        bool atLowIndex;
        size_t newIndex;
        if(!usedIndicesLow.empty())
        {
            atLowIndex = true;
            newIndex = *usedIndicesLow.begin();
            usedIndicesLow.erase(newIndex);
        }
        else
        {
            atLowIndex = false;
            newIndex = *usedIndicesHigh.begin();
            usedIndicesHigh.erase(newIndex);
        }

        unsigned int requiredIndicesAlongBorder = 0;
        for(size_t withinWidth = 0; withinWidth < width; withinWidth++)
        {
            if((atLowIndex && lowSideContainsAtoms[newIndex + withinWidth]) || 
                (!atLowIndex && highSideContainsAtoms[newIndex + withinWidth]))
            {
                requiredIndicesAlongBorder++;
            }
        }
        unsigned int maxAlongBorderIndices = pathwayVertical ? aodColLimit : aodRowLimit;
        if(aodTotalLimit < maxAlongBorderIndices)
        {
            maxAlongBorderIndices = aodTotalLimit;
        }
        if(requiredIndicesAlongBorder == 0)
        {
            continue;
        }

        size_t alongBorderIndex = newIndex;
        size_t alongBorderSegments = ceil((double)requiredIndicesAlongBorder / (double)maxAlongBorderIndices - DOUBLE_EQUIVALENCE_THRESHOLD);
        for(size_t alongBorderSeg = 0; alongBorderSeg < alongBorderSegments; alongBorderSeg++)
        {
            size_t usedIndicesAlongBorder = requiredIndicesAlongBorder / alongBorderSegments + 
                ((alongBorderSeg < requiredIndicesAlongBorder % alongBorderSegments) ? 1 : 0);
            unsigned int maxInwardIndices = pathwayVertical ? aodRowLimit : aodColLimit;
            if(usedIndicesAlongBorder * maxInwardIndices > aodTotalLimit)
            {
                maxInwardIndices = aodTotalLimit / usedIndicesAlongBorder;
            }
            std::vector<double> alongBorderIndices;
            for(; alongBorderIndices.size() < usedIndicesAlongBorder && 
                alongBorderIndex < newIndex + width; alongBorderIndex++)
            {
                if((atLowIndex && lowSideContainsAtoms[alongBorderIndex]) || 
                    (!atLowIndex && highSideContainsAtoms[alongBorderIndex]))
                {
                    alongBorderIndices.push_back(alongBorderIndex + (pathwayVertical ? compZoneColStart : compZoneRowStart));
                }
            }
            size_t inwardIndex;
            size_t stopIndex;
            if(atLowIndex)
            {
                inwardIndex = 0;
                if(pathwayVertical)
                {
                    stopIndex = compZoneRowStart;
                }
                else
                {
                    stopIndex = compZoneColStart;
                }
            }
            else
            {
                if(pathwayVertical)
                {
                    inwardIndex = stateArray.rows() - 1;
                    stopIndex = compZoneRowEnd - 1;
                }
                else
                {
                    inwardIndex = stateArray.cols() - 1;
                    stopIndex = compZoneColEnd - 1;
                }
            }
            std::vector<double> inwardIndices;
            for(; inwardIndex != stopIndex; inwardIndex += (atLowIndex ? 1 : -1))
            {
                bool indexRequired = false;
                for(const auto& usedIndex : alongBorderIndices)
                {
                    size_t usedIndexSt = floor(usedIndex + DOUBLE_EQUIVALENCE_THRESHOLD);
                    if(stateArray((pathwayVertical ? inwardIndex : usedIndexSt), (pathwayVertical ? usedIndexSt : inwardIndex)))
                    {
                        indexRequired = true;
                        auto location = pathwayVertical ? std::tuple(inwardIndex, usedIndexSt) : 
                            std::tuple(usedIndexSt, inwardIndex);
                        unusableAtoms.erase(location);
                        usableAtoms.erase(location);
                        logger->debug("Removing {}/{} from usable and unusable atom list", 
                            std::get<0>(location), std::get<1>(location));
                    }
                }
                if(indexRequired)
                {
                    if(atLowIndex)
                    {
                        inwardIndices.push_back(inwardIndex);
                    }
                    else
                    {
                        inwardIndices.insert(inwardIndices.begin(), inwardIndex);
                    }
                    if(inwardIndices.size() == maxInwardIndices)
                    {
                        logger->debug("Creating removal move with {} unchanging and {} changing indices", 
                            alongBorderIndices.size(), inwardIndices.size());
                        auto move = createRemovalMove(stateArray, pathwayVertical, alongBorderIndices, inwardIndices, atLowIndex);
                        inwardIndices.clear();

                        EigenArrayAccessor genericAccessor(stateArray);
                        move.execute(genericAccessor, logger);
                        moveList.push_back(std::move(move));
                    }
                }
            }
            if(!inwardIndices.empty())
            {
                logger->debug("Creating removal move with {} unchanging and {} changing indices", 
                    alongBorderIndices.size(), inwardIndices.size());
                auto move = createRemovalMove(stateArray, pathwayVertical, alongBorderIndices, inwardIndices, atLowIndex);

                EigenArrayAccessor genericAccessor(stateArray);
                move.execute(genericAccessor, logger);
                moveList.push_back(std::move(move));
            }
        }
    }
    return true;
}

bool sortLatticeGreedyMain(py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry, 
    std::vector<ParallelMove>& moveList, std::shared_ptr<spdlog::logger> logger)
{
    size_t borderRows = Config::getInstance().recommendedDistFromOccSites / (Config::getInstance().rowSpacing / 2.) + 1;
    size_t borderCols = Config::getInstance().recommendedDistFromOccSites / (Config::getInstance().columnSpacing / 2.) + 1;

    auto pathway = generatePathway(borderRows, borderCols, stateArray);
    auto [labelledPathway, labelCount] = labelPathway(pathway);

    std::stringstream pathwayStream;
    pathwayStream << "Labelled pathway: \n";
    for(Eigen::Index r = 0; r < labelledPathway.rows(); r++)
    {
        for(Eigen::Index c = 0; c < labelledPathway.cols(); c++)
        {
            pathwayStream << (labelledPathway(r, c) != 0 ? " " : "█");
        }
        pathwayStream << "\n";
    }
    logger->info(pathwayStream.str());

    auto [unusableAtoms, usableAtoms, usableTargetSites] = analyzeCurrentStateArray(stateArray, 
        targetGeometry, compZoneRowStart, compZoneRowEnd, compZoneColStart, compZoneColEnd, logger);

    auto removalMoves = removeAllDirectlyRemovableUnusableAtoms(stateArray, unusableAtoms,
        borderRows, borderCols, logger);
    for(const auto& move : removalMoves)
    {
        EigenArrayAccessor genericAccessor(stateArray);
        if(!move.execute(genericAccessor, logger))
        {
            return false;
        }
        moveList.push_back(move);
    }
    if(!createMinimallyInvasiveAccessPathway(stateArray, unusableAtoms, usableAtoms, borderRows, borderCols, 
        compZoneRowStart, compZoneRowEnd, compZoneColStart, compZoneColEnd, moveList, 10, logger))
    {
        return false;
    }

    auto penalizedPathway = generatePathway(borderRows, borderCols, stateArray, Config::getInstance().minDistFromOccSites, 0);

    std::stringstream ppathwayStream;
    ppathwayStream << "Penalized pathway: \n";
    for(Eigen::Index r = 0; r < penalizedPathway.rows(); r++)
    {
        for(Eigen::Index c = 0; c < penalizedPathway.cols(); c++)
        {
            ppathwayStream << (penalizedPathway(r, c) == 0 ? " " : "█");
        }
        ppathwayStream << "\n";
    }
    logger->info(ppathwayStream.str());

    pathway = generatePathway(borderRows, borderCols, stateArray);
    std::tie(labelledPathway, labelCount) = labelPathway(pathway);

    if(!findAndExecuteMoves(pathway, labelledPathway, labelCount, penalizedPathway, unusableAtoms, usableAtoms, usableTargetSites, compZoneRowStart, 
        compZoneRowEnd, compZoneColStart, compZoneColEnd, borderRows, borderCols, moveList,
        stateArray, targetGeometry, logger))
    {
        return false;
    }

    std::stringstream strstream;
    strstream << "Remaining usable target sites: \n";
    for(const auto& [row,col] : usableTargetSites)
    {
        strstream << row << "/" << col << "\n";
    }
    strstream << "Remaining unusable atoms: \n";
    for(const auto& [row,col] : unusableAtoms)
    {
        strstream << row << "/" << col << "\n";
    }
    strstream << "Remaining usable atoms: \n";
    for(const auto& [row,col] : usableAtoms)
    {
        strstream << row << "/" << col << "\n";
    }

    strstream << "PathwayState: \n";
    for(Eigen::Index r = 0; r < labelledPathway.rows(); r++)
    {
        for(Eigen::Index c = 0; c < labelledPathway.cols(); c++)
        {
            strstream << (labelledPathway(r, c) != 0 ? " " : "█");
        }
        strstream << "\n";
    }
    logger->info(strstream.str());

    if(Config::getInstance().alwaysGenerateAllAODTones)
    {
        for(auto& move : moveList)
        {
            move.extendToUseAllTones(stateArray.rows(), stateArray.cols(), logger, false);
        }
    }

    return true;
}

std::optional<std::vector<ParallelMove>> sortLatticeGreedyInternal(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry)
{
    std::shared_ptr<spdlog::logger> logger;
    Config& config = Config::getInstance();
    if((logger = spdlog::get(config.parallelLoggerName)) == nullptr)
    {
        logger = spdlog::basic_logger_mt(config.parallelLoggerName, config.logFileName);
    }
    logger->set_level(spdlog::level::debug);

    if(!(compZoneRowEnd - compZoneRowStart == (size_t)targetGeometry.rows()))
    {
        logger->error("Comp zone does not have same number of rows as target geometry, aborting");
        return std::nullopt;
    }
    if(!(compZoneColEnd - compZoneColStart == (size_t)targetGeometry.cols()))
    {
        logger->error("Comp zone does not have same number of cols as target geometry, aborting");
        return std::nullopt;
    }

    std::stringstream strstream;
    strstream << "Initial state: \n";
    for(size_t r = 0; r < (size_t)stateArray.rows(); r++)
    {
        for(size_t c = 0; c < (size_t)stateArray.cols(); c++)
        {
            if(r >= compZoneRowStart && r < compZoneRowEnd && c >= compZoneColStart && c < compZoneColEnd)
            {
                if(targetGeometry(r - compZoneRowStart, c - compZoneColStart))
                {
                    strstream << (stateArray(r,c) ? "█" : "▒");
                }
                else
                {
                    strstream << (stateArray(r,c) ? "X" : " ");
                }
            }
            else
            {
                strstream << (stateArray(r,c) ? "█" : " ");
            }
        }
        strstream << "\n";
    }
    logger->info(strstream.str());

    std::vector<ParallelMove> moveList;
    if(!checkTargetGeometryFeasibility(targetGeometry, logger))
    {
        return std::nullopt;
    }
    if(!sortLatticeGreedyMain(stateArray, compZoneRowStart, compZoneRowEnd, 
        compZoneColStart, compZoneColEnd, targetGeometry, moveList, logger))
    {
        return std::nullopt;
    }

    std::stringstream endstrstream;
    endstrstream << "Final state: \n";
    for(size_t r = 0; r < (size_t)stateArray.rows(); r++)
    {
        for(size_t c = 0; c < (size_t)stateArray.cols(); c++)
        {
            if(r >= compZoneRowStart && r < compZoneRowEnd && c >= compZoneColStart && c < compZoneColEnd)
            {
                if(targetGeometry(r - compZoneRowStart, c - compZoneColStart))
                {
                    endstrstream << (stateArray(r,c) != 0 ? "█" : "▒");
                }
                else
                {
                    endstrstream << (stateArray(r,c) != 0 ? "X" : " ");
                }
            }
            else
            {
                endstrstream << (stateArray(r,c) != 0 ? "█" : " ");
            }
        }
        endstrstream << "\n";
    }
    logger->info(endstrstream.str());

    return moveList;
}

std::optional<std::vector<ParallelMove>> sortLatticeGreedyParallel(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry)
{
    std::shared_ptr<spdlog::logger> logger = Config::getInstance().getGreedyLatticeLogger();

    if(!compZoneRowEnd - compZoneRowStart == (size_t)targetGeometry.rows())
    {
        logger->error("Comp zone does not have same number of rows as target geometry, aborting");
        return std::nullopt;
    }
    if(!compZoneColEnd - compZoneColStart == (size_t)targetGeometry.cols())
    {
        logger->error("Comp zone does not have same number of cols as target geometry, aborting");
        return std::nullopt;
    }

    std::stringstream strstream;
    strstream << "Initial state: \n";
    for(size_t r = 0; r < (size_t)stateArray.rows(); r++)
    {
        for(size_t c = 0; c < (size_t)stateArray.cols(); c++)
        {
            if(r >= compZoneRowStart && r < compZoneRowEnd && c >= compZoneColStart && c < compZoneColEnd)
            {
                if(targetGeometry(r - compZoneRowStart, c - compZoneColStart))
                {
                    strstream << (stateArray(r,c) ? "█" : "▒");
                }
                else
                {
                    strstream << (stateArray(r,c) ? "X" : " ");
                }
            }
            else
            {
                strstream << (stateArray(r,c) ? "█" : " ");
            }
        }
        strstream << "\n";
    }
    logger->info(strstream.str());

    std::vector<ParallelMove> moveList;
    if(!checkTargetGeometryFeasibility(targetGeometry, logger))
    {
        return std::nullopt;
    }
    if(!sortLatticeGreedyMain(stateArray, compZoneRowStart, compZoneRowEnd, 
        compZoneColStart, compZoneColEnd, targetGeometry, moveList, logger))
    {
        return std::nullopt;
    }

    std::stringstream endstrstream;
    endstrstream << "Final state: \n";
    for(size_t r = 0; r < (size_t)stateArray.rows(); r++)
    {
        for(size_t c = 0; c < (size_t)stateArray.cols(); c++)
        {
            if(r >= compZoneRowStart && r < compZoneRowEnd && c >= compZoneColStart && c < compZoneColEnd)
            {
                if(targetGeometry(r - compZoneRowStart, c - compZoneColStart))
                {
                    endstrstream << (stateArray(r,c) != 0 ? "█" : "▒");
                }
                else
                {
                    endstrstream << (stateArray(r,c) != 0 ? "X" : " ");
                }
            }
            else
            {
                endstrstream << (stateArray(r,c) != 0 ? "█" : " ");
            }
        }
        endstrstream << "\n";
    }
    logger->info(endstrstream.str());

    return moveList;
}