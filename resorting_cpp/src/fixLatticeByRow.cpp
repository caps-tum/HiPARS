#include "sortParallel.hpp"
#include "sortLattice.hpp"

#include <algorithm>
#include <cfloat>
#include <fstream>
#include <map>
#include <sstream>
#include <set>
#include <chrono>
#include <ranges>

#include "config.hpp"

bool createRemovalMoveRetracing(ArrayAccessor& stateArray, std::vector<ParallelMove>& moveList, int currentRow, int currentCol,
    Eigen::Ref<Eigen::Array<int,Eigen::Dynamic,Eigen::Dynamic>> pathway, size_t borderRows, size_t borderCols,
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> distancePathway, 
    int minRowDist, int minColDist, std::shared_ptr<spdlog::logger> logger)
{
    ParallelMove move;
    int currentDir = 0;
    bool findingPath = true;
    unsigned int dist = distancePathway(currentRow, currentCol);
    while(dist > 0 && findingPath)
    {
        for(int dirOffset = 0; dirOffset < 4; dirOffset++)
        {
            int dir = (currentDir + dirOffset) % 4;
            int newRow = currentRow + ((dir % 2 == 0) ? dir - 1 : 0);
            int newCol = currentCol + ((dir % 2 == 1) ? dir - 2 : 0);
            if(distancePathway(newRow, newCol) < dist)
            {
                dist = distancePathway(newRow, newCol);
                if(move.steps.empty())
                {
                    ParallelMove::Step step;
                    step.rowSelection.push_back((double)(currentRow - borderRows) / 2.);
                    step.colSelection.push_back((double)(currentCol - borderCols) / 2.);
                    move.steps.push_back(std::move(step));
                }
                else if(dirOffset != 0)
                {
                    ParallelMove::Step step;
                    step.rowSelection.push_back((double)(currentRow - borderRows) / 2.);
                    step.colSelection.push_back((double)(currentCol - borderCols) / 2.);
                    move.steps.push_back(std::move(step));
                }
                currentRow = newRow;
                currentCol = newCol;
                currentDir = dir;
                break;
            }
            if(dirOffset == 3)
            {
                logger->error("Path could not be retraced while fixing sorting deficiency. Aborting.");
                return -1;
            }
        }
    }
    logger->info("Successfully removed atom at {}/{} to {}/{}", 
        (currentRow - borderRows) / 2, (currentCol - borderCols) / 2, move.steps[0].rowSelection[0], move.steps[0].colSelection[0]);
    ParallelMove::Step end;
    end.rowSelection.push_back((currentRow - borderRows) / 2);
    end.colSelection.push_back((currentCol - borderCols) / 2);
    move.steps.push_back(std::move(end));
    move.execute(stateArray, logger);
    moveList.push_back(std::move(move));
    (pathway < 0).select(0, pathway);
    return true;
}

void removeAtoms(ArrayAccessor& stateArray, std::set<std::tuple<size_t,size_t>>& atomsToBeRemoved, std::vector<ParallelMove>& moveList, MinimalArrayInformation& arrayInfo,
    Eigen::Ref<Eigen::Array<int,Eigen::Dynamic,Eigen::Dynamic>> pathway, size_t borderRows, size_t borderCols,
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> distancePathway, std::shared_ptr<spdlog::logger> logger)
{
    std::vector<std::tuple<size_t,size_t>> coordsToSetDist1, coordsToSetDist2;
    std::vector<std::tuple<size_t,size_t>> *coordsToSetDist = &coordsToSetDist1, *coordsToSetDistNext = &coordsToSetDist2;

    distancePathway.setConstant(UINT_MAX);

    int firstNormalIndexXC, firstNormalIndexAC, lastNormalIndexXC;
    if(arrayInfo.vertical)
    {
        firstNormalIndexXC = arrayInfo.normalIndices.firstColOrAC;
        firstNormalIndexAC = arrayInfo.normalIndices.firstRowOrXC;
        lastNormalIndexXC = arrayInfo.normalIndices.lastColOrACExcl;
    }
    else
    {
        firstNormalIndexXC = arrayInfo.normalIndices.firstRowOrXC;
        firstNormalIndexAC = arrayInfo.normalIndices.firstColOrAC;
        lastNormalIndexXC = arrayInfo.normalIndices.lastRowOrXCExcl;
    }
    std::optional<int> targetIndicesAC[2] = {std::nullopt, std::nullopt};
    for(auto& dumpingIndex : arrayInfo.dumpingIndicesAC)
    {
        if(dumpingIndex < firstNormalIndexAC)
        {
            targetIndicesAC[0] = dumpingIndex;
        }
        else
        {
            targetIndicesAC[1] = dumpingIndex;
        }
    }
    for(const auto& targetIndexAC : targetIndicesAC)
    {
        if(targetIndexAC.has_value())
        {
            for(int indexXC = firstNormalIndexXC; indexXC < lastNormalIndexXC; indexXC++)
            {
                int row, col;
                if(arrayInfo.vertical)
                {
                    row = targetIndexAC.value();
                    col = indexXC;
                }
                else
                {
                    row = indexXC;
                    col = targetIndexAC.value();
                }
                coordsToSetDist->push_back(std::tuple(borderRows + 2 * row, borderCols + 2 * col));
                distancePathway(borderRows + 2 * row, borderCols + 2 * col) = 0;
            }
        }
    }
    unsigned int dist = 1;

    int minRowDist = ceil(Config::getInstance().minDistFromOccSites / Config::getInstance().rowSpacing);
    int minColDist = ceil(Config::getInstance().minDistFromOccSites / Config::getInstance().columnSpacing);

    while(!atomsToBeRemoved.empty() && !coordsToSetDist->empty())
    {
        for(auto [startRow, startCol] : *coordsToSetDist)
        {
            for(int dir = 0; dir < 4; dir++)
            {
                int newRow = startRow + ((dir % 2 == 0) ? dir - 1 : 0);
                int newCol = startCol + ((dir % 2 == 1) ? dir - 2 : 0);
                bool isInStateBorders = newRow >= (int)borderRows && newRow < (int)(pathway.rows() - borderRows) && 
                    newCol >= (int)borderCols && newCol < (int)(pathway.cols() - borderCols);
                bool inNormalIndexRangeOrGoingInwards = 
                    ((newRow >= (int)(borderRows + 2 * arrayInfo.normalIndices.firstRowOrXC) && newRow < (int)(borderRows + 2 * arrayInfo.normalIndices.lastRowOrXCExcl)) || 
                    (newRow < (int)(borderRows + 2 * arrayInfo.normalIndices.firstRowOrXC) && dir == 2) ||
                    (newRow >= (int)(borderRows + 2 * arrayInfo.normalIndices.lastRowOrXCExcl) && dir == 0)) && 
                    ((newCol >= (int)(borderCols + 2 * arrayInfo.normalIndices.firstColOrAC) && newCol < (int)(borderCols + 2 * arrayInfo.normalIndices.lastColOrACExcl)) || 
                    (newCol < (int)(borderCols + 2 * arrayInfo.normalIndices.firstColOrAC) && dir == 3) ||
                    (newCol >= (int)(borderCols + 2 * arrayInfo.normalIndices.lastColOrACExcl) && dir == 1));
                bool isAtomSite = isInStateBorders && (newRow - borderRows) % 2 == 0 && (newCol - borderCols) % 2 == 0;
                logger->debug("Going from {}/{} to {}/{}, dist: {}, inStateBorders: {}, inNormalRangeOrGoingThere: {}, pathway: {}, distancePathway: {}, isAtomSite: {}, stateArray: {}",
                    startRow, startCol, newRow, newCol, dist, isInStateBorders, inNormalIndexRangeOrGoingInwards, pathway(newRow, newCol), distancePathway(newRow, newCol), 
                    isAtomSite, isAtomSite ? stateArray((newRow - borderRows) / 2, (newCol - borderCols) / 2) : -1);
                if(isInStateBorders && inNormalIndexRangeOrGoingInwards && pathway(newRow, newCol) <= 0 && dist < distancePathway(newRow, newCol))
                {
                    if(isAtomSite && stateArray((newRow - borderRows) / 2, (newCol - borderCols) / 2))
                    {
                        if(atomsToBeRemoved.contains(std::tuple((newRow - borderRows) / 2, (newCol - borderCols) / 2)))
                        {
                            distancePathway(newRow, newCol) = dist;
                            coordsToSetDistNext->push_back(std::tuple(newRow, newCol));

                            // Found target site to move atom to, retracing steps
                            if(createRemovalMoveRetracing(stateArray, moveList, newRow, newCol, pathway, 
                                borderRows, borderCols, distancePathway, minRowDist, minColDist, logger))
                            {
                                atomsToBeRemoved.erase(std::tuple((newRow - borderRows) / 2, (newCol - borderCols) / 2));
                            }
                            else
                            {
                                logger->warn("Could not remove atom at {}/{}", (newRow - borderRows) / 2, (newCol - borderCols) / 2);
                            }
                        }
                    }
                    else
                    {
                        distancePathway(newRow, newCol) = dist;
                        coordsToSetDistNext->push_back(std::tuple(newRow, newCol));
                    }
                }
            }
        }
        coordsToSetDist->clear();
        auto tmp = coordsToSetDist;
        coordsToSetDist = coordsToSetDistNext;
        coordsToSetDistNext = tmp; 
        dist++;
    }
    return;
}

int removeSuperfluousAtoms(ArrayAccessor& stateArray, pybind11::detail::unchecked_reference<TargetState, 2>& targetGeometry, MinimalArrayInformation& arrayInfo, 
    std::vector<ParallelMove>& moveList, Eigen::Ref<Eigen::Array<int,Eigen::Dynamic,Eigen::Dynamic>> pathway, size_t borderRows, size_t borderCols,
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> distancePathway,
    Eigen::Ref<Eigen::Array<int,Eigen::Dynamic,Eigen::Dynamic>> occMask, std::shared_ptr<spdlog::logger> logger)
{
    Eigen::Array<int,Eigen::Dynamic,Eigen::Dynamic> blockingMask = generateMask(Config::getInstance().minDistFromOccSites).cast<int>();

    Eigen::Index halfBlockingRows = blockingMask.rows() / 2;
    Eigen::Index halfBlockingCols = blockingMask.cols() / 2;
    blockingMask(halfBlockingRows, halfBlockingCols) = 0;

    std::vector<std::tuple<size_t,size_t>> primaryAtomsToBeRemoved;
    std::set<std::tuple<size_t,size_t>> allAtomsConsideredForRemoval, allAtomsToBeRemoved;
    for(size_t row = arrayInfo.normalIndices.firstRowOrXC; row < arrayInfo.normalIndices.lastRowOrXCExcl; row++)
    {
        for(size_t col = arrayInfo.normalIndices.firstColOrAC; col < arrayInfo.normalIndices.lastColOrACExcl; col++)
        {
            if(targetGeometry(row, col) == TargetState::EMPTY && stateArray(row, col))
            {
                // Remove superfluous atom
                primaryAtomsToBeRemoved.push_back(std::tuple(row,col));
            }
        }
    }

    Eigen::Index halfOccRows = occMask.rows() / 2;
    Eigen::Index halfOccCols = occMask.cols() / 2;
    while(!primaryAtomsToBeRemoved.empty())
    {
        auto [row, col] = primaryAtomsToBeRemoved.back();
        primaryAtomsToBeRemoved.pop_back();
        allAtomsConsideredForRemoval.insert(std::tuple(row, col));
        if(targetGeometry(row, col) != TargetState::IRRELEVANT)
        {
            allAtomsToBeRemoved.insert(std::tuple(row, col));
        }

        // Already remove pathway limitations around atoms-to-be-removed, as they might otherwise block each other
        pathway(Eigen::seqN(2 * row + borderRows - halfOccRows, occMask.rows()), 
            Eigen::seqN(2 * col + borderCols - halfOccCols, occMask.cols())) -= occMask;

        for(Eigen::Index blockingRow = 0; blockingRow < blockingMask.rows(); blockingRow++)
        {
            for(Eigen::Index blockingCol = 0; blockingCol < blockingMask.cols(); blockingCol++)
            {
                int otherRow = (int)row + blockingRow - halfBlockingRows;
                int otherCol = (int)col + blockingCol - halfBlockingCols;
                if(blockingMask(blockingRow, blockingCol) && isInCompZone(otherRow, otherCol, 
                    arrayInfo.normalIndices.firstRowOrXC, arrayInfo.normalIndices.lastRowOrXCExcl, arrayInfo.normalIndices.firstColOrAC, arrayInfo.normalIndices.lastColOrACExcl) &&
                    stateArray(otherRow, otherCol) && !allAtomsConsideredForRemoval.contains(std::tuple(otherRow, otherCol)))
                {
                    primaryAtomsToBeRemoved.push_back(std::tuple(otherRow, otherCol));
                }
            }
        }
    }

    size_t totalAtomsToBeRemoved = allAtomsToBeRemoved.size();
    removeAtoms(stateArray, allAtomsToBeRemoved, moveList, arrayInfo, pathway, borderRows, borderCols, distancePathway, logger);

    return allAtomsToBeRemoved.size();
}

int fixVacancies(ArrayAccessor& stateArray, pybind11::detail::unchecked_reference<TargetState, 2>& targetGeometry, 
    std::vector<ParallelMove>& moveList, MinimalArrayInformation arrayInfo,
    Eigen::Ref<Eigen::Array<int,Eigen::Dynamic,Eigen::Dynamic>> pathway, size_t borderRows, size_t borderCols,
    Eigen::Ref<Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>> distancePathway,
    Eigen::Ref<Eigen::Array<int,Eigen::Dynamic,Eigen::Dynamic>> occMask, std::shared_ptr<spdlog::logger> logger)
{
    Eigen::Index halfOccRows = occMask.rows() / 2;
    Eigen::Index halfOccCols = occMask.cols() / 2;

    int problemsRemaining = 0;

    for(size_t row = arrayInfo.normalIndices.firstRowOrXC; row < arrayInfo.normalIndices.lastRowOrXCExcl; row++)
    {
        for(size_t col = arrayInfo.normalIndices.firstColOrAC; col < arrayInfo.normalIndices.lastColOrACExcl; col++)
        {
            if(targetGeometry(row, col) == TargetState::OCCUPIED && !stateArray(row, col))
            {
                std::vector<std::tuple<size_t,size_t>> coordsToSetDist1, coordsToSetDist2;
                std::vector<std::tuple<size_t,size_t>> *coordsToSetDist = &coordsToSetDist1, *coordsToSetDistNext = &coordsToSetDist2;
                coordsToSetDist->push_back(std::tuple(borderRows + 2 * row, borderCols + 2 * col));
                unsigned int dist = 1;
                distancePathway.setConstant(UINT_MAX);
                distancePathway(borderRows + 2 * row, borderCols + 2 * col) = 0;
                std::optional<std::tuple<size_t,size_t>> sourceAtom = std::nullopt;
                while(!sourceAtom.has_value() && !coordsToSetDist->empty())
                {
                    for(auto [startRow, startCol] : *coordsToSetDist)
                    {
                        for(int dir = 0; dir < 4; dir++)
                        {
                            int newRow = startRow + ((dir % 2 == 0) ? dir - 1 : 0);
                            int newCol = startCol + ((dir % 2 == 1) ? dir - 2 : 0);
                            if(newRow >= 0 && newRow < pathway.rows() && newCol >= 0 && newCol < pathway.cols() &&
                                pathway(newRow, newCol) <= 0 && dist < distancePathway(newRow, newCol))
                            {
                                distancePathway(newRow, newCol) = dist;
                                coordsToSetDistNext->push_back(std::tuple(newRow, newCol));
                                if(newRow > (int)borderRows && newRow < (int)(pathway.rows() - borderRows) && 
                                    newCol > (int)borderCols && newCol < (int)(pathway.cols() - borderCols) && 
                                    (newRow - borderRows) % 2 == 0 && (newCol - borderCols) % 2 == 0)
                                {
                                    int trapRow = (newRow - borderRows) / 2;
                                    int trapCol = (newCol - borderCols) / 2;
                                    if(targetGeometry(trapRow, trapCol) != TargetState::OCCUPIED && stateArray(trapRow, trapCol))
                                    {
                                        sourceAtom = std::tuple(newRow, newCol);
                                        break;
                                    }
                                }
                            }
                        }
                        if(sourceAtom.has_value())
                        {
                            break;
                        }
                    }
                    coordsToSetDist->clear();
                    auto tmp = coordsToSetDist;
                    coordsToSetDist = coordsToSetDistNext;
                    coordsToSetDistNext = tmp; 
                    dist++;
                }
                if(sourceAtom.has_value())
                {
                    // Found atom to fix vacancy with, retracing steps
                    ParallelMove move;
                    int currentDir = 0;
                    int currentRow = std::get<0>(sourceAtom.value());
                    int currentCol = std::get<1>(sourceAtom.value());
                    logger->info("Missing atom at {}/{} fixed with atom from {}/{}", 
                        row, col, (currentRow - borderRows) / 2, (currentCol - borderCols) / 2);
                    bool findingPath = true;

                    while(dist > 0 && findingPath)
                    {
                        for(int dirOffset = 0; dirOffset < 4; dirOffset++)
                        {
                            int dir = (currentDir + dirOffset) % 4;
                            int newRow = currentRow + ((dir % 2 == 0) ? dir - 1 : 0);
                            int newCol = currentCol + ((dir % 2 == 1) ? dir - 2 : 0);
                            if(distancePathway(newRow, newCol) < dist)
                            {
                                dist = distancePathway(newRow, newCol);
                                if(dirOffset != 0 || move.steps.empty())
                                {
                                    ParallelMove::Step step;
                                    step.rowSelection.push_back((double)(currentRow - borderRows) / 2.);
                                    step.colSelection.push_back((double)(currentCol - borderCols) / 2.);
                                    move.steps.push_back(std::move(step));
                                }
                                currentRow = newRow;
                                currentCol = newCol;
                                currentDir = dir;
                                break;
                            }
                            if(dirOffset == 3)
                            {
                                logger->error("Path could not be retraced while fixing sorting deficiency. Aborting.");
                                return -1;
                            }
                        }
                    }
                    ParallelMove::Step end;
                    end.rowSelection.push_back((currentRow - borderRows) / 2);
                    end.colSelection.push_back((currentCol - borderCols) / 2);
                    move.steps.push_back(std::move(end));
                    move.execute(stateArray, logger);
                    pathway(Eigen::seqN(currentRow - halfOccRows, occMask.rows()), 
                        Eigen::seqN(currentCol - halfOccCols, occMask.cols())) += occMask;
                    pathway(Eigen::seqN(std::get<0>(sourceAtom.value()) - halfOccRows, occMask.rows()), 
                        Eigen::seqN(std::get<1>(sourceAtom.value()) - halfOccCols, occMask.cols())) -= occMask;
                    (pathway < 0).select(0, pathway);
                    moveList.push_back(std::move(move));
                }
                else
                {
                    logger->warn("Missing atom at {}/{} could not be fixed", row, col);
                    problemsRemaining++;
                }
            }
            else if(targetGeometry(row, col) == TargetState::EMPTY && stateArray(row, col))
            {
                problemsRemaining++;
            }
        }
    }

    return problemsRemaining;
}

// Gather general information about the array and save in struct for later use
std::vector<std::vector<int>> findUsableAtomsByRow(ArrayAccessor& stateArray, pybind11::detail::unchecked_reference<TargetState, 2>& targetGeometry, 
    MinimalArrayInformation& arrayInfo, std::shared_ptr<spdlog::logger> logger)
{
    std::vector<std::vector<int>> usableAtomsByRow;
    usableAtomsByRow.resize(stateArray.rows());

    // Create mask where true existing atoms would prevent an atoms usability
    auto usabilityPreventingNeighborhoodMask = generateMask(Config::getInstance().minDistFromOccSites);
    int usabilityPreventingNeighborhoodMaskRowDist = usabilityPreventingNeighborhoodMask.rows() / 2;
    int usabilityPreventingNeighborhoodMaskColDist = usabilityPreventingNeighborhoodMask.cols() / 2;
    usabilityPreventingNeighborhoodMask(usabilityPreventingNeighborhoodMaskRowDist, usabilityPreventingNeighborhoodMaskColDist) = false;

    // Iterate over array, check for usability-preventing neighbors, and sort into structure accordingly
    for(size_t row = arrayInfo.normalIndices.firstRowOrXC; row < arrayInfo.normalIndices.lastRowOrXCExcl && row < stateArray.rows(); row++)
    {
        for(size_t col = arrayInfo.normalIndices.firstColOrAC; col < arrayInfo.normalIndices.lastColOrACExcl && col < stateArray.cols(); col++)
        {
            if(stateArray(row,col))
            {
                bool usable = true;
                for(int rowShift = -usabilityPreventingNeighborhoodMaskRowDist; 
                    usable && rowShift <= usabilityPreventingNeighborhoodMaskRowDist; rowShift++)
                {
                    int shiftedRow = (int)row + rowShift;
                    if(shiftedRow >= 0 && shiftedRow < (int)stateArray.rows())
                    {
                        for(int colShift = -usabilityPreventingNeighborhoodMaskColDist; 
                            colShift <= usabilityPreventingNeighborhoodMaskColDist; colShift++)
                        {
                            int shiftedCol = (int)col + colShift;
                            if(shiftedCol >= 0 && shiftedCol < (int)stateArray.cols() && usabilityPreventingNeighborhoodMask(
                                rowShift + usabilityPreventingNeighborhoodMaskRowDist, 
                                colShift + usabilityPreventingNeighborhoodMaskColDist) && 
                                stateArray(shiftedRow, shiftedCol))
                            {
                                usable = false;
                                break;
                            }
                        }
                    }
                }
                if(usable)
                {
                    usableAtomsByRow[row].push_back(col);
                }
            }
        }
    }

    return usableAtomsByRow;
}

std::optional<std::vector<ParallelMove>> fixLatticeByRowSortingDeficiencies(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    const py::array_t<TargetState>& targetGeometry, MinimalArrayInformation& arrayInfo)
{
    auto startTime = std::chrono::steady_clock::now();
    // Init logger
    std::shared_ptr<spdlog::logger> logger = Config::getInstance().getLatticeByRowLogger();

    auto targetGeometryUnchecked = targetGeometry.unchecked<2>();

    if(targetGeometry.shape()[0] != stateArray.rows())
    {
        logger->error("Target geometry does not have same number of rows as state array, aborting");
        return std::nullopt;
    }
    if(targetGeometry.shape()[1] != stateArray.cols())
    {
        logger->error("Target geometry does not have same number of cols as state array, aborting");
        return std::nullopt;
    }

    PyEigenArrayAccessor stateArrayAccessor(stateArray);

    std::vector<ParallelMove> moveList;

    auto usableAtomsByRow = findUsableAtomsByRow(stateArrayAccessor, targetGeometryUnchecked, arrayInfo, logger);

    auto duration = std::chrono::steady_clock::now() - startTime;
    startTime = std::chrono::steady_clock::now();
    logger->debug("Time after finding usable atoms: {}ns", duration.count());

    Eigen::Array<int,Eigen::Dynamic,Eigen::Dynamic> occMask = 
        generateMask(Config::getInstance().minDistFromOccSites, 0.5).cast<int>();
    int rowEndDist = ceil((double)Config::getInstance().minDistFromOccSites / Config::getInstance().rowSpacing);
    int colEndDist = ceil((double)Config::getInstance().minDistFromOccSites / Config::getInstance().columnSpacing);
    size_t borderRows = Config::getInstance().minDistFromOccSites / (Config::getInstance().rowSpacing / 2);
    size_t borderCols = Config::getInstance().minDistFromOccSites / (Config::getInstance().columnSpacing / 2);
    Eigen::Array<int,Eigen::Dynamic,Eigen::Dynamic> pathway = 
        generatePathway(borderRows, borderCols, stateArrayAccessor, Config::getInstance().minDistFromOccSites, 0,
            arrayInfo.normalIndices.firstRowOrXC, arrayInfo.normalIndices.lastRowOrXCExcl,
            arrayInfo.normalIndices.firstColOrAC, arrayInfo.normalIndices.lastColOrACExcl).cast<int>();
    auto distancePathway = Eigen::Array<unsigned int, Eigen::Dynamic, Eigen::Dynamic>(pathway.rows(), pathway.cols());

    if(logger->level() <= spdlog::level::debug)
    {
        std::stringstream pathwayStream;
        pathwayStream << "Pathway: \n";
        for(Eigen::Index r = 0; r < pathway.rows(); r++)
        {
            for(Eigen::Index c = 0; c < pathway.cols(); c++)
            {
                pathwayStream << pathway(r, c);
            }
            pathwayStream << "\n";
        }
        logger->debug(pathwayStream.str());

        duration = std::chrono::steady_clock::now() - startTime;
        startTime = std::chrono::steady_clock::now();
        logger->debug("Time for finding pathways: {}ns", duration.count());
    }

    for(int row = 0; row < stateArray.rows(); row++)
    {
        for(size_t col : usableAtomsByRow[row])
        {
            if(targetGeometryUnchecked(row, col) != TargetState::OCCUPIED)
            {
                for(int dir = 0; dir < 4; dir++)
                {
                    bool extractionAllowed = true;
                    int rowDir = (dir % 2 == 0) ? dir - 1 : 0;
                    int colDir = (dir % 2 == 1) ? dir - 2 : 0;
                    int endDist = (dir % 2 == 0) ? rowEndDist : colEndDist;
                    int endRow = row + rowDir * endDist;
                    int endCol = col + colDir * endDist;
                    if(endRow >= 0 && endRow < stateArray.rows() && endCol >= 0 && endCol < stateArray.cols())
                    {
                        for(int dist = 0; dist < endDist; dist++)
                        {
                            if(pathway(borderRows + (row + rowDir * dist) * 2, borderCols + (col + colDir * dist) * 2) > 1)
                            {
                                extractionAllowed = false;
                                break;
                            }
                        }
                        if(!extractionAllowed || pathway(borderRows + endRow * 2, borderCols + endCol * 2) > 0)
                        {
                            continue;
                        }
                        else
                        {
                            for(int dist = 0; dist < 2 * endDist; dist++)
                            {
                                pathway(borderRows + 2 * row + rowDir * dist, borderCols + 2 * col + colDir * dist) = 0;
                            }
                        }
                    }
                }
            }
        }
    }

    duration = std::chrono::steady_clock::now() - startTime;
    startTime = std::chrono::steady_clock::now();
    logger->debug("Time for opening pathways: {}ns", duration.count());

    auto remainingProblems = fixVacancies(stateArrayAccessor, targetGeometryUnchecked, 
        moveList, arrayInfo, pathway, borderRows, borderCols, distancePathway, occMask, logger);
    duration = std::chrono::steady_clock::now() - startTime;
    startTime = std::chrono::steady_clock::now();
    logger->debug("Time for initial vacancy fixing: {}ns", duration.count());

    if(remainingProblems > 0)
    {
        int previouslyRemainingSuperfluousAtoms = INT_MAX;
        int remainingSuperfluousAtoms = removeSuperfluousAtoms(stateArrayAccessor, targetGeometryUnchecked, 
            arrayInfo, moveList, pathway, borderRows, borderCols, distancePathway, occMask, logger);
        duration = std::chrono::steady_clock::now() - startTime;
        startTime = std::chrono::steady_clock::now();
        logger->debug("Time for removing superfluous atoms: {}ns", duration.count());
        while(remainingSuperfluousAtoms > 0 && remainingSuperfluousAtoms < previouslyRemainingSuperfluousAtoms)
        {
            previouslyRemainingSuperfluousAtoms = remainingSuperfluousAtoms;
            remainingSuperfluousAtoms = removeSuperfluousAtoms(stateArrayAccessor, targetGeometryUnchecked, 
                arrayInfo, moveList, pathway, borderRows, borderCols, distancePathway, occMask, logger);
            duration = std::chrono::steady_clock::now() - startTime;
            startTime = std::chrono::steady_clock::now();
            logger->debug("Time for removing superfluous atoms: {}ns", duration.count());
        }
        if(remainingSuperfluousAtoms > 0)
        {
            logger->error("Superfluous atom in the computational zone could not be removed after multiple removal rounds. Aborting");
            return std::nullopt;
        }
        remainingProblems = fixVacancies(stateArrayAccessor, targetGeometryUnchecked, 
            moveList, arrayInfo, pathway, borderRows, borderCols, distancePathway, occMask, logger);
        if(remainingProblems != 0)
        {
            return std::nullopt;
        }
        duration = std::chrono::steady_clock::now() - startTime;
        startTime = std::chrono::steady_clock::now();
        logger->debug("Time for fixing remaining vacancies: {}ns", duration.count());
    }
    else if(remainingProblems < 0)
    {
        return std::nullopt;
    }

    if(Config::getInstance().alwaysGenerateAllAODTones)
    {
        for(auto& move : moveList)
        {
            move.extendToUseAllTones(stateArray.rows(), stateArray.cols(), logger, false);
        }
    }

    duration = std::chrono::steady_clock::now() - startTime;
    startTime = std::chrono::steady_clock::now();
    logger->debug("Time until end of fixing function: {}ns", duration.count());
    return moveList;
}