/***
 * Sorting approach that imposes a fixed movement scheme instead of finding the best move in a greedy fashion
 * Idea by Francisco Romão and Jonas Winklmann, Implemented by Jonas Winklmann
 */

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

// Provides access to array data while possibly flipping indices, useful due to independence on dimension
bool& accessStateArrayDimIndepedent(ArrayAccessor& stateArray, 
    Eigen::Index indexXC, Eigen::Index indexAC, bool vertical)
{
    if(vertical)
    {
        return stateArray(indexAC, indexXC);
    }
    else
    {
        return stateArray(indexXC, indexAC);
    }
}

// Returns inclusive start and exclusive end index for sorting channel
std::optional<std::pair<int,int>> determineBestStartPosition(ArrayAccessor& stateArray, std::vector<ParallelMove>& moveList, const ArrayInformation& arrayInfo)
{
    unsigned int indicesToClear = (arrayInfo.sortingChannelWidth + 1) * 2;

    std::optional<int> bestLastIndexBeforeChannel = std::nullopt;
    std::optional<double> bestDist = std::nullopt;

    // Only consider starting from the middle if movement parallelization even makes sense
    if(((arrayInfo.vertical && Config::getInstance().aodColLimit > 1 && 
        Config::getInstance().aodRowLimit < Config::getInstance().aodTotalLimit) || 
        (!arrayInfo.vertical && Config::getInstance().aodRowLimit > 1 && 
        Config::getInstance().aodColLimit < Config::getInstance().aodTotalLimit)) && 
        arrayInfo.arraySizeXC > indicesToClear &&
        arrayInfo.spacingXC > Config::getInstance().minAodSpacing)
    {
        unsigned int usableBeforeChannel = 0;
        unsigned int usableAfterChannel = std::accumulate(
            arrayInfo.usableAtomsPerXCIndex.begin() + indicesToClear, arrayInfo.usableAtomsPerXCIndex.end(), 0u, 
            [](unsigned int init, const auto& elem) { return init + elem.size(); });

        unsigned int requiredBeforeChannel = std::accumulate(
            arrayInfo.targetSitesPerXCIndex.begin(), arrayInfo.targetSitesPerXCIndex.begin() + arrayInfo.sortingChannelWidth + 1, 0u, 
            [](unsigned int init, const auto& elem) { return init + elem.size(); });
        unsigned int requiredAfterChannel = std::accumulate(
            arrayInfo.targetSitesPerXCIndex.begin() + arrayInfo.sortingChannelWidth + 1, arrayInfo.targetSitesPerXCIndex.end(), 0u, 
            [](unsigned int init, const auto& elem) { return init + elem.size(); });

        // Iterate over possible starting positions and use the one that has enough atoms on both sides while being as centered as possible
        for(size_t lastIndexBeforeChannel = 0; lastIndexBeforeChannel < arrayInfo.arraySizeXC - indicesToClear - 1; lastIndexBeforeChannel++)
        {
            // Update number of existing and required atoms
            usableBeforeChannel += arrayInfo.usableAtomsPerXCIndex[lastIndexBeforeChannel].size();
            usableAfterChannel -= arrayInfo.usableAtomsPerXCIndex[lastIndexBeforeChannel + indicesToClear].size();
            requiredBeforeChannel += arrayInfo.targetSitesPerXCIndex[lastIndexBeforeChannel + arrayInfo.sortingChannelWidth + 1].size();
            requiredAfterChannel -= arrayInfo.targetSitesPerXCIndex[lastIndexBeforeChannel + arrayInfo.sortingChannelWidth + 1].size();

            // Check if atoms numbers suffice. If so, use it if it is the closest index to the center yet
            // This could be done more efficiently
            double dist = abs((double)(arrayInfo.lastNormalIndexXCExcl - arrayInfo.firstNormalIndexXC) / 2. - 
                ((double)(lastIndexBeforeChannel + arrayInfo.sortingChannelWidth) + 1.5));
            if(usableBeforeChannel >= requiredBeforeChannel && usableAfterChannel >= requiredAfterChannel && 
                (!bestLastIndexBeforeChannel.has_value() || !bestDist.has_value() || dist < bestDist.value()))
            {
                bestLastIndexBeforeChannel = lastIndexBeforeChannel;
                bestDist = dist;
            }
        }
    }

    if(bestLastIndexBeforeChannel.has_value())
    {
        if(bestLastIndexBeforeChannel.value() + 2 * arrayInfo.sortingChannelWidth + 3 > (int)arrayInfo.arraySizeXC)
        {
            return std::pair(bestLastIndexBeforeChannel.value() + 1, arrayInfo.arraySizeXC);
        }
        else
        {
            return std::pair(bestLastIndexBeforeChannel.value() + 1, bestLastIndexBeforeChannel.value() + 2 * arrayInfo.sortingChannelWidth + 3);
        }
    }
    else
    {
        // If no index was found before, take the first one that has sufficiently many atoms on both sides
        int totalTargetSites = std::accumulate(arrayInfo.targetSitesPerXCIndex.begin(), arrayInfo.targetSitesPerXCIndex.end(), 0u, 
            [](unsigned int init, const auto& elem) { return init + elem.size(); });
        int excessStartingLowIndex = (int)std::accumulate(
            arrayInfo.usableAtomsPerXCIndex.begin() + arrayInfo.sortingChannelWidth + 1, arrayInfo.usableAtomsPerXCIndex.end(), 0u, 
            [](unsigned int init, const auto& elem) { return init + elem.size(); }) - totalTargetSites;
        int excessStartingHighIndex = (int)std::accumulate(
            arrayInfo.usableAtomsPerXCIndex.begin(), arrayInfo.usableAtomsPerXCIndex.end() - arrayInfo.sortingChannelWidth - 1, 0u, 
            [](unsigned int init, const auto& elem) { return init + elem.size(); }) - totalTargetSites;
        if(excessStartingLowIndex >= 0)
        {
            return std::pair(arrayInfo.firstNormalIndexXC, arrayInfo.firstNormalIndexXC + arrayInfo.sortingChannelWidth + 1);
        }
        else if(excessStartingHighIndex >= 0)
        {
            return std::pair(arrayInfo.lastNormalIndexXCExcl - arrayInfo.sortingChannelWidth - 1, arrayInfo.lastNormalIndexXCExcl);
        }
        else
        {
            return std::nullopt;
        }
    }
}

std::pair<std::vector<int>,std::vector<int>> findChannelIndicesXC(std::vector<int>& remainingIndices, int startIndex, int endIndexExcl,
    ArrayInformation& arrayInfo, std::shared_ptr<spdlog::logger> logger)
{
    std::vector<int> startSelectionXC, endSelectionXC;
    int minExternalSpacing = ceil(Config::getInstance().minDistFromOccSites / arrayInfo.spacingXC);
    int minInternalSpacing = ceil(Config::getInstance().minAodSpacing / arrayInfo.spacingXC);
    if(minInternalSpacing > minExternalSpacing)
    {
        minExternalSpacing = minInternalSpacing;
    }
    if(minExternalSpacing < 1)
    {
        minExternalSpacing = 1;
    }
    int firstExtractionIndexXC = startIndex, lastExtractionIndexXCExcl = endIndexExcl;
    if(startIndex > 0)
    {
        firstExtractionIndexXC += (minExternalSpacing - 1);
    }
    if(endIndexExcl < (int)arrayInfo.arraySizeXC)
    {
        lastExtractionIndexXCExcl -= (minExternalSpacing - 1);
    }

    unsigned int count = arrayInfo.maxTonesXC;
    if(count * arrayInfo.maxTonesAC > Config::getInstance().aodTotalLimit)
    {
        count = Config::getInstance().aodTotalLimit / arrayInfo.maxTonesAC;
    }
    if(minInternalSpacing > 0 && (lastExtractionIndexXCExcl - firstExtractionIndexXC) / minExternalSpacing < (int)count)
    {
        count = (lastExtractionIndexXCExcl - firstExtractionIndexXC) / minExternalSpacing;
    }
    if(remainingIndices.size() < count)
    {
        count = remainingIndices.size();
    }
    if(Config::getInstance().minAodSpacing > arrayInfo.spacingAC)
    {
        // If min aod spacing exceeds ac spacing, clear one by one, otherwise indices mostly have to be cleared one by one
        count = 1;
    }

    std::vector<int> directAllowedIndices, directNotAllowedIndices;
    for(int i : remainingIndices)
    {
        bool directMoveAllowed = true;
        if(startIndex > 0 && i < startIndex + (minExternalSpacing - 1))
        {
            directMoveAllowed = false;
        }
        if(endIndexExcl < (int)arrayInfo.arraySizeXC && i > endIndexExcl - minExternalSpacing)
        {
            directMoveAllowed = false;
        }
        if(directMoveAllowed)
        {
            directAllowedIndices.push_back(i);
        }
        else
        {
            directNotAllowedIndices.push_back(i);
        }
    }
    if(directAllowedIndices.size() >= count)
    {
        for(unsigned int i = 0; i < count; i++)
        {
            startSelectionXC.push_back(directAllowedIndices[i]);
            std::erase(remainingIndices, directAllowedIndices[i]);
            endSelectionXC.push_back(firstExtractionIndexXC + i * minInternalSpacing);
        }
    }
    else
    {
        int firstUsedIndex = (remainingIndices.size() - count) / 2;
        std::vector<int>::iterator nextStartIndex = remainingIndices.begin() + firstUsedIndex;
        for(unsigned int i = 0; i < count; i++)
        {
            startSelectionXC.push_back(*nextStartIndex);
            nextStartIndex = remainingIndices.erase(nextStartIndex);
            endSelectionXC.push_back(firstExtractionIndexXC + i * minInternalSpacing);
        }
    }
    return std::pair(std::move(startSelectionXC), std::move(endSelectionXC));
}

// Remove all atoms from initial sorting channel
void clearChannel(ArrayAccessor& stateArray, int startIndex, int endIndexExcl, bool removeAroundElbow, std::vector<ParallelMove>& moveList, 
    ArrayInformation& arrayInfo, std::shared_ptr<spdlog::logger> logger)
{
    // Remove atoms at one side, atoms can just be moved away directly
    std::vector<int> indicesXC;
    
    // Remove atoms from registers
    for(int i = startIndex; i < endIndexExcl; i++)
    {
        arrayInfo.usableAtomsPerXCIndex[i].clear();
        arrayInfo.unusableAtomsPerXCIndex[i].clear();
        indicesXC.push_back(i);
    }

    while(!indicesXC.empty())
    {
        auto [startSelectionXC, endSelectionXC] = findChannelIndicesXC(indicesXC, startIndex, endIndexExcl, arrayInfo, logger);

        int nextIndexToDealWithLow = arrayInfo.firstNormalIndexAC, nextIndexToDealWithHigh = arrayInfo.lastNormalIndexACExcl - 1;
        while(nextIndexToDealWithLow < nextIndexToDealWithHigh)
        {
            // Create move to remove atoms from sorting channel
            ParallelMove move;
            ParallelMove::Step start, end;
            std::vector<double> *startSelectionAC, *endSelectionAC;
            if(arrayInfo.vertical)
            {
                start.colSelection = std::vector<double>(startSelectionXC.begin(), startSelectionXC.end());
                end.colSelection = std::vector<double>(endSelectionXC.begin(), endSelectionXC.end());
                startSelectionAC = &start.rowSelection;
                endSelectionAC = &end.rowSelection;
            }
            else
            {
                start.rowSelection = std::vector<double>(startSelectionXC.begin(), startSelectionXC.end());
                end.rowSelection = std::vector<double>(endSelectionXC.begin(), endSelectionXC.end());
                startSelectionAC = &start.colSelection;
                endSelectionAC = &end.colSelection;
            }

            // Add indices along sorting channel
            size_t indexCountLow = 0, indexCountHigh = 0;
            while(nextIndexToDealWithLow < nextIndexToDealWithHigh && indexCountLow < arrayInfo.dumpingIndicesLow)
            {
                bool indexRequired = false;
                for(int indexXC : startSelectionXC)
                {
                    if(accessStateArrayDimIndepedent(stateArray, indexXC, nextIndexToDealWithLow, arrayInfo.vertical))
                    {
                        indexRequired = true;
                    }
                }
                if(indexRequired)
                {
                    if(indexCountLow > 0 && 
                        (nextIndexToDealWithLow - startSelectionAC->back()) * arrayInfo.spacingAC < Config::getInstance().minAodSpacing)
                    {
                        break;
                    }
                    startSelectionAC->push_back(nextIndexToDealWithLow);
                    endSelectionAC->push_back(arrayInfo.dumpingIndicesAC[indexCountLow++]);
                }
                if(indexCountLow < arrayInfo.dumpingIndicesLow)
                {
                    nextIndexToDealWithLow++;
                }
                else
                {
                    break;
                }
            }
            while(nextIndexToDealWithHigh > nextIndexToDealWithLow && indexCountHigh < arrayInfo.dumpingIndicesHigh)
            {
                bool indexRequired = false;
                for(int indexXC : startSelectionXC)
                {
                    if(accessStateArrayDimIndepedent(stateArray, indexXC, nextIndexToDealWithHigh, arrayInfo.vertical))
                    {
                        indexRequired = true;
                    }
                }
                if(indexRequired)
                {
                    if(indexCountHigh > 0 &&
                        (startSelectionAC->back() - nextIndexToDealWithHigh) * arrayInfo.spacingAC < Config::getInstance().minAodSpacing)
                    {
                        break;
                    }
                    startSelectionAC->push_back(nextIndexToDealWithHigh);
                    endSelectionAC->push_back(arrayInfo.dumpingIndicesAC[arrayInfo.dumpingIndicesLow + indexCountHigh++]);
                }
                if(indexCountHigh < arrayInfo.dumpingIndicesHigh)
                {
                    nextIndexToDealWithHigh--;
                }
                else
                {
                    break;
                }
            }
            std::reverse(startSelectionAC->begin() + indexCountLow, startSelectionAC->end());
            if(startSelectionAC->size() > 0)
            {
                if(startSelectionXC != endSelectionXC)
                {
                    ParallelMove::Step elbow;
                    if(arrayInfo.vertical)
                    {
                        elbow.rowSelection = start.rowSelection;
                        elbow.colSelection = end.colSelection;
                    }
                    else
                    {
                        elbow.rowSelection = end.rowSelection;
                        elbow.colSelection = start.colSelection;
                    }
                    move.steps.push_back(std::move(start));
                    move.steps.push_back(std::move(elbow));
                }
                else
                {
                    move.steps.push_back(std::move(start));
                }
                move.steps.push_back(std::move(end));
                move.execute(stateArray, logger);
                moveList.push_back(move);
            }
        }
    }
}

// Remove all atoms from initial buffer and dumping indices along the same dimension as the sorting channel
bool clearBufferAndDumpingIndicesXC(ArrayAccessor& stateArray, std::vector<ParallelMove>& moveList, 
    ArrayInformation& arrayInfo, std::shared_ptr<spdlog::logger> logger)
{    
    std::vector<int> bufferIndices, doneIndices;
    for(unsigned int i = 0; i < arrayInfo.firstNormalIndexXC; i++)
    {
        bufferIndices.push_back(i);
    }
    for(unsigned int i = arrayInfo.lastNormalIndexXCExcl; i < arrayInfo.arraySizeXC; i++)
    {
        bufferIndices.push_back(i);
    }

    // Remove atoms from registers
    for(int i : bufferIndices)
    {
        arrayInfo.usableAtomsPerXCIndex[i].clear();
        arrayInfo.unusableAtomsPerXCIndex[i].clear();
    }

    while(!bufferIndices.empty())
    {
        std::stringstream bufferIndexStream;
        bufferIndexStream << "Unused indices: (";
        logger->info("{} buffer and dumping indices remaining", bufferIndices.size());
        std::vector<int> startSelectionXC, endSelectionXC;
        for(std::vector<int>::iterator bufferIndexIter = bufferIndices.begin(); bufferIndexIter != bufferIndices.end() && 
            startSelectionXC.size() < arrayInfo.maxTonesXC;)
        {
            bufferIndexStream << *bufferIndexIter << ", ";
            if(startSelectionXC.empty() || *bufferIndexIter - startSelectionXC.back() > Config::getInstance().minAodSpacing)
            {
                startSelectionXC.push_back(*bufferIndexIter);
                endSelectionXC.push_back(*bufferIndexIter);
                bufferIndexIter = bufferIndices.erase(bufferIndexIter);
            }
            else
            {
                bufferIndexIter++;
            }
        }
        bufferIndexStream << ")";
        if(Config::getInstance().alwaysGenerateAllAODTones && startSelectionXC.size() < arrayInfo.maxTonesXC)
        {
            bufferIndexStream << ", used indices: (";
            for(int doneIndex : doneIndices)
            {
                bufferIndexStream << doneIndex << ", ";
                bool allowInsertion = true;
                for(int insertedIndex : startSelectionXC)
                {
                    if(abs(insertedIndex - doneIndex) < Config::getInstance().minAodSpacing)
                    {
                        allowInsertion = false;
                        break;
                    }
                }
                if(allowInsertion)
                {
                    startSelectionXC.push_back(doneIndex);
                    endSelectionXC.push_back(doneIndex);
                }
                if(startSelectionXC.size() >= arrayInfo.maxTonesAC)
                {
                    break;
                }
            }
            bufferIndexStream << ")";
            logger->info(bufferIndexStream.str());
            if(startSelectionXC.size() < arrayInfo.maxTonesXC)
            {
                logger->error("Only {} / {} indices could be filled while clearing buffer and dumping indices", 
                    startSelectionXC.size(), arrayInfo.maxTonesXC);
                return false;
            }
            std::sort(startSelectionXC.begin(), startSelectionXC.end());
            std::sort(endSelectionXC.begin(), endSelectionXC.end());
        }
        for(int i : startSelectionXC)
        {
            doneIndices.push_back(i);
        }

        int nextIndexToDealWithLow = arrayInfo.firstNormalIndexAC, nextIndexToDealWithHigh = arrayInfo.lastNormalIndexACExcl - 1;
        while(nextIndexToDealWithLow < nextIndexToDealWithHigh)
        {
            // Create move to remove atoms from sorting channel
            ParallelMove move;
            ParallelMove::Step start, end;
            std::vector<double> *startSelectionAC, *endSelectionAC;
            if(arrayInfo.vertical)
            {
                start.colSelection = std::vector<double>(startSelectionXC.begin(), startSelectionXC.end());
                end.colSelection = std::vector<double>(endSelectionXC.begin(), endSelectionXC.end());
                startSelectionAC = &start.rowSelection;
                endSelectionAC = &end.rowSelection;
            }
            else
            {
                start.rowSelection = std::vector<double>(startSelectionXC.begin(), startSelectionXC.end());
                end.rowSelection = std::vector<double>(endSelectionXC.begin(), endSelectionXC.end());
                startSelectionAC = &start.colSelection;
                endSelectionAC = &end.colSelection;
            }

            // Add indices along sorting channel
            size_t indexCountLow = 0, indexCountHigh = 0;
            while(nextIndexToDealWithLow < nextIndexToDealWithHigh && indexCountLow < arrayInfo.dumpingIndicesLow)
            {
                bool indexRequired = false;
                for(int indexXC : startSelectionXC)
                {
                    if(accessStateArrayDimIndepedent(stateArray, indexXC, nextIndexToDealWithLow, arrayInfo.vertical))
                    {
                        indexRequired = true;
                    }
                }
                if(indexRequired)
                {
                    if(indexCountLow > 0 && 
                        (nextIndexToDealWithLow - startSelectionAC->back()) * arrayInfo.spacingAC < Config::getInstance().minAodSpacing)
                    {
                        break;
                    }
                    startSelectionAC->push_back(nextIndexToDealWithLow);
                    endSelectionAC->push_back(arrayInfo.dumpingIndicesAC[indexCountLow++]);
                }
                if(indexCountLow < arrayInfo.dumpingIndicesLow)
                {
                    nextIndexToDealWithLow++;
                }
                else
                {
                    break;
                }
            }
            while(nextIndexToDealWithHigh > nextIndexToDealWithLow && indexCountHigh < arrayInfo.dumpingIndicesHigh)
            {
                bool indexRequired = false;
                for(int indexXC : startSelectionXC)
                {
                    if(accessStateArrayDimIndepedent(stateArray, indexXC, nextIndexToDealWithHigh, arrayInfo.vertical))
                    {
                        indexRequired = true;
                    }
                }
                if(indexRequired)
                {
                    if(indexCountHigh > 0 &&
                        (startSelectionAC->back() - nextIndexToDealWithHigh) * arrayInfo.spacingAC < Config::getInstance().minAodSpacing)
                    {
                        break;
                    }
                    startSelectionAC->push_back(nextIndexToDealWithHigh);
                    endSelectionAC->push_back(arrayInfo.dumpingIndicesAC[arrayInfo.dumpingIndicesLow + indexCountHigh++]);
                }
                if(indexCountHigh < arrayInfo.dumpingIndicesHigh)
                {
                    nextIndexToDealWithHigh--;
                }
                else
                {
                    break;
                }
            }
            std::reverse(startSelectionAC->begin() + indexCountLow, startSelectionAC->end());
            if(startSelectionAC->size() > 0)
            {
                move.steps.push_back(std::move(start));
                move.steps.push_back(std::move(end));
                move.execute(stateArray, logger);
                moveList.push_back(move);
            }
        }
    }

    return true;
}

// Create move with only one index across the channel direction
bool createSingleIndexMoves(ArrayAccessor& stateArray, std::vector<ParallelMove>& moveList, 
    std::vector<int>& startIndices, int dir, std::vector<int>& endIndices, 
    int endIndexXC, unsigned int targetCount, int indexXC, ArrayInformation& arrayInfo, 
    bool parkingMove, bool dumpingMove, std::shared_ptr<spdlog::logger> logger)
{
    unsigned int count = 0;
    std::vector<int>::iterator indexAC = startIndices.begin();
    std::vector<int>::iterator targetIndexAC = endIndices.begin();

    if(startIndices.size() == 0 || endIndices.size() == 0)
    {
        return true;
    }
    
    // Create moves while there are atoms and only as many as requested
    while(!startIndices.empty() && !endIndices.empty() && count < targetCount)
    {
        indexAC = startIndices.begin();
        targetIndexAC = endIndices.begin();

        while(indexAC != startIndices.end() && count < targetCount)
        {
            // Create move data
            ParallelMove move;
            ParallelMove::Step start, elbow1, elbow2, end;
            std::vector<double> *startSelectionAC, *elbow1SelectionAC, *elbow2SelectionAC, *endSelectionAC;
            double channelMiddle = (double)indexXC - (double)dir * (double)(arrayInfo.sortingChannelWidth + 1) / 2.;
            if(arrayInfo.vertical)
            {
                start.colSelection.push_back(indexXC);
                elbow1.colSelection.push_back(channelMiddle);
                elbow2.colSelection.push_back(channelMiddle);
                end.colSelection.push_back(endIndexXC);
                startSelectionAC = &start.rowSelection;
                elbow1SelectionAC = &elbow1.rowSelection;
                elbow2SelectionAC = &elbow2.rowSelection;
                endSelectionAC = &end.rowSelection;
            }
            else
            {
                start.rowSelection.push_back(indexXC);
                elbow1.rowSelection.push_back(channelMiddle);
                elbow2.rowSelection.push_back(channelMiddle);
                end.rowSelection.push_back(endIndexXC);
                startSelectionAC = &start.colSelection;
                elbow1SelectionAC = &elbow1.colSelection;
                elbow2SelectionAC = &elbow2.colSelection;
                endSelectionAC = &end.colSelection;
            }

            double sqDist = (endIndexXC - indexXC) * (endIndexXC - indexXC) * arrayInfo.spacingXC * arrayInfo.spacingXC;
            double minElbow4ToTargetDist = sqrt(Config::getInstance().minDistFromOccSites * Config::getInstance().minDistFromOccSites - 
                (arrayInfo.spacingAC / 2) * (arrayInfo.spacingAC / 2)) / arrayInfo.spacingXC;

            bool needToMoveBetweenTrapsAfterSortingChannel = 
                sqDist > Config::getInstance().maxSubmoveDistInPenalizedArea * Config::getInstance().maxSubmoveDistInPenalizedArea && 
                sqDist > minElbow4ToTargetDist * minElbow4ToTargetDist;

            // If there are target indices, i.e, not a removal move, iterate over start and end indices and add to move accordingly
            while(indexAC != startIndices.end() && targetIndexAC != endIndices.end() && 
                startSelectionAC->size() < arrayInfo.maxTonesAC && count < targetCount)
            {
                while(indexAC != startIndices.end() && !startSelectionAC->empty() && 
                    (*indexAC - startSelectionAC->back()) * arrayInfo.spacingAC < Config::getInstance().minAodSpacing)
                {
                    indexAC++;
                }
                while(targetIndexAC != endIndices.end() && !endSelectionAC->empty() && 
                    (*targetIndexAC - endSelectionAC->back()) * arrayInfo.spacingAC < Config::getInstance().minAodSpacing)
                {
                    targetIndexAC++;
                }
                if(indexAC != startIndices.end() && targetIndexAC != endIndices.end())
                {
                    count++;
                    startSelectionAC->push_back(*indexAC);
                    elbow1SelectionAC->push_back(*indexAC);
                    elbow2SelectionAC->push_back(*targetIndexAC);
                    endSelectionAC->push_back(*targetIndexAC);

                    indexAC = startIndices.erase(indexAC);
                    if(parkingMove)
                    {
                        arrayInfo.usableAtomsPerXCIndex[endIndexXC].push_back(*targetIndexAC);
                    }

                    // Advance target index iterator
                    if(dumpingMove)
                    {
                        // Don't remove indices from dumping zone
                        targetIndexAC++;
                    }
                    else
                    {
                        targetIndexAC = endIndices.erase(targetIndexAC);
                    }
                }
            }
            if(startSelectionAC->size() > 0)
            {
                // Execute move if it contains something
                if(parkingMove)
                {
                    std::sort(elbow2SelectionAC->begin(), elbow2SelectionAC->end());
                    std::sort(endSelectionAC->begin(), endSelectionAC->end());
                }

                if(needToMoveBetweenTrapsAfterSortingChannel)
                {
                    // Move between traps if distance longer than allowed
                    for(auto indexAC = startSelectionAC->begin(), elbow2IndexAC = elbow2SelectionAC->begin(); 
                        indexAC != startSelectionAC->end() && elbow2IndexAC != elbow2SelectionAC->end(); 
                        indexAC++, elbow2IndexAC++)
                    {
                        if(*elbow2IndexAC > *indexAC)
                        {
                            *elbow2IndexAC -= 0.5;
                        }
                        else
                        {
                            *elbow2IndexAC += 0.5;
                        }
                    }

                    // With careful consideration of neighboring atoms, one might get away with omitting elbow4
                    ParallelMove::Step elbow3, elbow4;
                    if(arrayInfo.vertical)
                    {
                        elbow3.rowSelection = elbow2.rowSelection;
                        elbow3.colSelection.push_back((double)endIndexXC + (double)dir * minElbow4ToTargetDist);
                        elbow4.rowSelection = end.rowSelection;
                        elbow4.colSelection.push_back((double)endIndexXC + (double)dir * minElbow4ToTargetDist);
                    }
                    else
                    {
                        elbow3.rowSelection.push_back((double)endIndexXC + (double)dir * minElbow4ToTargetDist);
                        elbow3.colSelection = elbow2.colSelection;
                        elbow4.rowSelection.push_back((double)endIndexXC + (double)dir * minElbow4ToTargetDist);
                        elbow4.colSelection = end.colSelection;
                    }

                    move.steps.push_back(std::move(start));
                    move.steps.push_back(std::move(elbow1));
                    move.steps.push_back(std::move(elbow2));
                    move.steps.push_back(std::move(elbow3));
                    move.steps.push_back(std::move(elbow4));
                }
                else
                {
                    move.steps.push_back(std::move(start));
                    move.steps.push_back(std::move(elbow1));
                    move.steps.push_back(std::move(elbow2));
                }

                move.steps.push_back(std::move(end));
                move.execute(stateArray, logger);
                moveList.push_back(std::move(move));

                if(dumpingMove)
                {
                    targetIndexAC = endIndices.begin();
                }
            }
            else
            {
                return true;
            }
        }
    }

    return true;
}

// Create move that combines moves from both sides in a smart way
bool createCombinedMoves(ArrayAccessor& stateArray, std::vector<ParallelMove>& moveList,
    std::set<int>& excludedStartIndices, std::vector<int>& startIndicesLowIndex, std::vector<int>& startIndicesHighIndex,
    std::vector<int>& endIndicesLowIndex, int endIndexXCLowIndex, std::vector<int>& endIndicesHighIndex, int endIndexXCHighIndex, 
    unsigned int targetCountLowIndex, unsigned int targetCountHighIndex, 
    int indexXC[2], ArrayInformation& arrayInfo, bool parkingMove, bool dumpingMove, std::shared_ptr<spdlog::logger> logger)
{
    std::set<int> sharedIndices;

    if(startIndicesLowIndex.size() > 0 && startIndicesHighIndex.size() > 0 &&
        arrayInfo.maxTonesXC >= 2 && 2 * arrayInfo.maxTonesAC <= Config::getInstance().aodTotalLimit && 
        (targetCountLowIndex > 0 || targetCountHighIndex > 0) && indexXC[0] >= (int)arrayInfo.firstNormalIndexXC && 
        indexXC[1] < (int)arrayInfo.lastNormalIndexXCExcl)
    {
        // If spacing is sufficient, use union
        // Otherwise, shadow traps might interfere so in that case only use intersection
        std::set_intersection(startIndicesLowIndex.begin(), startIndicesLowIndex.end(),
            startIndicesHighIndex.begin(), startIndicesHighIndex.end(), std::inserter(sharedIndices, sharedIndices.begin()));
        for(const auto& excluded : excludedStartIndices)
        {
            sharedIndices.erase(excluded);
        }

        std::vector<int> sharedTargetIndices;
        std::set_intersection(endIndicesLowIndex.begin(), endIndicesLowIndex.end(),
            endIndicesHighIndex.begin(), endIndicesHighIndex.end(), 
            std::inserter(sharedTargetIndices, sharedTargetIndices.begin()));

        if(sharedIndices.size() > 0 && sharedTargetIndices.size() > 0)
        {
            // Calculate move count when using separate moves
            unsigned int lowIndicesSeparate = targetCountLowIndex;
            if(startIndicesLowIndex.size() < lowIndicesSeparate)
            {
                lowIndicesSeparate = startIndicesLowIndex.size();
            }
            if(endIndicesLowIndex.size() < lowIndicesSeparate)
            {
                lowIndicesSeparate = endIndicesLowIndex.size();
            }
            unsigned int highIndicesSeparate = targetCountHighIndex;
            if(startIndicesHighIndex.size() < highIndicesSeparate)
            {
                highIndicesSeparate = startIndicesHighIndex.size();
            }
            if(endIndicesHighIndex.size() < highIndicesSeparate)
            {
                highIndicesSeparate = endIndicesHighIndex.size();
            }
            unsigned int moveCountSeparate = (lowIndicesSeparate - 1) / arrayInfo.maxTonesAC + 1 + 
                (highIndicesSeparate - 1) / arrayInfo.maxTonesAC + 1;

            // Calculate move count when combining moves
            unsigned int usedSharedIndices = sharedIndices.size();
            if(sharedTargetIndices.size() < usedSharedIndices)
            {
                usedSharedIndices = sharedTargetIndices.size();
            }
            unsigned int moveCountCombined = (lowIndicesSeparate - usedSharedIndices - 1) / arrayInfo.maxTonesAC + 1 + 
                (highIndicesSeparate - usedSharedIndices - 1) / arrayInfo.maxTonesAC + 1 + 
                (usedSharedIndices - 1) / arrayInfo.maxTonesAC + 1;
            unsigned int fullCombinedMovesCount = usedSharedIndices / arrayInfo.maxTonesAC;
            unsigned int moveCountSplitUnfilledCombinedMove = moveCountSeparate - fullCombinedMovesCount;

            unsigned int indicesInUnfilledCombinedMove = usedSharedIndices % arrayInfo.maxTonesAC;
            unsigned int indicesInUnfilledLowMove = (lowIndicesSeparate - usedSharedIndices) % arrayInfo.maxTonesAC;
            unsigned int indicesInUnfilledHighMove = (highIndicesSeparate - usedSharedIndices) % arrayInfo.maxTonesAC;
            unsigned int moveCountAddedSinglesToCombinedMove = moveCountCombined;
            if(indicesInUnfilledCombinedMove + indicesInUnfilledLowMove + indicesInUnfilledHighMove <= arrayInfo.maxTonesAC &&
                indicesInUnfilledLowMove > 0 && indicesInUnfilledHighMove > 0)
            {
                moveCountAddedSinglesToCombinedMove -= 2;
            }
            else if((indicesInUnfilledCombinedMove + indicesInUnfilledLowMove <= arrayInfo.maxTonesAC && 
                indicesInUnfilledLowMove > 0) || 
                (indicesInUnfilledCombinedMove + indicesInUnfilledHighMove <= arrayInfo.maxTonesAC && 
                indicesInUnfilledHighMove > 0) )
            {
                moveCountAddedSinglesToCombinedMove--;
            }

            if(arrayInfo.spacingXC >= Config::getInstance().minDistFromOccSites && moveCountAddedSinglesToCombinedMove < moveCountCombined && 
                moveCountAddedSinglesToCombinedMove < moveCountSplitUnfilledCombinedMove)
            {
                if(indicesInUnfilledCombinedMove + indicesInUnfilledLowMove <= arrayInfo.maxTonesAC)
                {
                    indicesInUnfilledCombinedMove += indicesInUnfilledLowMove;
                    std::set<int> indicesInOnlyLowStartRegister;
                    std::set_difference(startIndicesLowIndex.begin(), startIndicesLowIndex.end(), 
                        startIndicesHighIndex.begin(), startIndicesHighIndex.end(),
                        std::inserter(indicesInOnlyLowStartRegister, indicesInOnlyLowStartRegister.begin()));
                    unsigned int insertCount = 0;
                    for(auto indexInOnlyLowStartRegister = indicesInOnlyLowStartRegister.begin();
                        indexInOnlyLowStartRegister != indicesInOnlyLowStartRegister.end() && 
                        insertCount < indicesInUnfilledLowMove; 
                        indexInOnlyLowStartRegister++)
                    {
                        if(!excludedStartIndices.contains(*indexInOnlyLowStartRegister))
                        {
                            insertCount++;
                            sharedIndices.insert(*indexInOnlyLowStartRegister);
                        }
                    }
                }
                if(indicesInUnfilledCombinedMove + indicesInUnfilledHighMove <= arrayInfo.maxTonesAC)
                {
                    std::set<int> indicesInOnlyHighStartRegister;
                    std::set_difference(startIndicesHighIndex.begin(), startIndicesHighIndex.end(), 
                        startIndicesLowIndex.begin(), startIndicesLowIndex.end(),
                        std::inserter(indicesInOnlyHighStartRegister, indicesInOnlyHighStartRegister.begin()));
                    unsigned int insertCount = 0;
                    for(auto indexInOnlyHighStartRegister = indicesInOnlyHighStartRegister.begin();
                        indexInOnlyHighStartRegister != indicesInOnlyHighStartRegister.end() && 
                        insertCount < indicesInUnfilledHighMove; 
                        indexInOnlyHighStartRegister++)
                    {
                        if(!excludedStartIndices.contains(*indexInOnlyHighStartRegister))
                        {
                            insertCount++;
                            sharedIndices.insert(*indexInOnlyHighStartRegister);
                        }
                    }
                }
                moveCountCombined = moveCountAddedSinglesToCombinedMove;
            }
            else if(moveCountSplitUnfilledCombinedMove < moveCountCombined)
            {
                auto endIndex = sharedIndices.begin();
                std::advance(endIndex, sharedIndices.size() - fullCombinedMovesCount * arrayInfo.maxTonesAC);
                sharedIndices.erase(sharedIndices.begin(), endIndex);
                moveCountCombined = moveCountSplitUnfilledCombinedMove;
            }

            if(moveCountSeparate <= moveCountCombined)
            {
                sharedIndices.clear();
            }

            std::vector<int>::iterator targetIndexAC = sharedTargetIndices.begin();
            auto indexAC = sharedIndices.begin();

            double minElbow4ToTargetDist;
            bool needToMoveBetweenTrapsAfterSortingChannel;

            minElbow4ToTargetDist = sqrt(Config::getInstance().minDistFromOccSites * Config::getInstance().minDistFromOccSites - 
                (arrayInfo.spacingAC / 2.) * (arrayInfo.spacingAC / 2.)) / arrayInfo.spacingXC;
            needToMoveBetweenTrapsAfterSortingChannel = 
                (((double)indexXC[0] - (double)(arrayInfo.sortingChannelWidth + 1) / 2. - (double)endIndexXCLowIndex) * 
                (double)arrayInfo.spacingXC > (double)Config::getInstance().maxSubmoveDistInPenalizedArea ||
                ((double)endIndexXCHighIndex - (double)(arrayInfo.sortingChannelWidth + 1) / 2. - (double)indexXC[1]) * 
                (double)arrayInfo.spacingXC > (double)Config::getInstance().maxSubmoveDistInPenalizedArea) && 
                (((double)indexXC[0] - (double)(arrayInfo.sortingChannelWidth + 1) / 2. - (double)endIndexXCLowIndex) * 
                (double)arrayInfo.spacingXC > minElbow4ToTargetDist ||
                ((double)endIndexXCHighIndex - (double)(arrayInfo.sortingChannelWidth + 1) / 2. - (double)indexXC[1]) * 
                (double)arrayInfo.spacingXC > minElbow4ToTargetDist);

            while(!sharedIndices.empty() && !sharedTargetIndices.empty() && (targetCountLowIndex > 0 || targetCountHighIndex > 0))
            {
                indexAC = sharedIndices.begin();
                targetIndexAC = sharedTargetIndices.begin();

                ParallelMove move;
                ParallelMove::Step start, elbow1, elbow2, end;
                std::vector<double> *startSelectionAC, *elbow1SelectionAC, *elbow2SelectionAC, *endSelectionAC;
                if(arrayInfo.vertical)
                {
                    start.colSelection.push_back(indexXC[0]);
                    start.colSelection.push_back(indexXC[1]);
                    elbow1.colSelection.push_back((double)indexXC[0] + (double)(arrayInfo.sortingChannelWidth + 1) / 2.);
                    elbow1.colSelection.push_back((double)indexXC[1] - (double)(arrayInfo.sortingChannelWidth + 1) / 2.);
                    elbow2.colSelection = elbow1.colSelection;
                    end.colSelection.push_back(endIndexXCLowIndex);
                    end.colSelection.push_back(endIndexXCHighIndex);
                    startSelectionAC = &start.rowSelection;
                    elbow1SelectionAC = &elbow1.rowSelection;
                    elbow2SelectionAC = &elbow2.rowSelection;
                    endSelectionAC = &end.rowSelection;
                }
                else
                {
                    start.rowSelection.push_back(indexXC[0]);
                    start.rowSelection.push_back(indexXC[1]);
                    elbow1.rowSelection.push_back((double)indexXC[0] + (double)(arrayInfo.sortingChannelWidth + 1) / 2.);
                    elbow1.rowSelection.push_back((double)indexXC[1] - (double)(arrayInfo.sortingChannelWidth + 1) / 2.);
                    elbow2.rowSelection = elbow1.rowSelection;
                    end.rowSelection.push_back(endIndexXCLowIndex);
                    end.rowSelection.push_back(endIndexXCHighIndex);
                    startSelectionAC = &start.colSelection;
                    elbow1SelectionAC = &elbow1.colSelection;
                    elbow2SelectionAC = &elbow2.colSelection;
                    endSelectionAC = &end.colSelection;
                }

                while(indexAC != sharedIndices.end() && targetIndexAC != sharedTargetIndices.end() && 
                    startSelectionAC->size() < arrayInfo.maxTonesAC && (targetCountLowIndex > 0 || targetCountHighIndex > 0))
                {
                    while(indexAC != sharedIndices.end() && !startSelectionAC->empty() && 
                        (*indexAC - startSelectionAC->back()) * arrayInfo.spacingAC < Config::getInstance().minAodSpacing)
                    {
                        indexAC++;
                    }
                    while(targetIndexAC != sharedTargetIndices.end() && !endSelectionAC->empty() && 
                        (*targetIndexAC - endSelectionAC->back()) * arrayInfo.spacingAC < Config::getInstance().minAodSpacing)
                    {
                        targetIndexAC++;
                    }
                    if(indexAC != sharedIndices.end() && targetIndexAC != sharedTargetIndices.end())
                    {
                        startSelectionAC->push_back(*indexAC);
                        elbow1SelectionAC->push_back(*indexAC);
                        elbow2SelectionAC->push_back(*targetIndexAC);
                        endSelectionAC->push_back(*targetIndexAC);

                        bool lowIndexAtom = std::erase(arrayInfo.usableAtomsPerXCIndex[indexXC[0]], *indexAC) > 0;
                        bool highIndexAtom = std::erase(arrayInfo.usableAtomsPerXCIndex[indexXC[1]], *indexAC) > 0;
                        std::erase(startIndicesLowIndex, *indexAC);
                        std::erase(startIndicesHighIndex, *indexAC);
                        if(lowIndexAtom)
                        {
                            targetCountLowIndex--;
                            if(!dumpingMove)
                            {
                                std::erase(endIndicesLowIndex, *targetIndexAC);
                            }
                        }
                        if(highIndexAtom)
                        {
                            targetCountHighIndex--;
                            if(!dumpingMove)
                            {
                                std::erase(endIndicesHighIndex, *targetIndexAC);
                            }
                        }
                        if(parkingMove)
                        {
                            if(lowIndexAtom)
                            {
                                arrayInfo.usableAtomsPerXCIndex[endIndexXCLowIndex].push_back(*targetIndexAC);
                            }
                            if(highIndexAtom)
                            {
                                arrayInfo.usableAtomsPerXCIndex[endIndexXCHighIndex].push_back(*targetIndexAC);
                            }
                        }
                        indexAC = sharedIndices.erase(indexAC);

                        // Advance target index iterator
                        if(dumpingMove)
                        {
                            // Don't remove indices from dumping zone
                            targetIndexAC++;
                        }
                        else
                        {
                            targetIndexAC = sharedTargetIndices.erase(targetIndexAC);
                        }
                    }
                }
                if(startSelectionAC->size() > 0)
                {
                    if(parkingMove)
                    {
                        std::sort(elbow2SelectionAC->begin(), elbow2SelectionAC->end());
                        std::sort(endSelectionAC->begin(), endSelectionAC->end());
                    }

                    if(needToMoveBetweenTrapsAfterSortingChannel)
                    {
                        // Move between traps if distance longer than allowed
                        for(auto indexAC = startSelectionAC->begin(), elbow2IndexAC = elbow2SelectionAC->begin(); 
                            indexAC != startSelectionAC->end() && elbow2IndexAC != elbow2SelectionAC->end(); 
                            indexAC++, elbow2IndexAC++)
                        {
                            if(*elbow2IndexAC > *indexAC)
                            {
                                *elbow2IndexAC -= 0.5;
                            }
                            else
                            {
                                *elbow2IndexAC += 0.5;
                            }
                        }

                        ParallelMove::Step elbow3, elbow4;
                        if(arrayInfo.vertical)
                        {
                            elbow3.rowSelection = elbow2.rowSelection;
                            elbow3.colSelection.push_back((double)endIndexXCLowIndex + (double)minElbow4ToTargetDist);
                            elbow3.colSelection.push_back((double)endIndexXCHighIndex - (double)minElbow4ToTargetDist);
                            elbow4.rowSelection = end.rowSelection;
                            elbow4.colSelection.push_back((double)endIndexXCLowIndex + (double)minElbow4ToTargetDist);
                            elbow4.colSelection.push_back((double)endIndexXCHighIndex - (double)minElbow4ToTargetDist);
                        }
                        else
                        {
                            elbow3.rowSelection.push_back((double)endIndexXCLowIndex + (double)minElbow4ToTargetDist);
                            elbow3.rowSelection.push_back((double)endIndexXCHighIndex - (double)minElbow4ToTargetDist);
                            elbow3.colSelection = elbow2.colSelection;
                            elbow4.rowSelection.push_back((double)endIndexXCLowIndex + (double)minElbow4ToTargetDist);
                            elbow4.rowSelection.push_back((double)endIndexXCHighIndex - (double)minElbow4ToTargetDist);
                            elbow4.colSelection = end.colSelection;
                        }

                        move.steps.push_back(std::move(start));
                        move.steps.push_back(std::move(elbow1));
                        move.steps.push_back(std::move(elbow2));
                        move.steps.push_back(std::move(elbow3));
                        move.steps.push_back(std::move(elbow4));
                    }
                    else
                    {
                        move.steps.push_back(std::move(start));
                        move.steps.push_back(std::move(elbow1));
                        move.steps.push_back(std::move(elbow2));
                    }

                    move.steps.push_back(std::move(end));
                    move.execute(stateArray, logger);
                    moveList.push_back(std::move(move));
                }
            }
        }
    }

    // For remaining indices, create individual moves
    if(indexXC[0] >= (int)arrayInfo.firstNormalIndexXC && startIndicesLowIndex.size() > 0 && targetCountLowIndex > 0)
    {
        if(!createSingleIndexMoves(stateArray, moveList, startIndicesLowIndex, -1, endIndicesLowIndex, endIndexXCLowIndex, 
            targetCountLowIndex, indexXC[0], arrayInfo, parkingMove, dumpingMove, logger))
        {
            return false;
        }
    }
    if(indexXC[1] < (int)arrayInfo.lastNormalIndexXCExcl && startIndicesHighIndex.size() > 0 && targetCountHighIndex > 0)
    {
        if(!createSingleIndexMoves(stateArray, moveList, startIndicesHighIndex, 1, endIndicesHighIndex, endIndexXCHighIndex, 
            targetCountHighIndex, indexXC[1], arrayInfo, parkingMove, dumpingMove, logger))
        {
            return false;
        }
    }

    return true;
}

// Iterates over all normal indices and creates moves to rearrange contained atoms towards target indices
bool sortRemainingRowsOrCols(ArrayAccessor& stateArray, std::pair<int,int> startIndex, std::vector<ParallelMove>& moveList, 
    ArrayInformation& arrayInfo, std::shared_ptr<spdlog::logger> logger)
{
    // If sorting channel starts in the middle, there are two indices: one going up and one down, 
    // [0] means index is decreasing, [1] means increasing
    int indexXC[2], currentTargetIndexXC[2];
    if(std::get<0>(startIndex) <= (int)arrayInfo.firstNormalIndexXC)
    {
        indexXC[0] = -1;
        indexXC[1] = arrayInfo.firstNormalIndexXC + arrayInfo.sortingChannelWidth + 1;
        currentTargetIndexXC[0] = -1;
        currentTargetIndexXC[1] = arrayInfo.firstNormalIndexXC;
    }
    else if(std::get<1>(startIndex) >= (int)arrayInfo.lastNormalIndexXCExcl)
    {
        indexXC[0] = arrayInfo.arraySizeXC;
        indexXC[1] = arrayInfo.lastNormalIndexXCExcl - arrayInfo.sortingChannelWidth - 2;
        currentTargetIndexXC[0] = arrayInfo.arraySizeXC;
        currentTargetIndexXC[1] = arrayInfo.lastNormalIndexXCExcl - 1;
    }
    else
    {
        indexXC[0] = std::get<0>(startIndex) - 1;
        indexXC[1] = std::get<1>(startIndex);
        currentTargetIndexXC[0] = std::get<0>(startIndex) + arrayInfo.sortingChannelWidth;
        currentTargetIndexXC[1] = std::get<0>(startIndex) + arrayInfo.sortingChannelWidth + 1;
    }
    unsigned int requiredAtoms[2] = {0, 0};
    std::vector<int> parkingSpotsRemainingAtCurrentIndexXC[2];
    for(size_t i = 0; i < 2; i++)
    {
        if(currentTargetIndexXC[i] >= (int)arrayInfo.firstNormalIndexXC && currentTargetIndexXC[i] < (int)arrayInfo.lastNormalIndexXCExcl)
        {
            parkingSpotsRemainingAtCurrentIndexXC[i] = arrayInfo.parkingSitesPerXCIndex[currentTargetIndexXC[i]];
        }
    }
    int dir[2] = {-1, 1};

    unsigned int totalRequiredAtoms = std::accumulate(arrayInfo.targetSitesPerXCIndex.begin(), arrayInfo.targetSitesPerXCIndex.end(), 0u, 
        [](unsigned int init, const auto& elem) { return init + elem.size(); });

    while(indexXC[0] >= (int)arrayInfo.firstNormalIndexXC || indexXC[1] < (int)arrayInfo.lastNormalIndexXCExcl)
    {
        if(logger->level() <= spdlog::level::debug)
        {
            logger->debug("indexXC[0]: {}, indexXC[1]: {}", indexXC[0], indexXC[1]);
            std::stringstream unusableAtomsStr;
            if(indexXC[0] >= 0 && indexXC[0] < (int)arrayInfo.unusableAtomsPerXCIndex.size())
            {
                unusableAtomsStr << "unusableAtoms[indexXC[0]]: ";
                for(auto unusableAtom : arrayInfo.unusableAtomsPerXCIndex[indexXC[0]])
                {
                    unusableAtomsStr << unusableAtom << ", ";
                }
                unusableAtomsStr << "\nusableAtoms[indexXC[0]]: ";
                for(auto usableAtom : arrayInfo.usableAtomsPerXCIndex[indexXC[0]])
                {
                    unusableAtomsStr << usableAtom << ", ";
                }
            }
            if(indexXC[1] >= 0 && indexXC[1] < (int)arrayInfo.unusableAtomsPerXCIndex.size())
            {
                unusableAtomsStr << "\nunusableAtoms[indexXC[1]]: ";
                for(auto unusableAtom : arrayInfo.unusableAtomsPerXCIndex[indexXC[1]])
                {
                    unusableAtomsStr << unusableAtom << ", ";
                }
                unusableAtomsStr << "\nusableAtoms[indexXC[1]]: ";
                for(auto usableAtom : arrayInfo.usableAtomsPerXCIndex[indexXC[1]])
                {
                    unusableAtomsStr << usableAtom << ", ";
                }
            }
            logger->debug(unusableAtomsStr.str());
        }

        if(indexXC[0] < (int)arrayInfo.firstRelevantXC && indexXC[1] >= (int)arrayInfo.lastRelevantXCExcl && totalRequiredAtoms == 0)
        {
            // If both indices are outside computational zone and we don't need more atoms, then we are done
            return true;
        }
        size_t targetIndexXC[2];
        unsigned int parkingSpots[2];
        std::set<int> excludedIndicesForRemovalAndParkingMoves;
        for(size_t i = 0; i < 2; i++)
        {
            if(indexXC[i] >= (int)arrayInfo.firstNormalIndexXC && indexXC[i] < (int)arrayInfo.lastNormalIndexXCExcl)
            {
                targetIndexXC[i] = indexXC[i] - dir[i] * (arrayInfo.sortingChannelWidth + 1);
                requiredAtoms[i] = arrayInfo.targetSitesPerXCIndex[currentTargetIndexXC[i]].size();
                parkingSpots[i] = parkingSpotsRemainingAtCurrentIndexXC[i].size();

                for(int indexXCTowardsLastTargetIndex = currentTargetIndexXC[i] + dir[i]; 
                    indexXCTowardsLastTargetIndex * dir[i] <= (int)targetIndexXC[i] * dir[i]; 
                    indexXCTowardsLastTargetIndex += dir[i])
                {
                    requiredAtoms[i] += arrayInfo.targetSitesPerXCIndex[indexXCTowardsLastTargetIndex].size();
                    parkingSpots[i] += arrayInfo.parkingSitesPerXCIndex[indexXCTowardsLastTargetIndex].size();
                }
                logger->debug("i: {}, indexXC: {}, requiredAtoms: {}, totalRequiredAtoms: {}, parkingSpots: {}", 
                    i, indexXC[i], requiredAtoms[i], totalRequiredAtoms, parkingSpots[i]);

                // Excess atoms that cannot be used for filling target sites or parking spots are thrown away
                while(arrayInfo.usableAtomsPerXCIndex[indexXC[i]].size() > requiredAtoms[i] + parkingSpots[i])
                {
                    if(arrayInfo.usableAtomsPerXCIndex[indexXC[i]][0] > 
                        (int)arrayInfo.arraySizeAC - arrayInfo.usableAtomsPerXCIndex[indexXC[i]].back() - 1)
                    {
                        arrayInfo.unusableAtomsPerXCIndex[indexXC[i]].insert(std::upper_bound(
                            arrayInfo.unusableAtomsPerXCIndex[indexXC[i]].begin(), 
                            arrayInfo.unusableAtomsPerXCIndex[indexXC[i]].end(), 
                            arrayInfo.usableAtomsPerXCIndex[indexXC[i]][0]), 
                            arrayInfo.usableAtomsPerXCIndex[indexXC[i]][0]);
                        arrayInfo.usableAtomsPerXCIndex[indexXC[i]].erase(arrayInfo.usableAtomsPerXCIndex[indexXC[i]].begin());
                    }
                    else
                    {
                        arrayInfo.unusableAtomsPerXCIndex[indexXC[i]].insert(std::upper_bound(
                            arrayInfo.unusableAtomsPerXCIndex[indexXC[i]].begin(), 
                            arrayInfo.unusableAtomsPerXCIndex[indexXC[i]].end(), 
                            arrayInfo.usableAtomsPerXCIndex[indexXC[i]].back()), 
                            arrayInfo.usableAtomsPerXCIndex[indexXC[i]].back());
                        arrayInfo.usableAtomsPerXCIndex[indexXC[i]].pop_back();
                    }
                }

                for(const auto& usableIndex : arrayInfo.usableAtomsPerXCIndex[indexXC[i]])
                {
                    for(int excludedIndex = usableIndex - arrayInfo.targetGapAC + 1; 
                        excludedIndex < usableIndex + arrayInfo.targetGapAC; excludedIndex++)
                    {
                        excludedIndicesForRemovalAndParkingMoves.insert(excludedIndex);
                    }
                }
            }
        }
        
        // Remove unusable atoms
        if(indexXC[0] >= (int)arrayInfo.firstNormalIndexXC && indexXC[1] < (int)arrayInfo.lastNormalIndexXCExcl)
        {
            if(!createCombinedMoves(stateArray, moveList, excludedIndicesForRemovalAndParkingMoves, 
                arrayInfo.unusableAtomsPerXCIndex[indexXC[0]], arrayInfo.unusableAtomsPerXCIndex[indexXC[1]], 
                arrayInfo.dumpingIndicesAC, currentTargetIndexXC[0], arrayInfo.dumpingIndicesAC, currentTargetIndexXC[1], 
                arrayInfo.unusableAtomsPerXCIndex[indexXC[0]].size(), 
                arrayInfo.unusableAtomsPerXCIndex[indexXC[1]].size(), indexXC, arrayInfo, false, true, logger))
            {
                return false;
            }
        }
        else if(indexXC[0] >= (int)arrayInfo.firstNormalIndexXC)
        {
            if(!createSingleIndexMoves(stateArray, moveList, arrayInfo.unusableAtomsPerXCIndex[indexXC[0]], -1, 
                arrayInfo.dumpingIndicesAC, currentTargetIndexXC[0], arrayInfo.unusableAtomsPerXCIndex[indexXC[0]].size(), 
                indexXC[0], arrayInfo, false, true, logger))
            {
                return false;
            }
        }
        else if(indexXC[1] < (int)arrayInfo.lastNormalIndexXCExcl)
        {
            if(!createSingleIndexMoves(stateArray, moveList, arrayInfo.unusableAtomsPerXCIndex[indexXC[1]], 1, 
                arrayInfo.dumpingIndicesAC, currentTargetIndexXC[1], arrayInfo.unusableAtomsPerXCIndex[indexXC[1]].size(), 
                indexXC[1], arrayInfo, false, true, logger))
            {
                return false;
            }
        };

        while((indexXC[0] >= (int)arrayInfo.firstNormalIndexXC && !arrayInfo.usableAtomsPerXCIndex[indexXC[0]].empty()) || 
            (indexXC[1] < (int)arrayInfo.lastNormalIndexXCExcl && !arrayInfo.usableAtomsPerXCIndex[indexXC[1]].empty()))
        {
            bool lowIndexContainsAtoms = indexXC[0] >= (int)arrayInfo.firstNormalIndexXC && !arrayInfo.usableAtomsPerXCIndex[indexXC[0]].empty();
            bool highIndexContainsAtoms = indexXC[1] < (int)arrayInfo.lastNormalIndexXCExcl && !arrayInfo.usableAtomsPerXCIndex[indexXC[1]].empty();
            // If there are too many atoms to use, move excess atoms to parking spots
            if(lowIndexContainsAtoms && highIndexContainsAtoms)
            {
                unsigned int atomsToParkLowIndex = arrayInfo.usableAtomsPerXCIndex[indexXC[0]].size() > requiredAtoms[0] ? 
                    arrayInfo.usableAtomsPerXCIndex[indexXC[0]].size() - requiredAtoms[0] : 0;
                unsigned int atomsToParkHighIndex = arrayInfo.usableAtomsPerXCIndex[indexXC[1]].size() > requiredAtoms[1] ? 
                    arrayInfo.usableAtomsPerXCIndex[indexXC[1]].size() - requiredAtoms[1] : 0;
                if(atomsToParkLowIndex > 0 || atomsToParkHighIndex > 0)
                {
                    logger->debug("Parking {} and {} atoms", atomsToParkLowIndex, atomsToParkHighIndex);
                    if(!createCombinedMoves(stateArray, moveList, excludedIndicesForRemovalAndParkingMoves,
                        arrayInfo.usableAtomsPerXCIndex[indexXC[0]], arrayInfo.usableAtomsPerXCIndex[indexXC[1]], 
                        parkingSpotsRemainingAtCurrentIndexXC[0], currentTargetIndexXC[0], 
                        parkingSpotsRemainingAtCurrentIndexXC[1], currentTargetIndexXC[1],
                        atomsToParkLowIndex, atomsToParkHighIndex, indexXC, arrayInfo, true, false, logger))
                    {
                        return false;
                    }
                }
            }
            else if(lowIndexContainsAtoms)
            {
                unsigned int atomsToParkLowIndex = arrayInfo.usableAtomsPerXCIndex[indexXC[0]].size() > requiredAtoms[0] ? 
                    arrayInfo.usableAtomsPerXCIndex[indexXC[0]].size() - requiredAtoms[0] : 0;
                if(atomsToParkLowIndex > 0)
                {
                    logger->debug("Parking {} atoms", atomsToParkLowIndex);
                    if(!createSingleIndexMoves(stateArray, moveList, arrayInfo.usableAtomsPerXCIndex[indexXC[0]], -1, 
                        parkingSpotsRemainingAtCurrentIndexXC[0], currentTargetIndexXC[0], 
                        atomsToParkLowIndex, indexXC[0], arrayInfo, true, false, logger))
                    {
                        return false;
                    }
                }
            }
            else if(highIndexContainsAtoms)
            {
                unsigned int atomsToParkHighIndex = arrayInfo.usableAtomsPerXCIndex[indexXC[1]].size() > requiredAtoms[1] ?
                    arrayInfo.usableAtomsPerXCIndex[indexXC[1]].size() - requiredAtoms[1] : 0;
                if(atomsToParkHighIndex > 0)
                {
                    logger->debug("Parking {} atoms", atomsToParkHighIndex);
                    if(!createSingleIndexMoves(stateArray, moveList, arrayInfo.usableAtomsPerXCIndex[indexXC[1]], 1, 
                        parkingSpotsRemainingAtCurrentIndexXC[1], currentTargetIndexXC[1], 
                        atomsToParkHighIndex, indexXC[1], arrayInfo, true, false, logger))
                    {
                        return false;
                    }
                }
            }

            // Increase row if there are no target sites
            bool atLeastOneWasMoved = false;
            bool lowIndexSideExists = indexXC[0] >= (int)arrayInfo.firstNormalIndexXC && indexXC[0] < (int)arrayInfo.lastNormalIndexXCExcl && 
                currentTargetIndexXC[0] >= (int)arrayInfo.firstNormalIndexXC && currentTargetIndexXC[0] < (int)arrayInfo.lastNormalIndexXCExcl;
            bool highIndexSideExists = indexXC[1] >= (int)arrayInfo.firstNormalIndexXC && indexXC[1] < (int)arrayInfo.lastNormalIndexXCExcl && 
                currentTargetIndexXC[1] >= (int)arrayInfo.firstNormalIndexXC && currentTargetIndexXC[1] < (int)arrayInfo.lastNormalIndexXCExcl;

            if(lowIndexSideExists && arrayInfo.targetSitesPerXCIndex[currentTargetIndexXC[0]].empty() && 
                currentTargetIndexXC[0] > (int)targetIndexXC[0] && currentTargetIndexXC[0] > 0)
            {
                atLeastOneWasMoved = true;
                currentTargetIndexXC[0]--;
                parkingSpotsRemainingAtCurrentIndexXC[0] = arrayInfo.parkingSitesPerXCIndex[currentTargetIndexXC[0]];
            }

            if(highIndexSideExists && arrayInfo.targetSitesPerXCIndex[currentTargetIndexXC[1]].empty() &&
                currentTargetIndexXC[1] < (int)targetIndexXC[1] && currentTargetIndexXC[1] < (int)arrayInfo.lastNormalIndexXCExcl - 1)
            {
                atLeastOneWasMoved = true;
                currentTargetIndexXC[1]++;
                parkingSpotsRemainingAtCurrentIndexXC[1] = arrayInfo.parkingSitesPerXCIndex[currentTargetIndexXC[1]];
            }

            if((!lowIndexSideExists || arrayInfo.targetSitesPerXCIndex[currentTargetIndexXC[0]].empty()) &&
                (!highIndexSideExists || arrayInfo.targetSitesPerXCIndex[currentTargetIndexXC[1]].empty()) && !atLeastOneWasMoved)
            {
                logger->debug("Breaking out of loop as there are no more targets and target can't be moved");
                break;
            }
            
            if(!atLeastOneWasMoved)
            {
                excludedIndicesForRemovalAndParkingMoves.clear();

                size_t requiredAtomsBefore[2] = {arrayInfo.targetSitesPerXCIndex[currentTargetIndexXC[0]].size(), 
                    arrayInfo.targetSitesPerXCIndex[currentTargetIndexXC[1]].size()};
                if(lowIndexContainsAtoms && highIndexContainsAtoms)
                {
                    if(!createCombinedMoves(stateArray, moveList, excludedIndicesForRemovalAndParkingMoves, 
                        arrayInfo.usableAtomsPerXCIndex[indexXC[0]], arrayInfo.usableAtomsPerXCIndex[indexXC[1]], 
                        arrayInfo.targetSitesPerXCIndex[currentTargetIndexXC[0]], currentTargetIndexXC[0], 
                        arrayInfo.targetSitesPerXCIndex[currentTargetIndexXC[1]], currentTargetIndexXC[1],
                        arrayInfo.usableAtomsPerXCIndex[indexXC[0]].size(), arrayInfo.usableAtomsPerXCIndex[indexXC[1]].size(), 
                        indexXC, arrayInfo, false, false, logger))
                    {
                        return false;
                    }
                }
                else if(lowIndexContainsAtoms)
                {
                    if(!createSingleIndexMoves(stateArray, moveList, arrayInfo.usableAtomsPerXCIndex[indexXC[0]], -1, 
                        arrayInfo.targetSitesPerXCIndex[currentTargetIndexXC[0]], currentTargetIndexXC[0], 
                        arrayInfo.usableAtomsPerXCIndex[indexXC[0]].size(), indexXC[0], arrayInfo, false, false, logger))
                    {
                        return false;
                    }
                }
                else if(highIndexContainsAtoms)
                {
                    if(!createSingleIndexMoves(stateArray, moveList, arrayInfo.usableAtomsPerXCIndex[indexXC[1]], 1, 
                        arrayInfo.targetSitesPerXCIndex[currentTargetIndexXC[1]], currentTargetIndexXC[1], 
                        arrayInfo.usableAtomsPerXCIndex[indexXC[1]].size(), indexXC[1], arrayInfo, false, false, logger))
                    {
                        return false;
                    }
                }

                // We need to fill this set again as there may be fewer usable atoms
                for(unsigned int side : {0,1})
                {
                    unsigned int sortedAtomsThisSide = requiredAtomsBefore[side] - arrayInfo.targetSitesPerXCIndex[currentTargetIndexXC[side]].size();
                    requiredAtoms[side] -= sortedAtomsThisSide;
                    totalRequiredAtoms -= sortedAtomsThisSide;
                    if(indexXC[side] >= (int)arrayInfo.firstNormalIndexXC && indexXC[side] < (int)arrayInfo.lastNormalIndexXCExcl)
                    {
                        for(const auto& usableIndex : arrayInfo.usableAtomsPerXCIndex[indexXC[side]])
                        {
                            for(int excludedIndex = usableIndex - arrayInfo.targetGapAC + 1; 
                                excludedIndex < usableIndex + arrayInfo.targetGapAC; excludedIndex++)
                            {
                                excludedIndicesForRemovalAndParkingMoves.insert(excludedIndex);
                            }
                        }
                    }
                }
            }
        }
        indexXC[0]--;
        indexXC[1]++;
    }

    return totalRequiredAtoms == 0;
}

// Gather general information about the array and save in struct for later use
std::optional<ArrayInformation> conductInitialAnalysis(ArrayAccessor& stateArray, 
    pybind11::detail::unchecked_reference<TargetState, 2>& targetGeometry, std::shared_ptr<spdlog::logger> logger)
{
    ArrayInformation arrayInfo;

    if(Config::getInstance().verticalSortingChannel.has_value())
    {
        arrayInfo.vertical = Config::getInstance().verticalSortingChannel.value();
    }
    else
    {
        arrayInfo.vertical = Config::getInstance().columnSpacing > Config::getInstance().rowSpacing;
    }

    // Initialize data independent of whether sorting channel is vertical or horizontal
    arrayInfo.maxTonesXC = Config::getInstance().aodTotalLimit;
    arrayInfo.maxTonesAC = Config::getInstance().aodTotalLimit;
    if(arrayInfo.vertical)
    {
        if(Config::getInstance().aodColLimit < arrayInfo.maxTonesXC)
        {
            arrayInfo.maxTonesXC = Config::getInstance().aodColLimit;
        }
        if(Config::getInstance().aodRowLimit < arrayInfo.maxTonesAC)
        {
            arrayInfo.maxTonesAC = Config::getInstance().aodRowLimit;
        }
        arrayInfo.spacingXC = Config::getInstance().columnSpacing;
        arrayInfo.spacingAC = Config::getInstance().rowSpacing;
        arrayInfo.arraySizeXC = stateArray.cols();
        arrayInfo.arraySizeAC = stateArray.rows();
    }
    else
    {
        if(Config::getInstance().aodRowLimit < arrayInfo.maxTonesXC)
        {
            arrayInfo.maxTonesXC = Config::getInstance().aodRowLimit;
        }
        if(Config::getInstance().aodColLimit < arrayInfo.maxTonesAC)
        {
            arrayInfo.maxTonesAC = Config::getInstance().aodColLimit;
        }
        arrayInfo.spacingXC = Config::getInstance().rowSpacing;
        arrayInfo.spacingAC = Config::getInstance().columnSpacing;
        arrayInfo.arraySizeXC = stateArray.rows();
        arrayInfo.arraySizeAC = stateArray.cols();
    }

    double maxMinOccDistMinAODSpacing = Config::getInstance().minDistFromOccSites;
    if(Config::getInstance().minAodSpacing > maxMinOccDistMinAODSpacing)
    {
        maxMinOccDistMinAODSpacing = Config::getInstance().minAodSpacing;
    }

    arrayInfo.sortingChannelWidth = (int)(ceil(Config::getInstance().minDistFromOccSites / (arrayInfo.spacingXC / 2.)) / 2) * 2;
    arrayInfo.targetGapXC = ceil(maxMinOccDistMinAODSpacing / arrayInfo.spacingXC);
    arrayInfo.targetGapAC = ceil(maxMinOccDistMinAODSpacing / arrayInfo.spacingAC);
    int targetRowGap = ceil(maxMinOccDistMinAODSpacing / Config::getInstance().rowSpacing);
    int targetColGap = ceil(maxMinOccDistMinAODSpacing / Config::getInstance().columnSpacing);

    // Create mask where true existing atoms would prevent an atoms usability
    auto usabilityPreventingNeighborhoodMask = generateMask(Config::getInstance().minDistFromOccSites);
    int usabilityPreventingNeighborhoodMaskRowDist = usabilityPreventingNeighborhoodMask.rows() / 2;
    int usabilityPreventingNeighborhoodMaskColDist = usabilityPreventingNeighborhoodMask.cols() / 2;
    usabilityPreventingNeighborhoodMask(usabilityPreventingNeighborhoodMaskRowDist, usabilityPreventingNeighborhoodMaskColDist) = false;

    // Iterate over array, check for usability-preventing neighbors, and sort into structure accordingly
    int maxIrrelevantRowLow = -1, minIrrelevantRowHigh = stateArray.rows(), 
        maxIrrelevantColLow = -1, minIrrelevantColHigh = stateArray.cols();
    for(size_t row = 0; row < stateArray.rows(); row++)
    {
        bool fullyIrrelevant = true;
        for(size_t col = 0; col < stateArray.cols(); col++)
        {
            if(targetGeometry(row,col) != TargetState::IRRELEVANT)
            {
                fullyIrrelevant = false;
                break;
            }
        }
        if(!fullyIrrelevant)
        {
            maxIrrelevantRowLow = (int)row - 1;
            break;
        }
    }
    for(size_t row = stateArray.rows() - 1; row >= 0 && (int)row >= maxIrrelevantRowLow; row--)
    {
        bool fullyIrrelevant = true;
        for(size_t col = 0; col < stateArray.cols(); col++)
        {
            if(targetGeometry(row,col) != TargetState::IRRELEVANT)
            {
                fullyIrrelevant = false;
                break;
            }
        }
        if(!fullyIrrelevant)
        {
            minIrrelevantRowHigh = row + 1;
            break;
        }
    }
    for(size_t col = 0; col < stateArray.cols(); col++)
    {
        bool fullyIrrelevant = true;
        for(size_t row = 0; row < stateArray.rows(); row++)
        {
            if(targetGeometry(row,col) != TargetState::IRRELEVANT)
            {
                fullyIrrelevant = false;
                break;
            }
        }
        if(!fullyIrrelevant)
        {
            maxIrrelevantColLow = (int)col - 1;
            break;
        }
    }
    for(size_t col = stateArray.cols() - 1; col >= 0 && (int)col >= maxIrrelevantColLow; col--)
    {
        bool fullyIrrelevant = true;
        for(size_t row = 0; row < stateArray.rows(); row++)
        {
            if(targetGeometry(row,col) != TargetState::IRRELEVANT)
            {
                fullyIrrelevant = false;
                break;
            }
        }
        if(!fullyIrrelevant)
        {
            minIrrelevantColHigh = col + 1;
            break;
        }
    }
    logger->info("Borders of fully irrelevant regions: {} - {}, cols: {} - {}", 
        maxIrrelevantRowLow, minIrrelevantRowHigh, maxIrrelevantColLow, minIrrelevantColHigh);

    // If there should always be the same number of tones, then we require a region along a vertical and a 
    // horizontal border that we empty to use for these additional tones. We call this the buffer
    
    double currentBufferRowLow = 0, currentBufferRowHigh = stateArray.rows() - 1, 
        currentBufferColLow = 0, currentBufferColHigh = stateArray.cols() - 1;
    int firstNormalRow = 0, lastNormalRowExcl = stateArray.rows(), firstNormalCol = 0, lastNormalColExcl = stateArray.cols();
    if(Config::getInstance().alwaysGenerateAllAODTones)
    {
        double rowBufferSpacing = ceil(Config::getInstance().minAodSpacing / Config::getInstance().rowSpacing);
        arrayInfo.bufferRows.resize(Config::getInstance().aodRowLimit);
        size_t insertedElems = 0, lowInsertionLocation = 0, highInsertionLocation = Config::getInstance().aodRowLimit - 1;
        while(insertedElems < Config::getInstance().aodRowLimit)
        {
            bool atLeastOneInserted = false;
            if(currentBufferRowLow <= maxIrrelevantRowLow)
            {
                arrayInfo.bufferRows[lowInsertionLocation++] = currentBufferRowLow;    
                currentBufferRowLow += rowBufferSpacing;
                insertedElems++;
                atLeastOneInserted = true;
            }
            if(insertedElems < Config::getInstance().aodRowLimit && currentBufferRowHigh >= minIrrelevantRowHigh)
            {
                arrayInfo.bufferRows[highInsertionLocation--] = currentBufferRowHigh;
                currentBufferRowHigh -= rowBufferSpacing;
                insertedElems++;
                atLeastOneInserted = true;
            }
            if(!atLeastOneInserted)
            {
                logger->error("Not enough buffer rows ({}/{}). Aborting...", insertedElems, Config::getInstance().aodRowLimit);
                return std::nullopt;
            }
        }

        double colBufferSpacing = ceil(Config::getInstance().minAodSpacing / Config::getInstance().columnSpacing);
        arrayInfo.bufferCols.resize(Config::getInstance().aodColLimit);
        insertedElems = 0, lowInsertionLocation = 0, highInsertionLocation = Config::getInstance().aodColLimit - 1;
        while(insertedElems < Config::getInstance().aodColLimit)
        {
            bool atLeastOneInserted = false;
            if(currentBufferColLow <= maxIrrelevantColLow)
            {
                arrayInfo.bufferCols[lowInsertionLocation++] = currentBufferColLow;    
                currentBufferColLow += colBufferSpacing;
                insertedElems++;
                atLeastOneInserted = true;
            }
            if(insertedElems < Config::getInstance().aodColLimit && currentBufferColHigh >= minIrrelevantColHigh)
            {
                arrayInfo.bufferCols[highInsertionLocation--] = currentBufferColHigh;
                currentBufferColHigh -= colBufferSpacing;
                insertedElems++;
                atLeastOneInserted = true;
            }
            if(!atLeastOneInserted)
            {
                logger->error("Not enough buffer cols ({}/{}). Aborting...", insertedElems, Config::getInstance().aodColLimit);
                return std::nullopt;
            }
        }
    }
    else
    
    if(maxMinOccDistMinAODSpacing > Config::getInstance().minAodSpacing)
    {
        currentBufferRowLow += (maxMinOccDistMinAODSpacing - Config::getInstance().minAodSpacing) / Config::getInstance().rowSpacing;
        currentBufferRowHigh -= (maxMinOccDistMinAODSpacing - Config::getInstance().minAodSpacing) / Config::getInstance().rowSpacing;
        currentBufferColLow += (maxMinOccDistMinAODSpacing - Config::getInstance().minAodSpacing) / Config::getInstance().columnSpacing;
        currentBufferColHigh -= (maxMinOccDistMinAODSpacing - Config::getInstance().minAodSpacing) / Config::getInstance().columnSpacing;
    }
    firstNormalRow = ceil(currentBufferRowLow);
    lastNormalRowExcl = floor(currentBufferRowHigh) + 1;
    firstNormalCol = ceil(currentBufferColLow);
    lastNormalColExcl = floor(currentBufferColHigh) + 1;

    int dumpingSpacing = ceil(Config::getInstance().minAodSpacing / arrayInfo.spacingAC);
    arrayInfo.dumpingIndicesAC.resize(arrayInfo.maxTonesAC);
    arrayInfo.dumpingIndicesLow = 0;
    arrayInfo.dumpingIndicesHigh = 0;
    size_t insertionLocation = arrayInfo.maxTonesAC - 1;;
    if(arrayInfo.vertical)
    {
        for(int nextLowestRow = ceil(currentBufferRowLow); nextLowestRow <= maxIrrelevantRowLow && 
            arrayInfo.dumpingIndicesLow < arrayInfo.maxTonesAC; nextLowestRow += dumpingSpacing)
        {
            arrayInfo.dumpingIndicesAC[arrayInfo.dumpingIndicesLow++] = nextLowestRow;
            firstNormalRow = nextLowestRow + targetRowGap;
        }
        for(int nextHighestRow = floor(currentBufferRowHigh); nextHighestRow >= minIrrelevantRowHigh && 
            arrayInfo.dumpingIndicesLow + arrayInfo.dumpingIndicesHigh < arrayInfo.maxTonesAC; nextHighestRow -= dumpingSpacing)
        {
            arrayInfo.dumpingIndicesAC[insertionLocation--] = nextHighestRow;
            arrayInfo.dumpingIndicesHigh++;
            lastNormalRowExcl = nextHighestRow - targetRowGap + 1;
        }
    }
    else
    {
        for(int nextLowestCol = ceil(currentBufferColLow); nextLowestCol <= maxIrrelevantColLow && 
            arrayInfo.dumpingIndicesLow < arrayInfo.maxTonesAC; nextLowestCol += dumpingSpacing)
        {
            arrayInfo.dumpingIndicesAC[arrayInfo.dumpingIndicesLow++] = nextLowestCol;
            firstNormalCol = nextLowestCol + targetColGap;
        }
        for(int nextHighestCol = floor(currentBufferColHigh); nextHighestCol >= minIrrelevantColHigh && 
            arrayInfo.dumpingIndicesLow + arrayInfo.dumpingIndicesHigh < arrayInfo.maxTonesAC; nextHighestCol -= dumpingSpacing)
        {
            arrayInfo.dumpingIndicesAC[insertionLocation--] = nextHighestCol;
            arrayInfo.dumpingIndicesHigh++;
            lastNormalColExcl = nextHighestCol - targetColGap + 1;
        }
    }
    if(arrayInfo.dumpingIndicesLow + arrayInfo.dumpingIndicesHigh < arrayInfo.maxTonesAC)
    {
        logger->error("Not enough dumping indices ({}/{}). Aborting...", 
            arrayInfo.dumpingIndicesLow + arrayInfo.dumpingIndicesHigh, arrayInfo.maxTonesAC);
        return std::nullopt;
    }

    arrayInfo.unusableAtomsPerXCIndex.resize(arrayInfo.arraySizeXC);
    arrayInfo.usableAtomsPerXCIndex.resize(arrayInfo.arraySizeXC);
    arrayInfo.targetSitesPerXCIndex.resize(arrayInfo.arraySizeXC);
    arrayInfo.parkingSitesPerXCIndex.resize(arrayInfo.arraySizeXC);

    if(arrayInfo.vertical)
    {
        arrayInfo.firstRelevantXC = maxIrrelevantColLow + 1;
        arrayInfo.lastRelevantXCExcl = minIrrelevantColHigh;
        arrayInfo.firstRelevantAC = maxIrrelevantRowLow + 1;
        arrayInfo.lastRelevantACExcl = minIrrelevantRowHigh;
        arrayInfo.firstNormalIndexXC = firstNormalCol;
        arrayInfo.lastNormalIndexXCExcl = lastNormalColExcl;
        arrayInfo.firstNormalIndexAC = firstNormalRow;
        arrayInfo.lastNormalIndexACExcl = lastNormalRowExcl;
    }
    else
    {
        arrayInfo.firstRelevantXC = maxIrrelevantRowLow + 1;
        arrayInfo.lastRelevantXCExcl = minIrrelevantRowHigh;
        arrayInfo.firstRelevantAC = maxIrrelevantColLow + 1;
        arrayInfo.lastRelevantACExcl = minIrrelevantColHigh;
        arrayInfo.firstNormalIndexXC = firstNormalRow;
        arrayInfo.lastNormalIndexXCExcl = lastNormalRowExcl;
        arrayInfo.firstNormalIndexAC = firstNormalCol;
        arrayInfo.lastNormalIndexACExcl = lastNormalColExcl;
    }

    // Iterate over array, check for usability-preventing neighbors, and sort into structure accordingly
    for(size_t row = firstNormalRow; (int)row < lastNormalRowExcl && row < stateArray.rows(); row++)
    {
        for(size_t col = firstNormalCol; (int)col < lastNormalColExcl && col < stateArray.cols(); col++)
        {
            size_t indexXC = arrayInfo.vertical ? col : row;
            size_t indexAC = arrayInfo.vertical ? row : col;
            
            if(targetGeometry(row, col) == TargetState::IRRELEVANT)
            {
                bool validParkingSite = true;
                if(!arrayInfo.parkingSitesPerXCIndex[indexXC].empty() && 
                    (indexAC - arrayInfo.parkingSitesPerXCIndex[indexXC].back()) * arrayInfo.spacingAC < maxMinOccDistMinAODSpacing)
                {
                    validParkingSite = false;
                }
                else
                {
                    for(int rowShift = -usabilityPreventingNeighborhoodMaskRowDist; 
                        validParkingSite && rowShift <= usabilityPreventingNeighborhoodMaskRowDist; rowShift++)
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
                                    (targetGeometry(shiftedRow, shiftedCol) != TargetState::IRRELEVANT ||
                                    (indexXC - arrayInfo.firstNormalIndexXC + 1) * arrayInfo.spacingXC < maxMinOccDistMinAODSpacing || 
                                    (arrayInfo.lastNormalIndexXCExcl - indexXC) * arrayInfo.spacingXC < maxMinOccDistMinAODSpacing || 
                                    (indexAC - arrayInfo.firstNormalIndexAC + 1) * arrayInfo.spacingAC < maxMinOccDistMinAODSpacing || 
                                    (arrayInfo.lastNormalIndexACExcl - indexAC) * arrayInfo.spacingAC < maxMinOccDistMinAODSpacing))
                                {
                                    validParkingSite = false;
                                    break;
                                }
                            }
                        }
                    }
                    for(int lastParkingSiteIndexXC = indexXC - 1; lastParkingSiteIndexXC > 0 && 
                        (indexXC - lastParkingSiteIndexXC) * arrayInfo.spacingXC < maxMinOccDistMinAODSpacing; lastParkingSiteIndexXC--)
                    {
                        for(int& indexACLastParkingSite : arrayInfo.parkingSitesPerXCIndex[lastParkingSiteIndexXC])
                        {
                            int distanceAC = (indexACLastParkingSite - (int)indexAC) * arrayInfo.spacingAC;
                            int distanceXC = ((int)indexXC - lastParkingSiteIndexXC) * arrayInfo.spacingXC;
                            int distanceSq = distanceAC * distanceAC + distanceXC * distanceXC;
                            if(distanceSq < maxMinOccDistMinAODSpacing)
                            {
                                validParkingSite = false;
                            }
                        }
                    }
                }
                if(validParkingSite)
                {
                    arrayInfo.parkingSitesPerXCIndex[indexXC].push_back(indexAC);
                }
            }
            else if(targetGeometry(row, col) == TargetState::OCCUPIED)
            {
                arrayInfo.targetSitesPerXCIndex[indexXC].push_back(indexAC);
            }
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
                    arrayInfo.usableAtomsPerXCIndex[indexXC].push_back(indexAC);
                }
                else
                {
                    arrayInfo.unusableAtomsPerXCIndex[indexXC].push_back(indexAC);
                }
            }
        }
    }

    return std::move(arrayInfo);
}

// Move atoms from parking spots to empty target sites
bool resolveSortingDeficiencies(ArrayAccessor& stateArray, std::pair<int,int> startIndex, std::vector<ParallelMove>& moveList, 
    ArrayInformation& arrayInfo, std::shared_ptr<spdlog::logger> logger)
{    
    for(int i = 0; i < 2; i++)
    {
        if((i == 0 && std::get<1>(startIndex) < (int)arrayInfo.lastNormalIndexXCExcl) || (i == 1 && std::get<0>(startIndex) > (int)arrayInfo.firstNormalIndexXC))
        {
            int lastIndexXCToPossiblyContainsAtoms, targetIndexXC, lastTargetIndexXCExclusive, 
                targetIndexXCDir, lastIndexXCWithUsableAtoms;
            if(i == 0)
            {
                lastIndexXCWithUsableAtoms = arrayInfo.lastNormalIndexXCExcl - 1;
                lastIndexXCToPossiblyContainsAtoms = arrayInfo.firstNormalIndexXC;
                targetIndexXC = std::get<0>(startIndex) + arrayInfo.sortingChannelWidth + 1;
                lastTargetIndexXCExclusive = arrayInfo.lastNormalIndexXCExcl;
                targetIndexXCDir = 1;
            }
            else
            {
                lastIndexXCWithUsableAtoms = arrayInfo.firstNormalIndexXC;
                lastIndexXCToPossiblyContainsAtoms = arrayInfo.lastNormalIndexXCExcl - 1;
                targetIndexXC = std::get<0>(startIndex) + arrayInfo.sortingChannelWidth;
                lastTargetIndexXCExclusive = arrayInfo.firstNormalIndexXC - 1;
                targetIndexXCDir = -1;
            }
            for(; targetIndexXC != lastTargetIndexXCExclusive; targetIndexXC += targetIndexXCDir)
            {
                while(!arrayInfo.targetSitesPerXCIndex[targetIndexXC].empty())
                {
                    // In this version, only atoms parked in border regions can be used for filling vacancies,
                    // as other atoms may not be reachable through a straight path
                    bool containsUsableAtomsInFullyIrrelevantBorder = false;
                    for(int indexAC : arrayInfo.usableAtomsPerXCIndex[lastIndexXCWithUsableAtoms])
                    {
                        if(indexAC < (int)arrayInfo.firstRelevantAC || indexAC >= (int)arrayInfo.lastRelevantACExcl)
                        {
                            containsUsableAtomsInFullyIrrelevantBorder = true;
                        }
                    }
                    while(arrayInfo.usableAtomsPerXCIndex[lastIndexXCWithUsableAtoms].empty() || !containsUsableAtomsInFullyIrrelevantBorder)
                    {
                        if(lastIndexXCWithUsableAtoms == lastIndexXCToPossiblyContainsAtoms)
                        {
                            return false;
                        }
                        lastIndexXCWithUsableAtoms -= targetIndexXCDir;

                        // Check for new index whether usable atoms exist in border
                        containsUsableAtomsInFullyIrrelevantBorder = false;
                        for(int indexAC : arrayInfo.usableAtomsPerXCIndex[lastIndexXCWithUsableAtoms])
                        {
                            if(indexAC < (int)arrayInfo.firstRelevantAC || indexAC >= (int)arrayInfo.lastRelevantACExcl)
                            {
                                containsUsableAtomsInFullyIrrelevantBorder = true;
                            }
                        }
                    }

                    logger->debug("indexXC: {}", lastIndexXCWithUsableAtoms);
                    std::stringstream unusableAtomsStr;
                    unusableAtomsStr << "usableAtoms[lastIndexXCWithUsableAtoms]: ";
                    for(auto usableAtom : arrayInfo.usableAtomsPerXCIndex[lastIndexXCWithUsableAtoms])
                    {
                        unusableAtomsStr << usableAtom << ", ";
                    }
                    logger->debug(unusableAtomsStr.str());

                    double channelIndexXC = (targetIndexXC * targetIndexXCDir > 
                        lastIndexXCWithUsableAtoms * targetIndexXCDir) ? 
                        targetIndexXC : lastIndexXCWithUsableAtoms;
                    channelIndexXC += (double)(arrayInfo.sortingChannelWidth + 1) / 2. * (double)targetIndexXCDir;

                    if(channelIndexXC < (double)arrayInfo.firstNormalIndexXC || channelIndexXC > (double)arrayInfo.lastNormalIndexXCExcl)
                    {
                        logger->error("Sorting deficiencies could as path would lead through buffer region. Aborting");
                        return false;
                    }

                    while(!arrayInfo.targetSitesPerXCIndex[targetIndexXC].empty() && !arrayInfo.usableAtomsPerXCIndex[lastIndexXCWithUsableAtoms].empty())
                    {
                        std::vector<int>::iterator targetIndexAC = arrayInfo.targetSitesPerXCIndex[targetIndexXC].begin();
                        std::vector<int>::iterator usableAtomAC = arrayInfo.usableAtomsPerXCIndex[lastIndexXCWithUsableAtoms].begin();

                        // Create move
                        ParallelMove move;
                        ParallelMove::Step start, elbow1, elbow2, end;
                        std::vector<double> *startSelectionAC, *endSelectionAC;
                        if(arrayInfo.vertical)
                        {
                            start.colSelection.push_back(lastIndexXCWithUsableAtoms);
                            elbow1.colSelection.push_back(channelIndexXC);
                            elbow2.colSelection.push_back(channelIndexXC);
                            end.colSelection.push_back(targetIndexXC);
                            startSelectionAC = &start.rowSelection;
                            endSelectionAC = &end.rowSelection;
                        }
                        else
                        {
                            start.rowSelection.push_back(lastIndexXCWithUsableAtoms);
                            elbow1.rowSelection.push_back(channelIndexXC);
                            elbow2.rowSelection.push_back(channelIndexXC);
                            end.rowSelection.push_back(targetIndexXC);
                            startSelectionAC = &start.colSelection;
                            endSelectionAC = &end.colSelection;
                        }
                        unsigned int targetCount = arrayInfo.targetSitesPerXCIndex[targetIndexXC].size();
                        if(targetCount > arrayInfo.maxTonesAC)
                        {
                            targetCount = arrayInfo.maxTonesAC;
                        }

                        while(usableAtomAC != arrayInfo.usableAtomsPerXCIndex[lastIndexXCWithUsableAtoms].end() && 
                            targetIndexAC != arrayInfo.targetSitesPerXCIndex[targetIndexXC].end() && 
                            startSelectionAC->size() < targetCount)
                        {
                            while(usableAtomAC != arrayInfo.usableAtomsPerXCIndex[lastIndexXCWithUsableAtoms].end() && !startSelectionAC->empty() && 
                                ((*usableAtomAC - startSelectionAC->back()) * arrayInfo.spacingAC < Config::getInstance().minAodSpacing ||
                                (*usableAtomAC >= (int)arrayInfo.firstRelevantAC && *usableAtomAC < (int)arrayInfo.lastRelevantACExcl)))
                            {
                                usableAtomAC++;
                            }
                            while(targetIndexAC != arrayInfo.targetSitesPerXCIndex[targetIndexXC].end() && !endSelectionAC->empty() && 
                                (*targetIndexAC - endSelectionAC->back()) * arrayInfo.spacingAC < Config::getInstance().minAodSpacing)
                            {
                                targetIndexAC++;
                            }
                            if(usableAtomAC != arrayInfo.usableAtomsPerXCIndex[lastIndexXCWithUsableAtoms].end() && 
                                targetIndexAC != arrayInfo.targetSitesPerXCIndex[targetIndexXC].end())
                            {
                                startSelectionAC->push_back(*usableAtomAC);
                                endSelectionAC->push_back(*targetIndexAC);

                                usableAtomAC = arrayInfo.usableAtomsPerXCIndex[lastIndexXCWithUsableAtoms].erase(usableAtomAC);
                                targetIndexAC = arrayInfo.targetSitesPerXCIndex[targetIndexXC].erase(targetIndexAC);
                            }
                        }
                        if(startSelectionAC->empty())
                        {
                            logger->error("Sorting deficiencies could not be resolved. Consider wider borders. Aborting");
                            return false;
                        }
                        std::sort(startSelectionAC->begin(), startSelectionAC->end());
                        std::sort(endSelectionAC->begin(), endSelectionAC->end());
                        if(arrayInfo.vertical)
                        {
                            elbow1.rowSelection = start.rowSelection;
                            elbow2.rowSelection = end.rowSelection;
                        }
                        else
                        {
                            elbow1.colSelection = start.colSelection;
                            elbow2.colSelection = end.colSelection;
                        }
                        move.steps.push_back(std::move(start));
                        move.steps.push_back(std::move(elbow1));
                        move.steps.push_back(std::move(elbow2));
                        move.steps.push_back(std::move(end));
                        move.execute(stateArray, logger);
                        moveList.push_back(std::move(move));
                    }
                }
            }
        }
    }

    return true;
}

// Internal main sorting function
bool sortArray(ArrayAccessor& stateArray, pybind11::detail::unchecked_reference<TargetState, 2>& targetGeometry, 
    std::vector<ParallelMove>& moveList, std::shared_ptr<spdlog::logger> logger)
{
    // Differentiate between unusable (too close to each other) and usable atoms and add into per-index buffers
    std::optional<ArrayInformation> arrayInfo = conductInitialAnalysis(stateArray, targetGeometry, logger);
    if(!arrayInfo.has_value())
    {
        return false;
    }

    size_t totalUsableAtoms = 0;
    for(auto usableAtomsAtIndex : arrayInfo.value().usableAtomsPerXCIndex)
    {
        totalUsableAtoms += usableAtomsAtIndex.size();
    }
    size_t totalTargetSites = 0;
    for(auto targetSitesAtIndex : arrayInfo.value().targetSitesPerXCIndex)
    {
        totalTargetSites += targetSitesAtIndex.size();
    }
    logger->info("Usable atoms: {}, Target sites: {}", totalUsableAtoms, totalTargetSites);

    // Determine were to start sorting. May either be from one side and iterate one way over the array or 
    // from the middle and outward simultaneously
    auto startingPosition = determineBestStartPosition(stateArray, moveList, arrayInfo.value());
    if(startingPosition.has_value())
    {
        logger->debug("Best starting position: {} - {}", std::get<0>(startingPosition.value()), std::get<1>(startingPosition.value()));
    }
    else
    {
        logger->error("Best starting position could not be determined!");
        return false;
    }

    // Clear buffer and dumping zones
    if(!clearBufferAndDumpingIndicesXC(stateArray, moveList, arrayInfo.value(), logger))
    {
        return false;
    }

    // Remove atoms from starting sorting channel
    clearChannel(stateArray, std::get<0>(startingPosition.value()), std::get<1>(startingPosition.value()), 
        false, moveList, arrayInfo.value(), logger);

    // Call main iterating function that sorts atoms row-by-row through sorting channel
    if(!sortRemainingRowsOrCols(stateArray, startingPosition.value(), moveList, arrayInfo.value(), logger))
    {
        // Fill remaining positions from parked atoms
        if(!resolveSortingDeficiencies(stateArray, startingPosition.value(), moveList, arrayInfo.value(), logger))
        {
            logger->error("Array could not be sorted. Aborting");
            return false;
        }
    }

    if(Config::getInstance().alwaysGenerateAllAODTones)
    {
        for(auto& move : moveList)
        {
            move.extendToUseAllTones(stateArray.rows(), stateArray.cols(), logger, false, 
                arrayInfo.value().bufferRows, arrayInfo.value().bufferCols);
        }
    }

    return true;
}

bool checkMoveListValidity(ArrayAccessor& stateArray, std::vector<ParallelMove>& moveList, std::shared_ptr<spdlog::logger> logger)
{
    for(size_t moveIndex = 0; moveIndex < moveList.size(); moveIndex++)
    {
        ParallelMove& move = moveList[moveIndex];
        for(ParallelMove::Step& step : move.steps)
        {
            for(size_t i = 0; i < step.rowSelection.size(); i++)
            {
                if(step.rowSelection[i] < 0 || step.rowSelection[i] > stateArray.rows() - 1)
                {
                    logger->error("Post-Check: Row frequency out of bounds");
                    return false;
                }
                if(i < step.rowSelection.size() - 1)
                {
                    if((step.rowSelection[i + 1] - step.rowSelection[i]) * Config::getInstance().rowSpacing < Config::getInstance().minAodSpacing)
                    {
                        std::stringstream startCols;
                        std::stringstream endCols;
                        std::stringstream startRows;
                        std::stringstream endRows;
                        for(const auto& col : move.steps[0].colSelection)
                        {
                            startCols << col << " ";
                        }
                        for(const auto& col : move.steps.back().colSelection)
                        {
                            endCols << col << " ";
                        }
                        for(const auto& row : move.steps[0].rowSelection)
                        {
                            startRows << row << " ";
                        }
                        for(const auto& row : move.steps.back().rowSelection)
                        {
                            endRows << row << " ";
                        }
                        logger->error("Post-Check: Two row tones in move are not ordered correctly or spaced too close together to be allowed in move {} ({})->({}), cols: ({})->({}). High minimum AOD spacings are currently not yet supported.", 
                            moveIndex, startRows.str(), endRows.str(), startCols.str(), endCols.str());
                        return false;
                    }
                }
            }
            for(size_t i = 0; i < step.colSelection.size(); i++)
            {
                if(step.colSelection[i] < 0 || step.colSelection[i] > stateArray.cols() - 1)
                {
                    logger->error("Post-Check: Col frequency out of bounds");
                    return false;
                }
                if(i < step.colSelection.size() - 1)
                {
                    if((step.colSelection[i + 1] - step.colSelection[i]) * Config::getInstance().columnSpacing < Config::getInstance().minAodSpacing)
                    {
                        std::stringstream startCols;
                        std::stringstream endCols;
                        std::stringstream startRows;
                        std::stringstream endRows;
                        for(const auto& col : move.steps[0].colSelection)
                        {
                            startCols << col << " ";
                        }
                        for(const auto& col : move.steps.back().colSelection)
                        {
                            endCols << col << " ";
                        }
                        for(const auto& row : move.steps[0].rowSelection)
                        {
                            startRows << row << " ";
                        }
                        for(const auto& row : move.steps.back().rowSelection)
                        {
                            endRows << row << " ";
                        }
                        logger->error("Post-Check: Two col tones in move are not ordered correctly or spaced too close together to be allowed in move {} ({})->({}), cols: ({})->({}). High minimum AOD spacings are currently not yet supported.", 
                            moveIndex, startRows.str(), endRows.str(), startCols.str(), endCols.str());
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

// Access function to be bound, Eigen array act as interfaces as they can act on Python array data without reallocation
std::optional<std::vector<ParallelMove>> sortLatticeByRowParallel(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &stateArray, 
    const py::array_t<TargetState>& targetGeometry)
{
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
    if(Config::getInstance().alwaysGenerateAllAODTones && 
        Config::getInstance().aodColLimit * Config::getInstance().aodRowLimit > Config::getInstance().aodTotalLimit)
    {
        logger->error("If all tones should always be generated, then aodTotalLimit needs to allow for the full row and column limit. Aborting");
        return std::nullopt;
    }

    // Log initial state
    if(logger->level() <= spdlog::level::info)
    {
        std::stringstream strstream;
        strstream << "Initial state: should be occupied █/▒, should not be occupied ●/○, irrelevant X/\" \"\n";
        for(size_t r = 0; r < (size_t)stateArray.rows(); r++)
        {
            for(size_t c = 0; c < (size_t)stateArray.cols(); c++)
            {
                if(targetGeometry.at(r, c) == TargetState::OCCUPIED)
                {
                    strstream << (stateArray(r,c) ? "█" : "▒");
                }
                else if(targetGeometry.at(r, c) == TargetState::EMPTY)
                {
                    strstream << (stateArray(r,c) ? "●" : "○");
                }
                else
                {
                    strstream << (stateArray(r,c) ? "X" : " ");
                }
            }
            strstream << "\n";
        }
        logger->info(strstream.str());
    }

    // Actual sorting call
    EigenArrayAccessor eigenStateArray(stateArray);
    std::vector<ParallelMove> moveList;
    if(!sortArray(eigenStateArray, targetGeometryUnchecked, moveList, logger))
    {
        return std::nullopt;
    }

    if(!checkMoveListValidity(eigenStateArray, moveList, logger))
    {
        return std::nullopt;
    }

    // Log final state if sorting was successful
    if(logger->level() <= spdlog::level::info)
    {
        std::stringstream endstrstream;
        endstrstream << "Final state: \n";
        for(size_t r = 0; r < (size_t)stateArray.rows(); r++)
        {
            for(size_t c = 0; c < (size_t)stateArray.cols(); c++)
            {
                    if(targetGeometry.at(r, c) == TargetState::OCCUPIED)
                    {
                        endstrstream << (stateArray(r,c) ? "█" : "▒");
                    }
                    else if(targetGeometry.at(r, c) == TargetState::EMPTY)
                    {
                        endstrstream << (stateArray(r,c) ? "●" : "○");
                    }
                    else
                    {
                        endstrstream << (stateArray(r,c) ? "X" : " ");
                    }
            }
            endstrstream << "\n";
        }
        logger->info(endstrstream.str());
    }

    return moveList;
}