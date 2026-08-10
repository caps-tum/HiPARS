#include "sortLattice.hpp"

Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> generateMask(double distance, double spacingFraction)
{
    int maskRowDist = floor((double)distance / (Config::getInstance().rowSpacing * spacingFraction));
    if(maskRowDist < 1)
    {
        maskRowDist = 0;
    }
    int maskRows = 2 * maskRowDist + 1;
    int maskColDist = floor((double)distance / (Config::getInstance().columnSpacing * spacingFraction));
    if(maskColDist < 1)
    {
        maskColDist = 0;
    }
    int maskCols = 2 * maskColDist + 1;
    Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> mask(maskRows, maskCols);
    
    for(int r = 0; r < maskRows; r++)
    {
        for(int c = 0; c < maskCols; c++)
        {
            mask(r,c) = pythagorasDist((r - maskRowDist) * (Config::getInstance().rowSpacing * spacingFraction), 
                (c - maskColDist) * (Config::getInstance().columnSpacing * spacingFraction)) < distance;
        }
    }

    return mask;
}

Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic> generatePathway(size_t borderRows, size_t borderCols, 
    ArrayAccessor& occupancy, double distFromOcc, double distFromEmpty, 
    std::optional<size_t> minRow, std::optional<size_t> maxRow, std::optional<size_t> minCol, std::optional<size_t> maxCol)
{
    auto occMask = generateMask(distFromOcc, 0.5);
    Eigen::Index halfOccRows = occMask.rows() / 2;
    Eigen::Index halfOccCols = occMask.cols() / 2;
    auto emptyMask = generateMask(distFromEmpty, 0.5);
    Eigen::Index halfEmptyRows = emptyMask.rows() / 2;
    Eigen::Index halfEmptyCols = emptyMask.cols() / 2;

    if(borderRows < (size_t)halfOccRows)
    {
        borderRows = halfOccRows;
    }
    if(borderRows < (size_t)halfEmptyRows)
    {
        borderRows = halfEmptyRows;
    }
    if(borderCols < (size_t)halfOccCols)
    {
        borderCols = halfOccCols;
    }
    if(borderCols < (size_t)halfEmptyCols)
    {
        borderCols = halfEmptyCols;
    }

    size_t pathwayRows = 2 * occupancy.rows() - 1 + 2 * borderRows;
    size_t pathwayCols = 2 * occupancy.cols() - 1 + 2 * borderCols;

    Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic> pathway = 
        Eigen::Array<unsigned int,Eigen::Dynamic,Eigen::Dynamic>::Zero(pathwayRows, pathwayCols);

    if(!minRow.has_value())
    {
        minRow = 0;
    }
    if(!maxRow.has_value())
    {
        maxRow = occupancy.rows();
    }
    if(!minRow.has_value())
    {
        minCol = 0;
    }
    if(!maxRow.has_value())
    {
        maxCol = occupancy.cols();
    }

    for(size_t r = minRow.value(); r < maxRow.value(); r++)
    {
        for(size_t c = minCol.value(); c < maxCol.value(); c++)
        {
            if(occupancy(r,c))
            {
                pathway(Eigen::seqN(2 * r + borderRows - halfOccRows, occMask.rows()), 
                    Eigen::seqN(2 * c + borderCols - halfOccCols, occMask.cols())) += occMask.cast<unsigned int>();
            }
            else
            {
                pathway(Eigen::seqN(2 * r + borderRows - halfEmptyRows, emptyMask.rows()), 
                    Eigen::seqN(2 * c + borderCols - halfEmptyCols, emptyMask.cols())) += emptyMask.cast<unsigned int>();
            }
        }
    }

    return pathway;
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
    int maxIrrelevantRowLow = -1, maxIrrelevantColLow = -1;
    unsigned int minIrrelevantRowHigh = stateArray.rows(), minIrrelevantColHigh = stateArray.cols();
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
    unsigned int firstNormalRow = 0, lastNormalRowExcl = stateArray.rows(), firstNormalCol = 0, lastNormalColExcl = stateArray.cols();
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
        for(int nextHighestRow = floor(currentBufferRowHigh); nextHighestRow >= (int)minIrrelevantRowHigh && 
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
        for(int nextHighestCol = floor(currentBufferColHigh); nextHighestCol >= (int)minIrrelevantColHigh && 
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

    arrayInfo.normalIndices = {firstNormalRow, lastNormalRowExcl, firstNormalCol, lastNormalColExcl};

    if(arrayInfo.vertical)
    {
        arrayInfo.normalIndicesXCAC = {firstNormalCol, lastNormalColExcl, firstNormalRow, lastNormalRowExcl};
        arrayInfo.relevantIndicesXCAC = {(unsigned int)(maxIrrelevantColLow + 1), minIrrelevantColHigh, (unsigned int)(maxIrrelevantRowLow + 1), minIrrelevantRowHigh};
    }
    else
    {
        arrayInfo.normalIndicesXCAC = {firstNormalRow, lastNormalRowExcl, firstNormalCol, lastNormalColExcl};
        arrayInfo.relevantIndicesXCAC = {(unsigned int)(maxIrrelevantRowLow + 1), minIrrelevantRowHigh, (unsigned int)(maxIrrelevantColLow + 1), minIrrelevantColHigh};
    }

    // Iterate over array, check for usability-preventing neighbors, and sort into structure accordingly
    for(size_t row = firstNormalRow; row < lastNormalRowExcl && row < stateArray.rows(); row++)
    {
        for(size_t col = firstNormalCol; col < lastNormalColExcl && col < stateArray.cols(); col++)
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
                                    (indexXC - arrayInfo.normalIndicesXCAC.firstRowOrXC + 1) * arrayInfo.spacingXC < maxMinOccDistMinAODSpacing || 
                                    (arrayInfo.normalIndicesXCAC.lastRowOrXCExcl - indexXC) * arrayInfo.spacingXC < maxMinOccDistMinAODSpacing || 
                                    (indexAC - arrayInfo.normalIndicesXCAC.firstColOrAC + 1) * arrayInfo.spacingAC < maxMinOccDistMinAODSpacing || 
                                    (arrayInfo.normalIndicesXCAC.lastColOrACExcl - indexAC) * arrayInfo.spacingAC < maxMinOccDistMinAODSpacing))
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