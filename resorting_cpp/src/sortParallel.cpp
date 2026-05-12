#include "sortParallel.hpp"

#include <optional>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cfloat>
#include <set>
#include <omp.h>
#include <chrono>

#include "config.hpp"

void moveMoveToSortedMoveListIfUseful(std::vector<std::tuple<ParallelMove,int,double>>& moveList, std::vector<size_t>& moveRanking, 
    ParallelMove& move, int correctedTargetSites, double costPerCorrectedTargetSite)
{
    if(!moveList.empty() && !moveRanking.empty())
    {
        if(costPerCorrectedTargetSite <= std::get<2>(moveList[moveRanking[0]]) / BENEFIT_FRACTION_TO_ALSO_EXECUTE)
        {
            size_t insertPosition = moveRanking.size();
            for(size_t nthBestMove = 0; nthBestMove < moveRanking.size(); nthBestMove++)
            {
                if(costPerCorrectedTargetSite <= std::get<2>(moveList[moveRanking[nthBestMove]]))
                {
                    insertPosition = nthBestMove;
                    break;
                }
            }
            double bestCostPerCorrectedTargetSite = insertPosition == 0 ? costPerCorrectedTargetSite : 
                std::get<2>(moveList[moveRanking[0]]);
            if(insertPosition < moveRanking.size() && std::get<2>(moveList[moveRanking.back()]) > 
                bestCostPerCorrectedTargetSite / BENEFIT_FRACTION_TO_ALSO_EXECUTE)
            {
                // Previously worst is removed as it is not good enough
                moveList[moveRanking.back()] = std::tuple(std::move(move), correctedTargetSites, costPerCorrectedTargetSite);
            }
            else
            {
                // Keep all other moves
                moveList.push_back(std::tuple(std::move(move), correctedTargetSites, costPerCorrectedTargetSite));
                moveRanking.push_back(moveList.size() - 1);
            }
            size_t insertIndex = moveRanking.back();
            for(size_t nthBestMove = moveRanking.size() - 1; nthBestMove > insertPosition; nthBestMove--)
            {
                moveRanking[nthBestMove] = moveRanking[nthBestMove - 1];
            }
            moveRanking[insertPosition] = insertIndex;
        }
    }
    else
    {
        moveList.push_back(std::tuple(std::move(move), correctedTargetSites, costPerCorrectedTargetSite));
        moveRanking.push_back(0);
    }
}

std::optional<int> calcCorrectedTargetSite(const ParallelMove& move, ArrayAccessor& stateArray, 
    size_t compZone[4], std::shared_ptr<spdlog::logger> logger,
    std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> alreadyMoved, 
    ArrayAccessor& targetGeometry)
{
    int correctedTargetSites = 0;
    for(size_t rowIndex = 0; rowIndex < move.steps[0].rowSelection.size(); rowIndex++)
    {
        size_t startRow = roundCoordDown(move.steps[0].rowSelection[rowIndex]);
        size_t endRow = roundCoordDown(move.steps.back().rowSelection[rowIndex]);
        for(size_t colIndex = 0; colIndex < move.steps[0].colSelection.size(); colIndex++)
        {
            size_t startCol = roundCoordDown(move.steps[0].colSelection[colIndex]);
            size_t endCol = roundCoordDown(move.steps.back().colSelection[colIndex]);
            if(stateArray(startRow, startCol))
            {
                if(stateArray(endRow, endCol) || (!Config::getInstance().allowMultipleMovesPerAtom && 
                    alreadyMoved.has_value() && alreadyMoved.value()(startRow, startCol)))
                {
                    return std::nullopt;
                }
                else
                {
                    if(isInCompZone(startRow, startCol, compZone[0], compZone[1], compZone[2], compZone[3]))
                    {
                        if(targetGeometry(startRow - compZone[0], startCol - compZone[2]))
                        {
                            correctedTargetSites--;
                        }
                        else
                        {
                            correctedTargetSites++;
                        }
                    }
                    if(isInCompZone(endRow, endCol, compZone[0], compZone[1], compZone[2], compZone[3]) && 
                        targetGeometry(endRow - compZone[0], endCol - compZone[2]))
                    {
                        correctedTargetSites++;
                    }
                }
            }
            else if(!Config::getInstance().allowMovingEmptyTrapOntoOccupied && stateArray(endRow, endCol))
            {
                return std::nullopt;
            }
        }
    }
    return correctedTargetSites;
}

bool matchable(const std::vector<int>& vec1, const std::vector<int>& vec2, int maxDist)
{
    bool vec1Smaller = vec1.size() < vec2.size();
    const std::vector<int> *smallerVec, *largerVec;
    std::vector<int>::const_iterator smallerVecIterator, largerVecIterator;

    if(vec1Smaller)
    {
        smallerVec = &vec1;
        smallerVecIterator = vec1.begin();
        largerVec = &vec2;
        largerVecIterator = vec2.begin();
    }
    else
    {
        smallerVec = &vec2;
        smallerVecIterator = vec2.begin();
        largerVec = &vec1;
        largerVecIterator = vec1.begin();
    }

    while(smallerVecIterator != smallerVec->end() && largerVecIterator != largerVec->end()) 
    {
        if(abs(*smallerVecIterator - *largerVecIterator) <= maxDist)
        {
            smallerVecIterator++;
            largerVecIterator++;
        }
        else
        {
            largerVecIterator++;
        }
    }

    return smallerVecIterator == smallerVec->end();
}

std::vector<std::pair<int,int>> getMostPairings(const std::vector<int>& vec1, const std::vector<int>& vec2, int maxDist)
{
    bool vec1Smaller = vec1.size() < vec2.size();
    const std::vector<int> *smallerVec, *largerVec;
    std::vector<int>::const_iterator smallerVecIterator, largerVecIterator, lastUnusedLargerVecIterator;
    std::vector<std::pair<int,int>> pairing;

    if(vec1Smaller)
    {
        smallerVec = &vec1;
        smallerVecIterator = vec1.begin();
        largerVec = &vec2;
        largerVecIterator = vec2.begin();
        lastUnusedLargerVecIterator = vec2.begin();
    }
    else
    {
        smallerVec = &vec2;
        smallerVecIterator = vec2.begin();
        largerVec = &vec1;
        largerVecIterator = vec1.begin();
        lastUnusedLargerVecIterator = vec1.begin();
    }

    while(smallerVecIterator != smallerVec->end() && largerVecIterator != largerVec->end()) 
    {
        int dist = *largerVecIterator - *smallerVecIterator;
        if(dist < -maxDist)
        {
            largerVecIterator++;
        }
        else if(dist <= maxDist)
        {
            if(vec1Smaller)
            {
                pairing.push_back(std::pair(*smallerVecIterator, *largerVecIterator));
            }
            else
            {
                pairing.push_back(std::pair(*largerVecIterator, *smallerVecIterator));
            }
            smallerVecIterator++;
            largerVecIterator++;
            lastUnusedLargerVecIterator = largerVecIterator;
        }
        else
        {
            smallerVecIterator++;
            largerVecIterator = lastUnusedLargerVecIterator;
        }
    }

    return pairing;
}

std::optional<std::vector<std::pair<int,int>>> getPairings(const std::vector<int>& vec1, const std::vector<int>& vec2, int maxDist)
{
    bool vec1Smaller = vec1.size() < vec2.size();
    const std::vector<int> *smallerVec, *largerVec;
    std::vector<int>::const_iterator smallerVecIterator, largerVecIterator;
    std::vector<std::pair<int,int>> pairing;

    if(vec1Smaller)
    {
        smallerVec = &vec1;
        smallerVecIterator = vec1.begin();
        largerVec = &vec2;
        largerVecIterator = vec2.begin();
    }
    else
    {
        smallerVec = &vec2;
        smallerVecIterator = vec2.begin();
        largerVec = &vec1;
        largerVecIterator = vec1.begin();
    }

    while(smallerVecIterator != smallerVec->end() && largerVecIterator != largerVec->end()) 
    {
        if(abs(*smallerVecIterator - *largerVecIterator) <= maxDist)
        {
            if(vec1Smaller)
            {
                pairing.push_back(std::pair(*smallerVecIterator, *largerVecIterator));
            }
            else
            {
                pairing.push_back(std::pair(*largerVecIterator, *smallerVecIterator));
            }
            smallerVecIterator++;
            largerVecIterator++;
        }
        else
        {
            largerVecIterator++;
        }
    }

    return pairing;
}

std::optional<std::vector<std::pair<int,int>>> bottleneckAssignmentMinimizeMaxDist(
    const std::vector<int>& vec1, const std::vector<int>& vec2)
{
    int minDist = 0;
    int maxDist = abs(vec1.back() - vec2.front());
    if(abs(vec2.back() - vec1.front()) > maxDist)
    {
        maxDist = abs(vec2.back() - vec1.front());
    }

    // Binary search over distance
    while(minDist < maxDist) 
    {
        int avgDist = (minDist + maxDist) / 2;
        if(matchable(vec1, vec2, avgDist))
        {
            maxDist = avgDist;
        } 
        else 
        {
            minDist = avgDist + 1;
        }
    }

    return getPairings(vec1, vec2, minDist);
}

int bottleneckAssignmentMinimizeMaxDistGetDist(
    const std::vector<int>& vec1, const std::vector<int>& vec2)
{
    int minDist = 0;
    int maxDist = abs(vec1.back() - vec2.front());
    if(abs(vec2.back() - vec1.front()) > maxDist)
    {
        maxDist = abs(vec2.back() - vec1.front());
    }

    // Binary search over distance
    while(minDist < maxDist) 
    {
        int avgDist = (minDist + maxDist) / 2;
        if(matchable(vec1, vec2, avgDist))
        {
            maxDist = avgDist;
        } 
        else 
        {
            minDist = avgDist + 1;
        }
    }

    return minDist;
}

bool orderedDblVecContainsElem(const std::vector<double>& vec, double elem)
{
    const auto& firstOccurrence = std::lower_bound(vec.begin(), vec.end(), elem - DOUBLE_EQUIVALENCE_THRESHOLD);
    if(firstOccurrence == vec.end())
    {
        return false;
    }
    else
    {
        return abs(*firstOccurrence - elem) < DOUBLE_EQUIVALENCE_THRESHOLD;
    }
}

bool& accessArrayDim(ArrayAccessor& stateArray,
    size_t i, size_t j, bool rowFirst)
{
    if(rowFirst)
    {
        return stateArray(i,j);
    }
    else
    {
        return stateArray(j,i);
    }
}

void fillDimensionDependantData(ArrayAccessor& stateArray, 
    size_t compZone[4], bool rowFirst, size_t outerDimCompZone[2], size_t innerDimCompZone[2],
    size_t& outerSize, size_t& innerSize, unsigned int& outerAODLimit, unsigned int& innerAODLimit)
{
    if(rowFirst)
    {
        outerDimCompZone[0] = compZone[0];
        outerDimCompZone[1] = compZone[1];
        innerDimCompZone[0] = compZone[2];
        innerDimCompZone[1] = compZone[3];
        outerSize = stateArray.rows();
        innerSize = stateArray.cols();
        outerAODLimit = Config::getInstance().aodRowLimit;
        innerAODLimit = Config::getInstance().aodColLimit;
    }
    else
    {
        outerDimCompZone[0] = compZone[2];
        outerDimCompZone[1] = compZone[3];
        innerDimCompZone[0] = compZone[0];
        innerDimCompZone[1] = compZone[1];
        outerSize = stateArray.cols();
        innerSize = stateArray.rows();
        outerAODLimit = Config::getInstance().aodColLimit;
        innerAODLimit = Config::getInstance().aodRowLimit;
    }
}

double approxCostPerMove(double dist1, double dist2)
{
    if(Config::getInstance().allowDiagonalMovement)
    {
        if(dist1 <= 1 + DOUBLE_EQUIVALENCE_THRESHOLD && dist2 <= 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            if(dist1 >= 1 - DOUBLE_EQUIVALENCE_THRESHOLD && dist2 >= 1 - DOUBLE_EQUIVALENCE_THRESHOLD)
            {
                return Config::getInstance().moveCostOffset + Config::getInstance().moveCostOffsetSubmove + 
                    Config::getInstance().moveCostScalingSqrt * M_4TH_ROOT_2 + Config::getInstance().moveCostScalingLinear * M_SQRT2;
            }
            else
            {
                // If not both are 1 then at least one has to be 0 so dist1+dist2 = sqrt(dist1^2 + dist2^2)
                return Config::getInstance().moveCostOffset + costPerSubMove(dist1 + dist2);
            }
        }
        else
        {
            // 2 * half diagonal step cost
            double cost = Config::getInstance().moveCostOffset + 2 * (Config::getInstance().moveCostOffsetSubmove + 
                Config::getInstance().moveCostScalingSqrt * M_4TH_ROOT_1_2 + Config::getInstance().moveCostScalingLinear * M_SQRT1_2);
            if(dist1 > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
            {
                cost += costPerSubMove(dist1 - 1);
            }
            if(dist2 > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
            {
                cost += costPerSubMove(dist2 - 1);
            }
            return cost;
        }
    }
    else
    {
        if(dist1 < 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            return Config::getInstance().moveCostOffset + 2 * (Config::getInstance().moveCostOffsetSubmove + 
                Config::getInstance().moveCostScalingSqrt * M_SQRT1_2 + Config::getInstance().moveCostScalingLinear / 2) + costPerSubMove(dist2);
        }
        else if(dist2 < 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            return Config::getInstance().moveCostOffset + 2 * (Config::getInstance().moveCostOffsetSubmove + 
                Config::getInstance().moveCostScalingSqrt * M_SQRT1_2 + Config::getInstance().moveCostScalingLinear / 2) + costPerSubMove(dist1);
        }
        else
        {
            return Config::getInstance().moveCostOffset + 2 * (Config::getInstance().moveCostOffsetSubmove + 
                Config::getInstance().moveCostScalingSqrt * M_SQRT1_2 + Config::getInstance().moveCostScalingLinear / 2) + 
                costPerSubMove(dist1 - 0.5) + costPerSubMove(dist2 - 0.5);
        }
    }
}

bool fillSteps(ArrayAccessor& stateArray, 
    ParallelMove::Step& start, ParallelMove::Step& end, ParallelMove::Step& step1, ParallelMove::Step& step2, 
    bool rowFirst, double& maxDist, std::shared_ptr<spdlog::logger> logger)
{
    bool completelyDirectMove = true;
    std::vector<double> *step1Selection, *step2Selection, *selStart, *selEnd, *otherStart, *otherEnd;
    if(rowFirst)
    {
        step1Selection = &step1.rowSelection;
        step2Selection = &step2.rowSelection;
        selStart = &start.rowSelection;
        selEnd = &end.rowSelection;
        otherStart = &start.colSelection;
        otherEnd = &end.colSelection;
    }
    else
    {
        step1Selection = &step1.colSelection;
        step2Selection = &step2.colSelection;
        selStart = &start.colSelection;
        selEnd = &end.colSelection;
        otherStart = &start.rowSelection;
        otherEnd = &end.rowSelection;
    }
    for(size_t i = 0; i < selStart->size(); i++)
    {
        double startRow = (*selStart)[i];
        double endRow = (*selEnd)[i];
        if(endRow > startRow + DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            completelyDirectMove = false;
            (*step1Selection)[i] += 0.5;
            (*step2Selection)[i] -= 0.5;
        }
        else if(endRow < startRow - DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            completelyDirectMove = false;
            (*step1Selection)[i] -= 0.5;
            (*step2Selection)[i] += 0.5;
        }
        else
        {
            bool directMove = true;

            for(size_t j = 0; j < otherStart->size() && directMove; j++)
            {
                size_t row = roundCoordDown(startRow);
                double col = (*otherStart)[j];

                double endCol = (*otherEnd)[j];
                int colStep = abs(endCol - col) > DOUBLE_EQUIVALENCE_THRESHOLD ? (signbit(endCol - col) ? -1 : 1) : 0;
                col += colStep;

                // If either has at least one more step to go (the other will be zero)
                for(; (endCol - col) * colStep >= 1 - DOUBLE_EQUIVALENCE_THRESHOLD; col += colStep)
                {
                    if(accessArrayDim(stateArray, row, roundCoordDown(col), rowFirst) && 
                        !orderedDblVecContainsElem(*otherStart, col))
                    {
                        directMove = false;
                        break;
                    }
                }
            }
            if(!directMove)
            {
                completelyDirectMove = false;
                (*step1Selection)[i] += 0.5;
                (*step2Selection)[i] += 0.5;
            }
        }
        double dist = abs(endRow - startRow);
        if(dist > maxDist)
        {
            maxDist = dist;
        }
    }
    return completelyDirectMove;
}

ParallelMove ParallelMove::fromStartAndEnd(
    ArrayAccessor& stateArray, ParallelMove::Step start, ParallelMove::Step end, std::shared_ptr<spdlog::logger> logger)
{
    ParallelMove move;
    ParallelMove::Step step1;
    ParallelMove::Step step2;

    step1.rowSelection = start.rowSelection;
    step1.colSelection = start.colSelection;
    step2.rowSelection = end.rowSelection;
    step2.colSelection = end.colSelection;

    double maxRowDist = 0;
    double maxColDist = 0;

    bool completelyDirectMove = fillSteps(stateArray, start, end, step1, step2, true, maxRowDist, logger);
    if(!completelyDirectMove)
    {
        completelyDirectMove = fillSteps(stateArray, start, end, step1, step2, false, maxColDist, logger);
    }
       
    move.steps.push_back(start);
    if(!completelyDirectMove && (!Config::getInstance().allowDiagonalMovement ||
        (maxRowDist > 1 + DOUBLE_EQUIVALENCE_THRESHOLD || maxColDist > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)))
    {
        bool require4Steps = maxRowDist > 1 + DOUBLE_EQUIVALENCE_THRESHOLD && maxColDist > 1 + DOUBLE_EQUIVALENCE_THRESHOLD;
        if(!Config::getInstance().allowDiagonalMovement)
        {
            if(maxRowDist > maxColDist)
            {
                step1.rowSelection = start.rowSelection;
                if(require4Steps)
                {
                    step2.colSelection = end.colSelection;
                }
                else
                {
                    step2.rowSelection = end.rowSelection;
                }
            }
            else
            {
                step1.colSelection = start.colSelection;
                if(require4Steps)
                {
                    step2.rowSelection = end.rowSelection;
                }
                else
                {
                    step2.colSelection = end.colSelection;
                }
            }
        }
        move.steps.push_back(step1);
        if(require4Steps)
        {
            ParallelMove::Step intermediateStep;
            if(maxRowDist > maxColDist)
            {
                intermediateStep.rowSelection = step2.rowSelection;
                intermediateStep.colSelection = step1.colSelection;
            }
            else
            {
                intermediateStep.rowSelection = step1.rowSelection;
                intermediateStep.colSelection = step2.colSelection;
            }
            move.steps.push_back(std::move(intermediateStep));
        }
        move.steps.push_back(std::move(step2));
    }
    move.steps.push_back(end);
    return move;
}

double ParallelMove::cost() const
{
    double cost = Config::getInstance().moveCostOffset;
    const ParallelMove::Step *lastStep = &this->steps[0];
    for(size_t step = 1; step < this->steps.size(); step++)
    {
        const ParallelMove::Step *newStep = &this->steps[step];
        double longestColDist = 0;
        for(size_t tone = 0; tone < lastStep->colSelection.size() && tone < newStep->colSelection.size(); tone++)
        {
            double dist = abs((double)(lastStep->colSelection[tone]) - (double)(newStep->colSelection[tone]));
            if(dist > longestColDist)
            {
                longestColDist = dist;
            }
        }
        double longestRowDist = 0;
        for(size_t tone = 0; tone < lastStep->rowSelection.size() && tone < newStep->rowSelection.size(); tone++)
        {
            double dist = abs(lastStep->rowSelection[tone] - newStep->rowSelection[tone]);
            if(dist > longestRowDist)
            {
                longestRowDist = dist;
            }
        }
        if(longestColDist == 0 || longestRowDist == 0)
        {
            cost += costPerSubMove(longestRowDist + longestColDist);
        }
        else if(abs(longestColDist - longestRowDist) < DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            cost += costPerSubMove(longestRowDist * M_SQRT2);
        }
        else
        {
            cost += costPerSubMove(sqrt(longestRowDist * longestRowDist + longestColDist * longestColDist));
        }
        lastStep = newStep;
    }
    return cost;
}

bool ParallelMove::execute(ArrayAccessor& stateArray, std::shared_ptr<spdlog::logger> logger,
    std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> alreadyMoved, double minDist) const
{
    const auto& firstStep = this->steps.front();
    const auto& lastStep = this->steps.back();

    if(logger->level() <= spdlog::level::info)
    {
        std::stringstream startCols;
        std::stringstream endCols;
        std::stringstream startRows;
        std::stringstream endRows;

        for(const auto& col : firstStep.colSelection)
        {
            startCols << col << " ";
        }
        for(const auto& col : lastStep.colSelection)
        {
            endCols << col << " ";
        }
        for(const auto& row : firstStep.rowSelection)
        {
            startRows << row << " ";
        }
        for(const auto& row : lastStep.rowSelection)
        {
            endRows << row << " ";
        }
        logger->info("Executing move, rows: ({})->({}), cols: ({})->({})", 
            startRows.str(), endRows.str(), startCols.str(), endCols.str());
    }

    if(firstStep.colSelection.size() != lastStep.colSelection.size() || firstStep.rowSelection.size() != lastStep.rowSelection.size())
    {
        logger->error("Move does not have equally many tones at beginning and end");
        return false;
    }
    if(firstStep.colSelection.size() == 0 || firstStep.rowSelection.size() == 0)
    {
        logger->error("No tone selected!");
        return false;
    }

    for(const auto& step : this->steps)
    {
        double lastTone = step.rowSelection[0];
        for(size_t i = 1; i < step.rowSelection.size(); i++)
        {
            if(step.rowSelection[i] < lastTone + minDist)
            {
                logger->error("Move tones not in correct order! {} < {}", step.rowSelection[i], lastTone + minDist);
                return false;
            }
            lastTone = step.rowSelection[i];
        }
        lastTone = step.colSelection[0];
        for(size_t i = 1; i < step.colSelection.size(); i++)
        {
            if(step.colSelection[i] < lastTone + minDist)
            {
                logger->error("Move tones not in correct order! {} < {}", step.colSelection[i], lastTone + minDist);
                return false;
            }
            lastTone = step.colSelection[i];
        }
    }

    std::vector<std::tuple<double,double>> removalQueue;
    std::vector<std::tuple<double,double>> addQueue;
    double rowMax = (double)stateArray.rows() - 1 + DOUBLE_EQUIVALENCE_THRESHOLD;
    double colMax = (double)stateArray.cols() - 1 + DOUBLE_EQUIVALENCE_THRESHOLD;
    for(size_t rowTone = 0; rowTone < firstStep.rowSelection.size(); rowTone++)
    {
        for(size_t colTone = 0; colTone < firstStep.colSelection.size(); colTone++)
        {
            if(firstStep.rowSelection[rowTone] >= -DOUBLE_EQUIVALENCE_THRESHOLD && 
                firstStep.rowSelection[rowTone] <= rowMax && 
                firstStep.colSelection[colTone] >= -DOUBLE_EQUIVALENCE_THRESHOLD && 
                firstStep.colSelection[colTone] <= colMax)
            {
                removalQueue.push_back(std::tuple(firstStep.rowSelection[rowTone], 
                    firstStep.colSelection[colTone]));

                if(stateArray(roundCoordDown(firstStep.rowSelection[rowTone]),
                    roundCoordDown(firstStep.colSelection[colTone])) && 
                    lastStep.rowSelection[rowTone] >= -DOUBLE_EQUIVALENCE_THRESHOLD && 
                    lastStep.rowSelection[rowTone] <= rowMax && 
                    lastStep.colSelection[colTone] >= -DOUBLE_EQUIVALENCE_THRESHOLD && 
                    lastStep.colSelection[colTone] <= colMax)
                {
                    addQueue.push_back(std::tuple(lastStep.rowSelection[rowTone], 
                        lastStep.colSelection[colTone]));
                    if(alreadyMoved.has_value())
                    {
                        alreadyMoved.value()(roundCoordDown(lastStep.rowSelection[rowTone]),
                            roundCoordDown(lastStep.colSelection[colTone])) = true;
                    }
                }
            }
        }
    }
    for(const auto& [row,col] : removalQueue)
    {
        stateArray(roundCoordDown(row), roundCoordDown(col)) = false;
    }
    for(const auto& [row,col] : addQueue)
    {
        stateArray(roundCoordDown(row), roundCoordDown(col)) = true;
    }
    return true;
}

bool ParallelMove::extendToUseAllTones(unsigned int stateArrayRows, unsigned int stateArrayCols, std::shared_ptr<spdlog::logger> logger, 
    bool considerSpacing, std::optional<std::vector<int>> bufferRows, std::optional<std::vector<int>> bufferCols)
{
    if(Config::getInstance().alwaysGenerateAllAODTones)
    {
        int requiredRowTones = Config::getInstance().aodRowLimit;
        int requiredColTones = Config::getInstance().aodColLimit;
        if(Config::getInstance().aodTotalLimit < requiredRowTones * requiredColTones)
        {
            logger->warn("AOD tone count cannot be maximized as total trap limit is smaller than row and column limit would allow");
            return false;
        }
        else if(this->steps.empty())
        {
            logger->warn("Empty move cannot meaningfully be extended to use all tones");
            return false;
        }
        else if(this->steps[0].rowSelection.size() > requiredRowTones || this->steps[0].colSelection.size() > requiredColTones)
        {
            logger->warn("Move already uses more tones than allowed so cannot be extended to use even more");
            return false;
        }
        else
        {
            requiredRowTones -= this->steps[0].rowSelection.size();
            requiredColTones -= this->steps[0].colSelection.size();

            if(requiredRowTones > 0)
            {
                if(!bufferRows.has_value())
                {
                    int rowDist = 1;
                    if(considerSpacing)
                    {
                        if(Config::getInstance().minDistFromOccSites > Config::getInstance().rowSpacing)
                        {
                            rowDist = ceil(Config::getInstance().rowSpacing / Config::getInstance().minDistFromOccSites);
                        }
                    }
                    int minTone = -rowDist, maxTone = stateArrayRows - 1 + rowDist;
                    for(const ParallelMove::Step& step : this->steps)
                    {
                        for(const auto& row : step.rowSelection)
                        {
                            if(row - rowDist < minTone)
                            {
                                minTone = row - rowDist;
                            }
                            if(row + rowDist > maxTone)
                            {
                                maxTone = row + rowDist;
                            }
                        }
                    }
                    // Add tones to move
                    for(ParallelMove::Step& step : this->steps)
                    {
                        for(int i = 0; i < (requiredRowTones + 1) / 2; i++)
                        {
                            step.rowSelection.insert(step.rowSelection.begin(), minTone - i);
                        }
                        for(int i = 0; i < requiredRowTones / 2; i++)
                        {
                            step.rowSelection.push_back(maxTone + i);
                        }
                    }
                }
                else
                {
                    // Add tones to move
                    for(ParallelMove::Step& step : this->steps)
                    {
                        for(int i = 0; i < requiredRowTones; i++)
                        {
                            step.rowSelection.push_back(bufferRows.value()[i]);
                        }
                        std::sort(step.rowSelection.begin(), step.rowSelection.end());
                    }
                }
            }
            if(requiredColTones > 0)
            {
                if(!bufferCols.has_value())
                {
                    int colDist = 1;
                    if(considerSpacing)
                    {
                    if(Config::getInstance().minDistFromOccSites > Config::getInstance().columnSpacing)
                    {
                        colDist = ceil(Config::getInstance().columnSpacing / Config::getInstance().minDistFromOccSites);
                    }
                    }
                    int minTone = -colDist, maxTone = stateArrayCols - 1 + colDist;
                    for(const ParallelMove::Step& step : this->steps)
                    {
                        for(const auto& col : step.colSelection)
                        {
                            if(col - colDist < minTone)
                            {
                                minTone = col - colDist;
                            }
                            if(col + colDist > maxTone)
                            {
                                maxTone = col + colDist;
                            }
                        }
                    }
                    // Add tones to move
                    for(ParallelMove::Step& step : this->steps)
                    {
                        for(int i = 0; i < (requiredColTones + 1) / 2; i++)
                        {
                            step.colSelection.insert(step.colSelection.begin(), minTone - i);
                        }
                        for(int i = 0; i < requiredColTones / 2; i++)
                        {
                            step.colSelection.push_back(maxTone + i);
                        }
                    }
                }
                else
                {
                    // Add tones to move
                    for(ParallelMove::Step& step : this->steps)
                    {
                        for(int i = 0; i < requiredColTones; i++)
                        {
                            step.colSelection.push_back(bufferCols.value()[i]);
                        }
                        std::sort(step.colSelection.begin(), step.colSelection.end());
                    }
                }
            }
        }
    }
    return true;
}

std::tuple<std::optional<ParallelMove>,int,double> improveMoveByAddingIndependentAtom(
    ArrayAccessor& stateArray, size_t compZone[4], std::shared_ptr<spdlog::logger> logger, ParallelMove move, 
    double cost, unsigned int correctedTargetSites,
    std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> alreadyMoved, 
    ArrayAccessor& targetGeometry)
{
    unsigned int aodRowLimit = Config::getInstance().aodRowLimit;
    unsigned int aodColLimit = Config::getInstance().aodColLimit;
    unsigned int aodTotalLimit = Config::getInstance().aodTotalLimit;
    bool allowMovingEmptyTrapOntoOccupied = Config::getInstance().allowMovingEmptyTrapOntoOccupied;

    if(aodColLimit <= 1 || aodRowLimit <= 1 || 
        !Config::getInstance().allowMovesBetweenCols || !Config::getInstance().allowMovesBetweenRows)
    {
        logger->info("Move limitations prevent method for improving move by adding independent atoms");
        return std::tuple(std::nullopt, 0, DBL_MAX);
    }

    ParallelMove::Step start = move.steps[0];
    ParallelMove::Step end = move.steps.back();

    double rowDiff = 0;
    for(size_t i = 0; i < start.rowSelection.size() && i < end.rowSelection.size(); i++)
    {
        double diff = abs(start.rowSelection[i] - end.rowSelection[i]);
        if(diff > rowDiff + DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            rowDiff = diff;
        }
    }
    double colDiff = 0;
    for(size_t i = 0; i < start.colSelection.size() && i < end.colSelection.size(); i++)
    {
        double diff = abs(start.colSelection[i] - end.colSelection[i]);
        if(diff > colDiff + DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            colDiff = diff;
        }
    }
    double baseCost = approxCostPerMove(rowDiff, colDiff);
    double rowCost = 0;
    if(rowDiff > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
    {
        rowCost = costPerSubMove(rowDiff - 1);
    }
    double colCost = 0;
    if(colDiff > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
    {
        colCost = costPerSubMove(colDiff - 1);
    }
    
    unsigned int selectedRows = start.rowSelection.size();
    unsigned int selectedCols = start.colSelection.size();

    double bestCostPerCorrectedTargetSite = cost / (double)correctedTargetSites;
    while(selectedRows < aodRowLimit && selectedCols < aodColLimit && (selectedCols + 1) * (selectedRows + 1) < aodTotalLimit)
    {
        std::optional<std::tuple<size_t,size_t,size_t,size_t>> addedOuterAndInnerRowAndCol = std::nullopt;

        std::vector<std::tuple<size_t,size_t,size_t>> viableColsWithInnerStartAndEnd;
        for(size_t colIndex = 0; colIndex <= selectedCols; colIndex++)
        {
            size_t outerStartIndex = colIndex > 0 ? start.colSelection[colIndex - 1] + 1 : 0;
            size_t outerEndIndex = colIndex < selectedCols ? start.colSelection[colIndex] : stateArray.cols();
            size_t innerStartIndex = colIndex > 0 ? end.colSelection[colIndex - 1] + 1 : compZone[2];
            size_t innerEndIndex = colIndex < selectedCols ? end.colSelection[colIndex] : compZone[3];
            if(innerEndIndex <= innerStartIndex)
            {
                continue;
            }
            for(size_t outerCol = outerStartIndex; outerCol < outerEndIndex; outerCol++)
            {
                bool allowed = true;
                for(double row : start.rowSelection)
                {
                    if(stateArray(roundCoordDown(row), outerCol))
                    {
                        allowed = false;
                        break;
                    }
                }
                if(allowed)
                {
                    viableColsWithInnerStartAndEnd.push_back(std::tuple(outerCol, innerStartIndex, innerEndIndex));
                }
            }
        }
        for(size_t rowIndex = 0; rowIndex <= selectedRows; rowIndex++)
        {
            size_t outerStartIndex = rowIndex > 0 ? start.rowSelection[rowIndex - 1] + 1 : 0;
            size_t outerEndIndex = rowIndex < selectedRows ? start.rowSelection[rowIndex] : stateArray.rows();
            size_t innerStartIndex = rowIndex > 0 ? end.rowSelection[rowIndex - 1] + 1 : compZone[0];
            size_t innerEndIndex = rowIndex < selectedRows ? end.rowSelection[rowIndex] : compZone[1];
            if(innerEndIndex <= innerStartIndex)
            {
                continue;
            }
            for(size_t outerRow = outerStartIndex; outerRow < outerEndIndex; outerRow++)
            {
                bool allowed = true;
                for(double col : start.colSelection)
                {
                    if(stateArray(outerRow,roundCoordDown(col)))
                    {
                        allowed = false;
                        break;
                    }
                }
                if(allowed)
                {
                    for(const auto& [outerCol, innerColStart, innerColEnd] : viableColsWithInnerStartAndEnd)
                    {
                        bool originatesInCompZone = outerRow >= compZone[0] && outerRow < compZone[1] && 
                            outerCol >= compZone[2] && outerCol < compZone[3];
                        if(stateArray(outerRow, outerCol) && (!originatesInCompZone || 
                            (originatesInCompZone && !targetGeometry(outerRow - compZone[0], outerCol - compZone[2]))))
                        {
                            for(size_t innerRow = innerStartIndex; innerRow < innerEndIndex; innerRow++)
                            {
                                bool allowed = true;
                                if(!allowMovingEmptyTrapOntoOccupied)
                                {
                                    for(double col : end.colSelection)
                                    {
                                        if(stateArray(innerRow, roundCoordDown(col)))
                                        {
                                            allowed = false;
                                            break;
                                        }
                                    }
                                }
                                for(size_t innerCol = innerColStart; innerCol < innerColEnd; innerCol++)
                                {
                                    if(!allowMovingEmptyTrapOntoOccupied)
                                    {
                                        for(double row : end.rowSelection)
                                        {
                                            if(stateArray(roundCoordDown(row), innerCol))
                                            {
                                                allowed = false;
                                                break;
                                            }
                                        }
                                    }
                                    if(allowed && !stateArray(innerRow, innerCol) && 
                                        targetGeometry(innerRow - compZone[0], innerCol - compZone[2]))
                                    {
                                        double newCost = baseCost;
                                        double newRowDiff = abs((double)innerRow - (double)outerRow);
                                        if(newRowDiff > rowDiff + DOUBLE_EQUIVALENCE_THRESHOLD && newRowDiff > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
                                        {
                                            newCost += costPerSubMove(newRowDiff - 1) - rowCost;
                                        }
                                        double newColDiff = abs((double)innerCol - (double)outerCol);
                                        if(newColDiff > colDiff + DOUBLE_EQUIVALENCE_THRESHOLD && newColDiff > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
                                        {
                                            newCost += costPerSubMove(newColDiff - 1) - colCost;
                                        }
                                        if(newCost / (correctedTargetSites + originatesInCompZone + 1) < bestCostPerCorrectedTargetSite)
                                        {
                                            bestCostPerCorrectedTargetSite = newCost / (correctedTargetSites + originatesInCompZone + 1);
                                            addedOuterAndInnerRowAndCol = std::tuple(outerRow, innerRow, outerCol, innerCol);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if(addedOuterAndInnerRowAndCol.has_value())
        {
            double outerRow = std::get<0>(addedOuterAndInnerRowAndCol.value());
            double innerRow = std::get<1>(addedOuterAndInnerRowAndCol.value());
            double outerCol = std::get<2>(addedOuterAndInnerRowAndCol.value());
            double innerCol = std::get<3>(addedOuterAndInnerRowAndCol.value());

            // One filled anyways, one more if new atom is from inside target area
            correctedTargetSites++;
            if(std::get<0>(addedOuterAndInnerRowAndCol.value()) >= compZone[0] && 
                std::get<0>(addedOuterAndInnerRowAndCol.value()) < compZone[1] && 
                std::get<2>(addedOuterAndInnerRowAndCol.value()) >= compZone[2] && 
                std::get<2>(addedOuterAndInnerRowAndCol.value()) < compZone[3])
            {
                correctedTargetSites++;
            }

            start.rowSelection.insert(std::upper_bound(start.rowSelection.begin(), start.rowSelection.end(), 
                outerRow), outerRow);
            end.rowSelection.insert(std::upper_bound(end.rowSelection.begin(), end.rowSelection.end(), 
                innerRow), innerRow);
            start.colSelection.insert(std::upper_bound(start.colSelection.begin(), start.colSelection.end(), 
                outerCol), outerCol);
            end.colSelection.insert(std::upper_bound(end.colSelection.begin(), end.colSelection.end(), 
                innerCol), innerCol);
            selectedRows++;
            selectedCols++;
            double newRowDiff = abs((double)innerRow - (double)outerRow);
            if(newRowDiff > rowDiff + DOUBLE_EQUIVALENCE_THRESHOLD)
            {
                rowDiff = newRowDiff;
                rowCost = 0;
                if(newRowDiff > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
                {
                    rowCost = costPerSubMove(newRowDiff - 1);
                }
            }
            double newColDiff = abs((double)innerCol - (double)outerCol);
            if(newColDiff > colDiff + DOUBLE_EQUIVALENCE_THRESHOLD)
            {
                colDiff = newColDiff;
                colCost = 0;
                if(newColDiff > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
                {
                    colCost = costPerSubMove(newColDiff - 1);
                }
            }

            logger->debug("Adding row {}->{} and col {}->{} for 1 additional corrected target sites", 
                outerRow, innerRow, outerCol, innerCol);
        }
        else
        {
            break;
        }
    }
    
    ParallelMove improvedMove = ParallelMove::fromStartAndEnd(stateArray, start, end, logger);
    bestCostPerCorrectedTargetSite = improvedMove.cost() / correctedTargetSites;
    return std::tuple(std::move(improvedMove), correctedTargetSites, bestCostPerCorrectedTargetSite);
}

std::tuple<std::optional<ParallelMove>,int,double> improveComplexMove(
    ArrayAccessor& stateArray, size_t compZone[4], std::shared_ptr<spdlog::logger> logger, ParallelMove move, 
    double cost, unsigned int correctedTargetSites,
    std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> alreadyMoved, 
    ArrayAccessor& targetGeometry)
{
    unsigned int aodRowLimit = Config::getInstance().aodRowLimit;
    unsigned int aodColLimit = Config::getInstance().aodColLimit;
    unsigned int aodTotalLimit = Config::getInstance().aodTotalLimit;
    bool allowMovingEmptyTrapOntoOccupied = Config::getInstance().allowMovingEmptyTrapOntoOccupied;
    bool allowMultipleMovesPerAtom = Config::getInstance().allowMultipleMovesPerAtom;

    if(aodColLimit <= 1 || aodRowLimit <= 1 || !Config::getInstance().allowMovesBetweenCols || !Config::getInstance().allowMovesBetweenRows)
    {
        logger->info("Move limitations prevent method for improving complex row and col move");
        return std::tuple(std::nullopt, 0, DBL_MAX);
    }

    ParallelMove::Step start = move.steps[0];
    ParallelMove::Step end = move.steps.back();

    logger->debug("Move to optimize already corrects {} target sites", correctedTargetSites);

    double rowDiff = 0;
    for(size_t i = 0; i < start.rowSelection.size() && i < end.rowSelection.size(); i++)
    {
        double diff = abs(start.rowSelection[i] - end.rowSelection[i]);
        if(diff > rowDiff + DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            rowDiff = diff;
        }
    }
    double colDiff = 0;
    for(size_t i = 0; i < start.colSelection.size() && i < end.colSelection.size(); i++)
    {
        double diff = abs(start.colSelection[i] - end.colSelection[i]);
        if(diff > colDiff + DOUBLE_EQUIVALENCE_THRESHOLD)
        {
            colDiff = diff;
        }
    }
    double baseCost = approxCostPerMove(rowDiff, colDiff);
    logger->debug("Base cost: {} for r {} c {}", baseCost, rowDiff, colDiff);
    double costWithoutRowMove = approxCostPerMove(0, colDiff);
    double costWithoutColMove = approxCostPerMove(rowDiff, 0);

    unsigned int selectedRows = start.rowSelection.size();
    unsigned int selectedCols = start.colSelection.size();

    int bestAdditionalCorrectedTargetSites = 0;
    double bestCostPerCorrectedTargetSite = cost / (double)correctedTargetSites;
    unsigned int bestAdditionalDiff = 0;
    while(true)
    {
        std::optional<std::tuple<size_t,size_t,bool>> addedOuterAndInnerIndexAndIsRow = std::nullopt;

        if(selectedRows < aodRowLimit && selectedCols * (selectedRows + 1) < aodTotalLimit)
        {
            for(size_t rowIndex = 0; rowIndex <= selectedRows; rowIndex++)
            {
                size_t outerStartIndex = rowIndex > 0 ? start.rowSelection[rowIndex - 1] + 1 : 0;
                size_t outerEndIndex = rowIndex < selectedRows ? start.rowSelection[rowIndex] : stateArray.rows();
                size_t innerStartIndex = rowIndex > 0 ? end.rowSelection[rowIndex - 1] + 1 : compZone[0];
                size_t innerEndIndex = rowIndex < selectedRows ? end.rowSelection[rowIndex] : compZone[1];
                if(innerEndIndex <= innerStartIndex)
                {
                    continue;
                }
                for(size_t outerRow = outerStartIndex; outerRow < outerEndIndex; outerRow++)
                {
                    if(!allowMultipleMovesPerAtom && alreadyMoved.has_value())
                    {
                        bool allowed = true;
                        for(size_t colIndex = 0; colIndex < selectedCols; colIndex++)
                        {
                            if(alreadyMoved.value()(outerRow, roundCoordDown(start.colSelection[colIndex])))
                            {
                                allowed = false;
                            }
                        }
                        if(!allowed)
                        {
                            continue;
                        }
                    }
                    for(size_t innerRow = innerStartIndex; innerRow < innerEndIndex; innerRow++)
                    {
                        int additionalCorrectedTargetSites = 0;
                        bool allowed = true;
                        for(size_t colIndex = 0; colIndex < selectedCols; colIndex++)
                        {
                            size_t startCol = roundCoordDown(start.colSelection[colIndex]);
                            size_t endCol = roundCoordDown(end.colSelection[colIndex]);
                            if(stateArray(outerRow, startCol))
                            {
                                if(stateArray(innerRow, endCol))
                                {
                                    allowed = false;
                                    break;
                                }
                                if(isInCompZone(innerRow, endCol, compZone[0], compZone[1], compZone[2], compZone[3]))
                                {
                                    if(targetGeometry(innerRow - compZone[0], endCol - compZone[2]))
                                    {
                                        additionalCorrectedTargetSites++;
                                    }
                                    else
                                    {
                                        additionalCorrectedTargetSites--;
                                    }
                                }
                                if(isInCompZone(outerRow, startCol, compZone[0], compZone[1], compZone[2], compZone[3]))
                                {
                                    if(targetGeometry(outerRow - compZone[0], startCol - compZone[2]))
                                    {
                                        additionalCorrectedTargetSites--;
                                    }
                                    else
                                    {
                                        additionalCorrectedTargetSites++;
                                    }
                                }
                            }
                            else if(stateArray(innerRow, endCol) && !allowMovingEmptyTrapOntoOccupied)
                            {
                                allowed = false;
                            }
                        }
                        if(allowed && additionalCorrectedTargetSites > 0)
                        {
                            double newCost = baseCost;
                            double diff = abs((double)outerRow - (double)innerRow);
                            if(diff > rowDiff + DOUBLE_EQUIVALENCE_THRESHOLD && diff > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
                            {
                                newCost = costWithoutRowMove + costPerSubMove(diff - 1);
                            }
                            double costPerCorrectedTargetSite = newCost / (correctedTargetSites + additionalCorrectedTargetSites);
                            if(costPerCorrectedTargetSite < bestCostPerCorrectedTargetSite)
                            {
                                if(diff > rowDiff + DOUBLE_EQUIVALENCE_THRESHOLD)
                                {
                                    bestAdditionalDiff = diff - colDiff;
                                }
                                else
                                {
                                    bestAdditionalDiff = 0;
                                }
                                bestAdditionalCorrectedTargetSites = additionalCorrectedTargetSites;
                                bestCostPerCorrectedTargetSite = costPerCorrectedTargetSite;
                                addedOuterAndInnerIndexAndIsRow = std::tuple(outerRow, innerRow, true);
                            }
                        }
                    }
                }
            }
        }
        if(selectedCols < aodColLimit && selectedRows * (selectedCols + 1) < aodTotalLimit)
        {
            for(size_t colIndex = 0; colIndex <= selectedCols; colIndex++)
            {
                size_t outerStartIndex = colIndex > 0 ? start.colSelection[colIndex - 1] + 1 : 0;
                size_t outerEndIndex = colIndex < selectedCols ? start.colSelection[colIndex] : stateArray.cols();
                size_t innerStartIndex = colIndex > 0 ? end.colSelection[colIndex - 1] + 1 : compZone[2];
                size_t innerEndIndex = colIndex < selectedCols ? end.colSelection[colIndex] : compZone[3];
                if(innerEndIndex <= innerStartIndex)
                {
                    continue;
                }
                for(size_t outerCol = outerStartIndex; outerCol < outerEndIndex; outerCol++)
                {
                    if(!allowMultipleMovesPerAtom && alreadyMoved.has_value())
                    {
                        bool allowed = true;
                        for(size_t rowIndex = 0; rowIndex < selectedRows; rowIndex++)
                        {
                            if(alreadyMoved.value()(roundCoordDown(start.rowSelection[rowIndex]), outerCol))
                            {
                                allowed = false;
                            }
                        }
                        if(!allowed)
                        {
                            continue;
                        }
                    }
                    for(size_t innerCol = innerStartIndex; innerCol < innerEndIndex; innerCol++)
                    {
                        int additionalCorrectedTargetSites = 0;
                        bool allowed = true;
                        for(size_t rowIndex = 0; rowIndex < selectedRows; rowIndex++)
                        {
                            size_t startRow = roundCoordDown(start.rowSelection[rowIndex]);
                            size_t endRow = roundCoordDown(end.rowSelection[rowIndex]);
                            if(stateArray(startRow, outerCol))
                            {
                                if(stateArray(endRow, innerCol))
                                {
                                    allowed = false;
                                    break;
                                }
                                if(isInCompZone(endRow, innerCol, compZone[0], compZone[1], compZone[2], compZone[3]))
                                {
                                    if(targetGeometry(endRow - compZone[0], innerCol - compZone[2]))
                                    {
                                        additionalCorrectedTargetSites++;
                                    }
                                    else
                                    {
                                        additionalCorrectedTargetSites--;
                                    }
                                }
                                if(isInCompZone(startRow, outerCol, compZone[0], compZone[1], compZone[2], compZone[3]))
                                {
                                    if(targetGeometry(startRow - compZone[0], outerCol - compZone[2]))
                                    {
                                        additionalCorrectedTargetSites--;
                                    }
                                    else
                                    {
                                        additionalCorrectedTargetSites++;
                                    }
                                }
                            }
                            else if(stateArray(endRow, innerCol) && !allowMovingEmptyTrapOntoOccupied)
                            {
                                allowed = false;
                            }
                        }
                        if(allowed && additionalCorrectedTargetSites > 0)
                        {
                            double newCost = baseCost;
                            double diff = abs((double)outerCol - (double)innerCol);
                            if(diff > colDiff + DOUBLE_EQUIVALENCE_THRESHOLD && diff > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
                            {
                                newCost = costWithoutColMove + costPerSubMove(diff - 1);
                            }
                            double costPerFilledVacancy = newCost / (correctedTargetSites + additionalCorrectedTargetSites);
                            if(costPerFilledVacancy < bestCostPerCorrectedTargetSite)
                            {
                                if(diff > 1 + DOUBLE_EQUIVALENCE_THRESHOLD)
                                {
                                    bestAdditionalDiff = diff - rowDiff;
                                }
                                else
                                {
                                    bestAdditionalDiff = 0;
                                }
                                bestAdditionalCorrectedTargetSites = additionalCorrectedTargetSites;
                                bestCostPerCorrectedTargetSite = costPerFilledVacancy;
                                addedOuterAndInnerIndexAndIsRow = std::tuple(outerCol, innerCol, false);
                            }
                        }
                    }
                }
            }
        }

        if(addedOuterAndInnerIndexAndIsRow.has_value())
        {
            double startIndex = std::get<0>(addedOuterAndInnerIndexAndIsRow.value());
            double endIndex = std::get<1>(addedOuterAndInnerIndexAndIsRow.value());
            correctedTargetSites += bestAdditionalCorrectedTargetSites;
            if(std::get<2>(addedOuterAndInnerIndexAndIsRow.value()))
            {
                start.rowSelection.insert(std::upper_bound(start.rowSelection.begin(), start.rowSelection.end(), 
                    startIndex), startIndex);
                end.rowSelection.insert(std::upper_bound(end.rowSelection.begin(), end.rowSelection.end(), 
                    endIndex), endIndex);
                selectedRows++;
                rowDiff += bestAdditionalDiff;
                logger->debug("Adding row {}->{} for {} additional vacancy fillings", 
                    startIndex, endIndex, bestAdditionalCorrectedTargetSites);
            }
            else
            {
                start.colSelection.insert(std::upper_bound(start.colSelection.begin(), start.colSelection.end(), 
                    startIndex), startIndex);
                end.colSelection.insert(std::upper_bound(end.colSelection.begin(), end.colSelection.end(), 
                    endIndex), endIndex);
                selectedCols++;
                colDiff += bestAdditionalDiff;
                logger->debug("Adding col {}->{} for {} additional vacancy fillings", 
                    startIndex, endIndex, bestAdditionalCorrectedTargetSites);
            }
            break;
        }
        else
        {
            break;
        }
    }

    ParallelMove improvedMove = ParallelMove::fromStartAndEnd(stateArray, start, end, logger);
    bestCostPerCorrectedTargetSite = improvedMove.cost() / correctedTargetSites;
    return std::tuple(std::move(improvedMove), correctedTargetSites, bestCostPerCorrectedTargetSite);
}

std::pair<std::vector<RowBitMask>, unsigned int> determineWhichRowsToUse(
    std::map<unsigned int,std::vector<RowBitMask>>& rowBitMasks, unsigned int targetIterCount, size_t lastConsideredIndexExclusive)
{
    std::vector<RowBitMask> usedRows;
    if(rowBitMasks.size() == 0)
    {
        return std::pair(usedRows, 0);
    }

    // Skew towards earlier by giving 150% to the first and 50% to the last and interpolating linearly inbetween
    unsigned int usedIterCount = 0;

    unsigned int cutoffOverlap = rowBitMasks.rbegin()->first * CUTOFF_OVERLAP_FRACTION;
    for(auto reverseIterator = rowBitMasks.rbegin(); reverseIterator != rowBitMasks.rend() && usedIterCount < targetIterCount; 
        reverseIterator++)
    {
        auto& [overlap, rows] = *reverseIterator;
        if(overlap < cutoffOverlap)
        {
            return std::pair(usedRows, usedIterCount);
        }
        for(auto& row : rows)
        {
            int requiredIterations = (int)lastConsideredIndexExclusive - (int)row.indices.back() - 1;
            if(requiredIterations > 0)
            {
                if(usedIterCount + requiredIterations > targetIterCount && !usedRows.empty())
                {
                    return std::pair(usedRows, usedIterCount);
                }
                else
                {
                    usedIterCount += requiredIterations;
                    usedRows.push_back(std::move(row));
                }
            }
        }
    }
    return std::pair(usedRows, usedIterCount);
}

std::optional<ParallelMove> constructMoveWithSeveralRowsAndCols(ArrayAccessor& stateArray,
    RowBitMask& source, RowBitMask& target, bool rowFirst,
    size_t compZone[4], unsigned int maxCols, std::shared_ptr<spdlog::logger> logger)
{
    ParallelMove::Step start;
    ParallelMove::Step end;
    std::vector<double> *innerDimStartVector, *innerDimEndVector;

    if(rowFirst)
    {
        start.rowSelection = std::vector<double>(source.indices.begin(), source.indices.end());
        end.rowSelection = std::vector<double>(target.indices.begin(), target.indices.end());
        innerDimStartVector = &start.colSelection;
        innerDimEndVector = &end.colSelection;
    }
    else
    {
        start.colSelection = std::vector<double>(source.indices.begin(), source.indices.end());
        end.colSelection = std::vector<double>(target.indices.begin(), target.indices.end());
        innerDimStartVector = &start.rowSelection;
        innerDimEndVector = &end.rowSelection;
    }

    size_t colDimCompZoneStart = rowFirst ? compZone[2] : compZone[0];
    size_t colDimCompZoneEnd = rowFirst ? compZone[3] : compZone[1];
    std::vector<int> sourceIndices, innerSourceIndices;
    unsigned int sourceIndicesSmallerThanCompStart = 0;
    for(size_t i = 0; i < source.count; i++)
    {
        if(i == colDimCompZoneStart)
        {
            sourceIndicesSmallerThanCompStart = sourceIndices.size();
        }
        if(source[i])
        {
            if(i >= colDimCompZoneStart && i < colDimCompZoneEnd)
            {
                innerSourceIndices.push_back(i);
            }
            sourceIndices.push_back(i);
        }
    }
    std::vector<int> targetIndices;
    for(size_t i = 0; i < target.count && targetIndices.size() < maxCols; i++)
    {
        if(target[i])
        {
            targetIndices.push_back(i + colDimCompZoneStart);
        }
    }

    if(innerSourceIndices.size() > targetIndices.size())
    {
        // Use only source indices from comp zone if sufficient
        auto pairing = bottleneckAssignmentMinimizeMaxDist(innerSourceIndices, targetIndices);
        if(!pairing.has_value())
        {
            return std::nullopt;
        }
        for(auto& [startIndex, endIndex] : pairing.value())
        {
            innerDimStartVector->push_back(startIndex);
            innerDimEndVector->push_back(endIndex);
        }
    }
    else if(sourceIndices.size() == targetIndices.size())
    {
        for(size_t i = 0; i < sourceIndices.size(); i++)
        {
            innerDimStartVector->push_back(sourceIndices[i]);
            innerDimEndVector->push_back(targetIndices[i]);
        }
    }
    else if(sourceIndices.size() > targetIndices.size())
    {
        int excessSourceIndices = sourceIndices.size() - targetIndices.size();
        int minIndicesFromLeft = sourceIndicesSmallerThanCompStart - excessSourceIndices;
        if(minIndicesFromLeft < 0)
        {
            minIndicesFromLeft = 0;
        }
        std::optional<int> minDist = std::nullopt;
        int bestIndicesFromLeft = minIndicesFromLeft;
        for(int indicesFromLeft = minIndicesFromLeft; indicesFromLeft <= (int)sourceIndicesSmallerThanCompStart; indicesFromLeft++)
        {
            int currentDist = 0;
            for(int i = 0; i < (int)targetIndices.size(); i++)
            {
                int dist = abs(sourceIndices[sourceIndicesSmallerThanCompStart - indicesFromLeft + i] - targetIndices[i]);
                if(dist > currentDist)
                {
                    currentDist = dist;
                }
            }
            if(minDist.has_value())
            {
                if(currentDist < minDist.value())
                {
                    minDist = currentDist;
                    bestIndicesFromLeft = indicesFromLeft;
                }
                else
                {
                    break;
                }
            }
            else
            {
                minDist = currentDist;
                bestIndicesFromLeft = indicesFromLeft;
            }
        }
        for(int i = 0; i < (int)targetIndices.size(); i++)
        {
            innerDimStartVector->push_back(sourceIndices[sourceIndicesSmallerThanCompStart - bestIndicesFromLeft + i]);
            innerDimEndVector->push_back(targetIndices[i]);
        }
    }
    else
    {
        auto pairing = bottleneckAssignmentMinimizeMaxDist(sourceIndices, targetIndices);
        if(!pairing.has_value())
        {
            return std::nullopt;
        }
        for(auto& [startIndex, endIndex] : pairing.value())
        {
            innerDimStartVector->push_back(startIndex);
            innerDimEndVector->push_back(endIndex);
        }
    }

    return ParallelMove::fromStartAndEnd(stateArray, start, end, logger);
}

std::vector<std::tuple<ParallelMove,int,double>> moveSeveralRowsAndCols(ArrayAccessor& stateArray, 
    size_t compZone[4], std::shared_ptr<spdlog::logger> logger,
    std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> alreadyMoved, 
    ArrayAccessor& targetGeometry)
{
    unsigned int aodTotalLimit = Config::getInstance().aodTotalLimit;

    if(Config::getInstance().aodColLimit <= 1 || Config::getInstance().aodRowLimit <= 1 || 
        !Config::getInstance().allowMovesBetweenCols || !Config::getInstance().allowMovesBetweenRows)
    {
        logger->info("Move limitations prevent method for complicated row and col move");
        return std::vector<std::tuple<ParallelMove,int,double>>();
    }

    unsigned int roundedDownSqrtTotalAOD = sqrt(aodTotalLimit);
    std::optional<ParallelMove> bestMove = std::nullopt;
    unsigned int bestFillableGaps = 0;
    double bestCostPerFilledGap = 0;

    RowBitMask bestStart, bestEnd;
    bool bestRowFirst = false;
    unsigned int bestColLimit;

    for(bool rowFirst : {true, false})
    {
        size_t rowDimCompZone[2];
        size_t colDimCompZone[2];
        size_t rows;
        size_t cols;
        unsigned int rowAODLimit = 0;
        unsigned int colAODLimit = 0;
        fillDimensionDependantData(stateArray, compZone, rowFirst, rowDimCompZone, colDimCompZone, rows, cols, rowAODLimit, colAODLimit);

        if(rowAODLimit <= 1)
        {
            continue;
        }

        unsigned int maxRows = rowAODLimit;
        if(maxRows > roundedDownSqrtTotalAOD)
        {
            maxRows = roundedDownSqrtTotalAOD;
        }

        std::vector<RowBitMask> bitMaskByInnerRow;
        std::vector<RowBitMask> bitMaskByOuterRow;
        std::map<unsigned int,std::vector<RowBitMask>> bitMaskInnerVec1;
        std::map<unsigned int,std::vector<RowBitMask>> bitMaskOuterVec1;
        for(size_t i = 0; i < rows; i++)
        {
            RowBitMask outerRowBitMask(cols, i);
            for(size_t j = 0; j < cols; j++)
            {
                bool inCompZone = i >= rowDimCompZone[0] && i < rowDimCompZone[1] && j >= colDimCompZone[0] && j < colDimCompZone[1];
                outerRowBitMask.set(j, accessArrayDim(stateArray, i, j, rowFirst) && (!inCompZone || 
                    !accessArrayDim(targetGeometry, i - rowDimCompZone[0], j - colDimCompZone[0], rowFirst)));
            }
            bitMaskByOuterRow.push_back(outerRowBitMask);
            bitMaskOuterVec1[outerRowBitMask.bitsSet()].push_back(std::move(outerRowBitMask));
            if(i >= rowDimCompZone[0] && i < rowDimCompZone[1])
            {
                RowBitMask innerRowBitMask(colDimCompZone[1] - colDimCompZone[0], i);
                for(size_t j = colDimCompZone[0]; j < colDimCompZone[1]; j++)
                {
                    innerRowBitMask.set(j - colDimCompZone[0], !accessArrayDim(stateArray, i, j, rowFirst) &&
                        accessArrayDim(targetGeometry, i - rowDimCompZone[0], j - colDimCompZone[0], rowFirst));
                } 
                bitMaskByInnerRow.push_back(innerRowBitMask);
                bitMaskInnerVec1[innerRowBitMask.bitsSet()].push_back(std::move(innerRowBitMask));
            }
        }

        // Calculate maximum target sizes
        std::map<unsigned int,std::vector<RowBitMask>> bitMaskInnerVec2;
        std::map<unsigned int,std::vector<RowBitMask>> *prevBitMasksPerInnerRowSet = &bitMaskInnerVec2;
        std::map<unsigned int,std::vector<RowBitMask>> *bitMasksPerInnerRowSet = &bitMaskInnerVec1;

        // Calculate maximum sizes for border
        std::map<unsigned int,std::vector<RowBitMask>> bitMaskOuterVec2;
        std::map<unsigned int,std::vector<RowBitMask>> *prevBitMasksPerOuterRowSet = &bitMaskOuterVec2;
        std::map<unsigned int,std::vector<RowBitMask>> *bitMasksPerOuterRowSet = &bitMaskOuterVec1;

        unsigned int iterCount = 0;
        std::vector<RowBitMask> rowsToUseInner, rowsToUseOuter;

        // If both row and column distance is larger than 1, then 4 submoves are required
        double approxBaselineCost = Config::getInstance().moveCostOffset + 4 * Config::getInstance().moveCostOffsetSubmove;
            
        std::vector<const RowBitMask*> outerRowsToInvestigate, innerRowsToInvestigate;
        unsigned int targetIterCount = MAX_MULTI_ITER_COUNT;
        unsigned int maxCols = colAODLimit;
        
        unsigned int approxOuterMaxOverlap = 2, approxInnerMaxOverlap = 2;

        #pragma omp parallel
        {
            for(unsigned int rowCount = 2; rowCount <= maxRows && !bitMasksPerInnerRowSet->empty() && 
                !bitMasksPerOuterRowSet->empty(); rowCount++)
            {
                if(rowCount > 2)
                {
                    // Take remaining iterations and use 1-3 times as many as the average allows (more at beginning)
                    targetIterCount = (MAX_MULTI_ITER_COUNT - iterCount) / (maxRows - rowCount + 1) / 2 * 
                        (1 + (maxRows - rowCount) / (maxRows - 2) * 2);
                }
                if(aodTotalLimit / rowCount < maxCols)
                {
                    maxCols = aodTotalLimit / rowCount;
                }
                std::map<unsigned int,std::vector<RowBitMask>> localInner, localOuter;
                unsigned int maxLocalInner = 2, maxLocalOuter = 2;

                // Prepare list of index combinations to investigate
                #pragma omp barrier
                #pragma omp single nowait
                {
                    approxOuterMaxOverlap = 2;
                    auto *tmpRef = prevBitMasksPerInnerRowSet;
                    prevBitMasksPerInnerRowSet = bitMasksPerInnerRowSet;
                    bitMasksPerInnerRowSet = tmpRef;
                    bitMasksPerInnerRowSet->clear();

                    unsigned int usedIterations;
                    std::tie(rowsToUseInner, usedIterations) = determineWhichRowsToUse(
                        *prevBitMasksPerInnerRowSet, targetIterCount, rowDimCompZone[1]);
                    logger->debug("Investigating {} rows for inner section at count {}", rowsToUseInner.size(), rowCount);
                    iterCount += usedIterations;
                }
                #pragma omp single
                {
                    approxInnerMaxOverlap = 2;
                    auto *tmpRef = prevBitMasksPerOuterRowSet;
                    prevBitMasksPerOuterRowSet = bitMasksPerOuterRowSet;
                    bitMasksPerOuterRowSet = tmpRef;
                    bitMasksPerOuterRowSet->clear();
                    if(!prevBitMasksPerInnerRowSet->empty())
                    {
                        logger->debug("Max empty columns for {} rows: {}", rowCount - 1, prevBitMasksPerInnerRowSet->rbegin()->first);
                    }

                    unsigned int usedIterations;
                    std::tie(rowsToUseOuter, usedIterations) = determineWhichRowsToUse(
                        *prevBitMasksPerOuterRowSet, targetIterCount, rows);
                    logger->debug("Investigating {} rows for outer section at count {}", rowsToUseOuter.size(), rowCount);
                    iterCount += usedIterations;
                }
                // Investigate index sets by adding additional indices
                #pragma omp for nowait schedule(dynamic, 4)
                for(const auto& prevBitMask : rowsToUseInner)
                {
                    for(size_t i = prevBitMask.indices.back() + 1; i < rowDimCompZone[1]; i++)
                    {
                        RowBitMask combBitMask = RowBitMask::fromAnd(prevBitMask, bitMaskByInnerRow[i - rowDimCompZone[0]]);
                        unsigned int overlap = combBitMask.bitsSet();
                        if(overlap >= approxInnerMaxOverlap * CUTOFF_OVERLAP_FRACTION &&
                            overlap >= bestFillableGaps / (rowCount + 1) && overlap >= 2)
                        {
                            if(overlap > maxLocalInner)
                            {
                                maxLocalInner = overlap;
                            }
                            if(maxLocalInner > approxInnerMaxOverlap)
                            {
                                approxInnerMaxOverlap = maxLocalInner;
                            }
                            localInner[overlap].push_back(std::move(combBitMask));
                        }
                    }
                }
                #pragma omp critical
                {
                    for(auto rIter = localInner.rbegin(); rIter != localInner.rend(); rIter++)
                    {
                        unsigned int overlap = std::get<0>(*rIter);
                        if(overlap < approxInnerMaxOverlap * CUTOFF_OVERLAP_FRACTION)
                        {
                            break;
                        }
                        for(auto combBitMask : std::get<1>(*rIter))
                        {
                            (*bitMasksPerInnerRowSet)[overlap].push_back(RowBitMask(combBitMask));
                        }
                    }
                }
                #pragma omp for nowait schedule(dynamic, 1)
                for(const auto& prevBitMask : rowsToUseOuter)
                {
                    for(size_t i = prevBitMask.indices.back() + 1; i < rows; i++)
                    {
                        RowBitMask combBitMask = RowBitMask::fromAnd(prevBitMask, bitMaskByOuterRow[i]);
                        unsigned int overlap = combBitMask.bitsSet();
                        if(overlap >= approxOuterMaxOverlap * CUTOFF_OVERLAP_FRACTION &&
                            overlap >= bestFillableGaps / (rowCount + 1) && overlap >= 2)
                        {
                            if(overlap > maxLocalOuter)
                            {
                                maxLocalOuter = overlap;
                            }
                            if(maxLocalOuter > approxOuterMaxOverlap)
                            {
                                approxOuterMaxOverlap = maxLocalOuter;
                            }
                            localOuter[overlap].push_back(std::move(combBitMask));
                        }
                    }
                }
                #pragma omp critical
                {
                    for(auto rIter = localOuter.rbegin(); rIter != localOuter.rend(); rIter++)
                    {
                        unsigned int overlap = std::get<0>(*rIter);
                        if(overlap < approxOuterMaxOverlap * CUTOFF_OVERLAP_FRACTION)
                        {
                            break;
                        }
                        for(auto combBitMask : std::get<1>(*rIter))
                        {
                            (*bitMasksPerOuterRowSet)[overlap].push_back(RowBitMask(combBitMask));
                        }
                    }
                }
                #pragma omp barrier
                if(!bitMasksPerInnerRowSet->empty() && !bitMasksPerOuterRowSet->empty())
                {
                    // Select index sets that are to be used in move creation
                    #pragma omp single nowait
                    {
                        outerRowsToInvestigate.clear();
                        unsigned int maxOuterOverlap = bitMasksPerOuterRowSet->rbegin()->first;
                        for(auto reverseIteratorOuter = bitMasksPerOuterRowSet->rbegin(); 
                            reverseIteratorOuter != bitMasksPerOuterRowSet->rend(); 
                            reverseIteratorOuter++)
                        {
                            const auto& [overlap, rows] = *reverseIteratorOuter;
                            if(overlap < maxOuterOverlap * CUTOFF_OVERLAP_FRACTION ||
                                (bestMove.has_value() && approxBaselineCost / (overlap * rowCount) > bestCostPerFilledGap) || 
                                outerRowsToInvestigate.size() >= MAX_INVESTIGATED_OVERLAP_SELECTIONS)
                            {
                                break;
                            }
                            for(const auto& row : rows)
                            {
                                outerRowsToInvestigate.push_back(&row);
                            }
                        }
                    }
                    #pragma omp single
                    {
                        innerRowsToInvestigate.clear();
                        unsigned int maxInnerOverlap = bitMasksPerInnerRowSet->rbegin()->first;
                        for(auto reverseIteratorInner = bitMasksPerInnerRowSet->rbegin(); 
                            reverseIteratorInner != bitMasksPerInnerRowSet->rend(); 
                            reverseIteratorInner++)
                        {
                            const auto& [overlap, rows] = *reverseIteratorInner;
                            if(overlap < maxInnerOverlap * CUTOFF_OVERLAP_FRACTION ||
                                (bestMove.has_value() && approxBaselineCost / (overlap * rowCount) > bestCostPerFilledGap) || 
                                innerRowsToInvestigate.size() >= MAX_INVESTIGATED_OVERLAP_SELECTIONS)
                            {
                                break;
                            }
                            for(const auto& row : rows)
                            {
                                innerRowsToInvestigate.push_back(&row);
                            }
                        }
                    }
                    // Approximate for each combination of start and end indices how good the resulting move would be
                    #pragma omp for collapse(2) nowait schedule(dynamic, 1)
                    for(auto rowOuter : outerRowsToInvestigate)
                    {
                        for(auto rowInner : innerRowsToInvestigate)
                        {
                            // Calculate approximate move cost                    
                            double cost = Config::getInstance().moveCostOffset + 2 * (Config::getInstance().moveCostOffsetSubmove + 
                                Config::getInstance().moveCostScalingSqrt * M_4TH_ROOT_1_2 + 
                                Config::getInstance().moveCostScalingLinear * M_SQRT1_2);

                            int maxDist = 0;
                            for(size_t i = 0; i < rowOuter->indices.size(); i++)
                            {
                                int dist = abs((int)rowOuter->indices[i] - (int)rowInner->indices[i]);
                                if(dist > maxDist)
                                {
                                    maxDist = dist;
                                }
                            }
                            cost += costPerSubMove(maxDist);
                            
                            std::vector<int> sourceIndices;
                            for(size_t i = 0; i < rowOuter->count; i++)
                            {
                                if((*rowOuter)[i])
                                {
                                    sourceIndices.push_back((int)i);
                                }
                            }
                            std::vector<int> targetIndices;
                            for(size_t i = 0; i < rowInner->count; i++)
                            {
                                if((*rowInner)[i])
                                {
                                    targetIndices.push_back((int)i + colDimCompZone[0]);
                                }
                            }

                            int approxInnerDist = 0;
                            size_t outerOffset = 0, innerOffset = 0;
                            int excessOuterIndices = (int)rowOuter->bitsSet() - (int)rowInner->bitsSet();
                            if(excessOuterIndices >= 0)
                            {
                                outerOffset = excessOuterIndices / 2;
                            }
                            else
                            {
                                innerOffset = -excessOuterIndices / 2;
                            }
                            for(size_t i = 0; i < sourceIndices.size() && i < targetIndices.size(); i++)
                            {
                                int dist = abs(sourceIndices[i + outerOffset] - targetIndices[i + innerOffset]);
                                if(dist > approxInnerDist)
                                {
                                    approxInnerDist = dist;
                                }
                            }
                            cost += costPerSubMove(approxInnerDist);

                            // Calculate number of corrected target sites
                            unsigned int minCols = rowOuter->bitsSet();
                            if(rowInner->bitsSet() < minCols)
                            {
                                minCols = rowInner->bitsSet();
                            }
                            if(colAODLimit < minCols)
                            {
                                minCols = colAODLimit;
                            }
                            unsigned int correctedTargetSites = rowOuter->indices.size() * minCols;
                            unsigned int rowsInCompZone = 0;
                            for(auto index : rowOuter->indices)
                            {
                                if(index >= rowDimCompZone[0] && index < rowDimCompZone[1])
                                {
                                    rowsInCompZone++;
                                }
                            }
                            unsigned colsInCompZone = 0;
                            for(size_t i = colDimCompZone[0]; i < colDimCompZone[1]; i++)
                            {
                                if((*rowOuter)[i])
                                {
                                    colsInCompZone++;
                                }
                            }
                            if(minCols < colsInCompZone)
                            {
                                colsInCompZone = minCols;
                            }
                            correctedTargetSites += rowsInCompZone * colsInCompZone;

                            // Overwrite best-move buffer if new move is best
                            double costPerCorrectedTargetSite = cost / (double)correctedTargetSites;
                            if(correctedTargetSites > 0 && (bestFillableGaps == 0 || costPerCorrectedTargetSite < bestCostPerFilledGap))
                            {
                                #pragma omp critical
                                {
                                    if(correctedTargetSites > 0 && (bestFillableGaps == 0 || 
                                        costPerCorrectedTargetSite < bestCostPerFilledGap))
                                    {
                                        bestStart = RowBitMask(*rowOuter);
                                        bestEnd = RowBitMask(*rowInner);
                                        bestFillableGaps = correctedTargetSites;
                                        bestCostPerFilledGap = costPerCorrectedTargetSite;
                                        bestRowFirst = rowFirst;
                                        bestColLimit = colAODLimit;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Create move for most beneficial locations
    auto move = constructMoveWithSeveralRowsAndCols(stateArray, bestStart, bestEnd, bestRowFirst, compZone, bestColLimit, logger);

    if(move.has_value())
    {
        double costPerCorrectedTargetSite = move.value().cost() / (double)bestFillableGaps;
        logger->info("Best multi row and column move can fill {} at a cost of {} per gap", bestFillableGaps, costPerCorrectedTargetSite);
        std::vector<std::tuple<ParallelMove,int,double>> bestMoves;
        bestMoves.push_back(std::tuple(move.value(), bestFillableGaps, costPerCorrectedTargetSite));
        return bestMoves;
    }
    else
    {
        logger->info("No multi row and column move could be found");
        return std::vector<std::tuple<ParallelMove,int,double>>();
    }
}

std::vector<std::tuple<ParallelMove,int,double>> removeUnwantedAtoms(ArrayAccessor& stateArray, 
    size_t compZone[4], std::shared_ptr<spdlog::logger> logger,
    std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> alreadyMoved, 
    ArrayAccessor& targetGeometry)
{
    std::vector<std::tuple<ParallelMove,int,double>> bestMoves;

    std::vector<std::vector<int>> atomsToBeRemoved;
    atomsToBeRemoved.resize(compZone[1] - compZone[0]);

    for(size_t row = compZone[0]; row < compZone[1]; row++)
    {
        for(size_t col = compZone[2]; col < compZone[3]; col++)
        {
            bool atomPresent = stateArray(row, col);
            bool atomRequired = targetGeometry(row - compZone[0], col - compZone[2]);
            if(!atomPresent && atomRequired)
            {
                logger->debug("No removal move returned as there are unfilled target sites.");
                return bestMoves;
            }
            else if(atomPresent && !atomRequired)
            {
                atomsToBeRemoved[row - compZone[0]].push_back(col);
            }
        }
    }

    if(!Config::getInstance().allowMovesBetweenCols && !Config::getInstance().allowMovesBetweenRows)
    {
        logger->error("Cannot remove atom from within target area without allowing movement between rows or columns.");
        return bestMoves;
    }

    size_t aodColLimit = Config::getInstance().aodColLimit;
    size_t aodRowLimit = Config::getInstance().aodRowLimit;
    size_t aodTotalLimit = Config::getInstance().aodTotalLimit;

    std::set<int> colSelection;
    std::set<int> rowSelection;

    unsigned int removedUnusedAtoms = 0;

    for(size_t row = compZone[0]; row < compZone[1] && rowSelection.size() < aodRowLimit && 
        colSelection.size() * (rowSelection.size() + 1) <= aodTotalLimit ; row++)
    {
        bool allowed = true;
        for(int col : colSelection)
        {
            if(stateArray(row,col) && targetGeometry(row - compZone[0], col - compZone[2]))
            {
                allowed = false;
                break;
            }
        }
        if(!allowed)
        {
            continue;
        }
        bool useRow = false;
        unsigned int newlyRemovableAtoms = 0;
        for(int col : atomsToBeRemoved[row - compZone[0]])
        {
            if(colSelection.contains(col))
            {
                useRow = true;
                newlyRemovableAtoms++;
            }
            else
            {
                // Only allowed if adding one more col satisfies both limits
                allowed = colSelection.size() < aodColLimit && 
                    (colSelection.size() + 1) * (rowSelection.size() + 1) <= aodTotalLimit;
                if(allowed)
                {
                    for(int row : rowSelection)
                    {
                        if(stateArray(row,col) && targetGeometry(row - compZone[0], col - compZone[2]))
                        {
                            allowed = false;
                            break;
                        }
                    }
                    if(allowed)
                    {
                        colSelection.insert(col);
                        useRow = true;
                        newlyRemovableAtoms++;
                    }
                }
            }
        }
        if(useRow)
        {
            rowSelection.insert(row);
            removedUnusedAtoms += newlyRemovableAtoms;
        }
    }

    ParallelMove move;
    ParallelMove::Step start, elbow1, end;
    std::move(rowSelection.begin(), rowSelection.end(), 
        std::inserter(start.rowSelection, start.rowSelection.begin()));
    std::move(colSelection.begin(), colSelection.end(), 
        std::inserter(start.colSelection, start.colSelection.begin()));
    
    if(Config::getInstance().allowMovesBetweenCols)
    {
        int indicesBelowHalf = 0;
        for(double row : start.rowSelection)
        {
            if(row <= (double)stateArray.rows() / 2.)
            {
                indicesBelowHalf++;
            }
        }
        int indicesAboveHalf = start.rowSelection.size() - indicesBelowHalf;

        elbow1.rowSelection = start.rowSelection;
        elbow1.colSelection = start.colSelection;
        for(double& col : elbow1.colSelection)
        {
            col += 0.5;
        }
        for(int i = 0; i < indicesBelowHalf; i++)
        {
            end.rowSelection.push_back(-indicesBelowHalf + i);
            if(Config::getInstance().allowDiagonalMovement)
            {
                elbow1.rowSelection[i] -= 0.5;
            }
        }
        for(int i = 0; i < indicesAboveHalf; i++)
        {
            end.rowSelection.push_back(stateArray.rows() + i);
            if(Config::getInstance().allowDiagonalMovement)
            {
                elbow1.rowSelection[indicesBelowHalf + i] += 0.5;
            }
        }
        end.colSelection = elbow1.colSelection;
    }
    else if(Config::getInstance().allowMovesBetweenRows)
    {
        int indicesBelowHalf = 0;
        for(double col : start.colSelection)
        {
            if(col <= (double)stateArray.cols() / 2)
            {
                indicesBelowHalf++;
            }
        }
        int indicesAboveHalf = start.colSelection.size() - indicesBelowHalf;

        elbow1.rowSelection = start.rowSelection;
        elbow1.colSelection = start.colSelection;
        for(double& row : elbow1.rowSelection)
        {
            row += 0.5;
        }
        for(int i = 0; i < indicesBelowHalf; i++)
        {
            end.colSelection.push_back(-indicesBelowHalf + i);
            if(Config::getInstance().allowDiagonalMovement)
            {
                elbow1.colSelection[i] -= 0.5;
            }
        }
        for(int i = 0; i < indicesAboveHalf; i++)
        {
            end.colSelection.push_back(stateArray.cols() + i);
            if(Config::getInstance().allowDiagonalMovement)
            {
                elbow1.colSelection[indicesBelowHalf + i] += 0.5;
            }
        }
        end.rowSelection = elbow1.rowSelection;
    }
    else
    {
        logger->error("Cannot remove atom from within target area without allowing movement between rows or columns.");
        return bestMoves;
    }

    bestMoves.push_back(std::tuple(move, removedUnusedAtoms, move.cost() / removedUnusedAtoms));
    return bestMoves;
}

std::vector<std::tuple<ParallelMove,int,double>> moveSingleIndex(ArrayAccessor& stateArray, 
    size_t compZone[4], std::shared_ptr<spdlog::logger> logger,
    std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> alreadyMoved, 
    ArrayAccessor& targetGeometry)
{
    std::vector<std::tuple<ParallelMove,int,double>> bestMoves;
    std::vector<size_t> moveRanking;
    double moveCostCutoffThreshold = DBL_MAX;

    if(!Config::getInstance().allowDiagonalMovement && !Config::getInstance().allowMovesBetweenCols && 
        !Config::getInstance().allowMovesBetweenRows)
    {
        logger->error("Single-index move needs any type of movement between rows or columns");
        return bestMoves;
    }
    unsigned int aodTotalLimit = Config::getInstance().aodTotalLimit;

    double baseCost = Config::getInstance().moveCostOffset + 2 * (Config::getInstance().moveCostOffsetSubmove + 
        Config::getInstance().moveCostScalingSqrt * M_4TH_ROOT_1_2 + 
        Config::getInstance().moveCostScalingLinear * M_SQRT1_2);

    for(bool rowFirst : {true, false})
    {
        size_t outerDimCompZone[2];
        size_t innerDimCompZone[2];
        size_t outerSize;
        size_t innerSize;
        unsigned int innerAODLimit = 0;
        unsigned int outerAODLimit = 0;
        fillDimensionDependantData(stateArray, compZone, rowFirst, outerDimCompZone, innerDimCompZone, outerSize, innerSize, outerAODLimit, innerAODLimit);

        std::vector<std::vector<int>> targetIndices;
        targetIndices.resize(outerDimCompZone[1] - outerDimCompZone[0]);
        for(size_t i = outerDimCompZone[0]; i < outerDimCompZone[1]; i++)
        {
            for(size_t j = innerDimCompZone[0]; j < innerDimCompZone[1]; j++)
            {
                if(!accessArrayDim(stateArray, i, j, rowFirst) && 
                    accessArrayDim(targetGeometry, i - outerDimCompZone[0], j - innerDimCompZone[0], rowFirst))
                {
                    targetIndices[i - outerDimCompZone[0]].push_back(j);
                }
            }
        }

        bool disallowLateralMovement = (rowFirst && !Config::getInstance().allowMovesBetweenCols) || 
            (!rowFirst && !Config::getInstance().allowMovesBetweenRows);
        bool disallowLengthwiseMovement = (rowFirst && !Config::getInstance().allowMovesBetweenRows) || 
            (!rowFirst && !Config::getInstance().allowMovesBetweenCols);
        
        size_t iStartStart = 0, iStartEnd = outerSize;
        if(disallowLateralMovement)
        {
            iStartStart = outerDimCompZone[0] - 1;
            if(iStartStart < 0)
            {
                iStartStart = 0;
            }
            iStartEnd = outerDimCompZone[1] + 1;
            if(iStartEnd > outerSize)
            {
                iStartEnd = outerSize;
            }
        }

        // Movement constrictions between rows/columns not yet adhered to
        #pragma omp parallel for schedule(dynamic, 1)
        for(size_t iStart = iStartStart; iStart < iStartEnd; iStart++)
        {
            std::vector<int> sourceIndices;
            for(size_t j = 0; j < innerSize; j++)
            {
                if(accessArrayDim(stateArray, iStart, j, rowFirst))
                {
                    if((iStart < outerDimCompZone[0] || iStart >= outerDimCompZone[1] || 
                        j < innerDimCompZone[0] || j >= innerDimCompZone[1] ||
                        !accessArrayDim(targetGeometry, iStart - outerDimCompZone[0], j - innerDimCompZone[0], rowFirst)) && 
                        (Config::getInstance().allowMultipleMovesPerAtom || !alreadyMoved.has_value() ||
                        !alreadyMoved.value()(rowFirst ? iStart : j, rowFirst ? j : iStart)))
                    {
                        sourceIndices.push_back(j);
                    }
                }
            }
            if(!bestMoves.empty() && baseCost / (double)sourceIndices.size() > moveCostCutoffThreshold)
            {
                continue;
            }
            size_t iTargetStart = outerDimCompZone[0], iTargetEnd = outerDimCompZone[1];
            if(disallowLateralMovement)
            {
                iTargetStart = iStart - 1;
                if(iTargetStart < outerDimCompZone[0])
                {
                    iTargetStart = outerDimCompZone[0];
                }
                iTargetEnd = iStart + 2;
                if(iTargetEnd > outerDimCompZone[1])
                {
                    iTargetEnd = outerDimCompZone[1];
                }
            }
            for(size_t iTarget = iTargetStart; iTarget < iTargetEnd; iTarget++)
            {
                if(!bestMoves.empty() && baseCost / (double)targetIndices[iTarget - outerDimCompZone[0]].size() > 
                    moveCostCutoffThreshold)
                {
                    continue;
                }
                unsigned int outerDist = abs((int)iStart - (int)iTarget);
                unsigned int maxInnerDist = innerSize / 4;
                if(disallowLengthwiseMovement)
                {
                    maxInnerDist = 2;
                }
                for(unsigned int innerDist = 1; innerDist < maxInnerDist; innerDist++)
                {
                    auto pairings = getMostPairings(sourceIndices, targetIndices[iTarget - outerDimCompZone[0]], innerDist);
                    double approxCost = baseCost;
                    if(Config::getInstance().allowDiagonalMovement)
                    {
                        if(outerDist > 1)
                        {
                            approxCost += costPerSubMove(outerDist - 1);
                        }
                        if(innerDist > 1)
                        {
                            approxCost += costPerSubMove(innerDist - 1);
                        }
                    }
                    else
                    {
                        if(outerDist > 0)
                        {
                            approxCost += costPerSubMove(outerDist);
                        }
                        if(innerDist > 0)
                        {
                            approxCost += costPerSubMove(innerDist);
                        }
                    }

                    while(pairings.size() > innerAODLimit || pairings.size() > aodTotalLimit)
                    {
                        pairings.erase(std::next(pairings.rbegin()).base());
                    }

                    int correctedTargetSites = pairings.size();
                    // If it started in the comp zone, it must have been a non-target occupied site
                    if(iStart >= outerDimCompZone[0] && iStart < outerDimCompZone[1])
                    {
                        for(auto& [startIndex, endIndex] : pairings)
                        {
                            if(startIndex >= (int)innerDimCompZone[0] && startIndex < (int)innerDimCompZone[1])
                            {
                                correctedTargetSites++;
                            }
                        }
                    }

                    if(bestMoves.empty() || approxCost / (double)correctedTargetSites <= moveCostCutoffThreshold)
                    {
                        ParallelMove::Step start, end;
                        std::vector<double> *startVector, *endVector;
                        if(rowFirst)
                        {
                            startVector = &start.colSelection;
                            endVector = &end.colSelection;
                            start.rowSelection.push_back(iStart);
                            end.rowSelection.push_back(iTarget);
                        }
                        else
                        {
                            startVector = &start.rowSelection;
                            endVector = &end.rowSelection;
                            start.colSelection.push_back(iStart);
                            end.colSelection.push_back(iTarget);
                        }

                        for(auto& [startIndex, endIndex] : pairings)
                        {
                            startVector->push_back(startIndex);
                            endVector->push_back(endIndex);
                        }

                        ParallelMove move = ParallelMove::fromStartAndEnd(stateArray, start, end, logger);
                        auto costPerCorrectedTargetSite = move.cost() / correctedTargetSites;
                        if(costPerCorrectedTargetSite <= moveCostCutoffThreshold)
                        {
                            #pragma omp critical
                            {
                                if(costPerCorrectedTargetSite <= moveCostCutoffThreshold)
                                {
                                    moveMoveToSortedMoveListIfUseful(bestMoves, moveRanking, 
                                        move, correctedTargetSites, costPerCorrectedTargetSite);
                                    moveCostCutoffThreshold = std::get<2>(bestMoves[moveRanking[0]]) / 
                                        BENEFIT_FRACTION_TO_ALSO_EXECUTE;
                                }
                            }
                        }
                    }
                
                    if(pairings.size() == sourceIndices.size() || 
                        pairings.size() == targetIndices[iTarget - outerDimCompZone[0]].size() || 
                        pairings.size() >= innerAODLimit || pairings.size() >= aodTotalLimit)
                    {
                        break;
                    }
                }
            }
        }
    }

    logger->info("Best single index move function found {} moves", bestMoves.size());
    if(!bestMoves.empty())
    {
        logger->info("Best single index move can fill {} at a cost of {} per gap", std::get<1>(bestMoves[0]), std::get<2>(bestMoves[0]));
    }
    return bestMoves;
}

std::vector<std::tuple<ParallelMove,int,double>> fillRowThroughSubspace(ArrayAccessor& stateArray, 
    size_t compZone[4], std::shared_ptr<spdlog::logger> logger,
    std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>>, 
    ArrayAccessor& targetGeometry)
{
    unsigned int aodTotalLimit = Config::getInstance().aodTotalLimit;

    std::vector<std::tuple<ParallelMove,int,double>> bestMoves;
    std::vector<size_t> moveRanking;
    double moveCostCutoffThreshold = DBL_MAX;

    for(bool rowFirst : {true, false})
    {
        if((rowFirst && !Config::getInstance().allowMovesBetweenRows) || (!rowFirst && !Config::getInstance().allowMovesBetweenCols))
        {
            continue;
        }
        size_t outerDimCompZone[2];
        size_t innerDimCompZone[2];
        size_t outerSize;
        size_t innerSize;
        unsigned int innerAODLimit = 0;
        unsigned int outerAODLimit = 0;
        fillDimensionDependantData(stateArray, compZone, rowFirst, outerDimCompZone, innerDimCompZone, outerSize, innerSize, outerAODLimit, innerAODLimit);

        unsigned int *borderAtomsLeft = new (std::nothrow) unsigned int[outerDimCompZone[1] - outerDimCompZone[0]]();
        if(borderAtomsLeft == 0)
        {
            logger->error("Could not allocate memory");
            return bestMoves;
        }
        unsigned int *borderAtomsRight = new (std::nothrow) unsigned int[outerDimCompZone[1] - outerDimCompZone[0]]();
        if(borderAtomsRight == 0)
        {
            logger->error("Could not allocate memory");
            return bestMoves;
        }
        unsigned int *excessInternalAtoms = new (std::nothrow) unsigned int[outerDimCompZone[1] - outerDimCompZone[0]]();
        if(excessInternalAtoms == 0)
        {
            logger->error("Could not allocate memory");
            return bestMoves;
        }
        unsigned int *emptyCompZoneLocations = new (std::nothrow) unsigned int[outerDimCompZone[1] - outerDimCompZone[0]]();
        if(emptyCompZoneLocations == 0)
        {
            logger->error("Could not allocate memory");
            return bestMoves;
        }
        for(size_t i = outerDimCompZone[0]; i < outerDimCompZone[1]; i++)
        {
            for(size_t j = 0; j < innerSize; j++)
            {
                if(accessArrayDim(stateArray, i, j, rowFirst))
                {
                    if(j < innerDimCompZone[0])
                    {
                        borderAtomsLeft[i - outerDimCompZone[0]]++;
                    }
                    else if(j >= innerDimCompZone[1])
                    {
                        borderAtomsRight[i - outerDimCompZone[0]]++;
                    }
                    else if(!accessArrayDim(targetGeometry, i - outerDimCompZone[0], j - innerDimCompZone[0], rowFirst))
                    {
                        excessInternalAtoms[i - outerDimCompZone[0]]++;
                    }
                }
                else if(j >= innerDimCompZone[0] && j < innerDimCompZone[1] && 
                    accessArrayDim(targetGeometry, i - outerDimCompZone[0], j - innerDimCompZone[0], rowFirst))
                {
                    emptyCompZoneLocations[i - outerDimCompZone[0]]++;
                }
            }
        }
        
        #pragma omp parallel for schedule(dynamic, 1)
        for(size_t iBorder = outerDimCompZone[0]; iBorder < outerDimCompZone[1]; iBorder++)
        {
            const unsigned int& currentBorderAtomsLeft = borderAtomsLeft[iBorder - outerDimCompZone[0]];
            const unsigned int& currentBorderAtomsRight = borderAtomsRight[iBorder - outerDimCompZone[0]];
            const unsigned int& currentExcessInternalAtoms = excessInternalAtoms[iBorder - outerDimCompZone[0]];
            for(size_t iTarget = outerDimCompZone[0]; iTarget < outerDimCompZone[1]; iTarget++)
            {
                unsigned int outerDist = abs((int)iBorder - (int)iTarget);
                if(outerDist > 0)
                {
                    outerDist--;
                }
                const unsigned int& currentEmptyCompZoneLocations = emptyCompZoneLocations[iTarget - outerDimCompZone[0]];

                unsigned int fillableGaps = currentBorderAtomsLeft + currentBorderAtomsRight + currentExcessInternalAtoms;
                if(fillableGaps > currentEmptyCompZoneLocations)
                {
                    fillableGaps = currentEmptyCompZoneLocations;
                }
                if(fillableGaps > innerAODLimit)
                {
                    fillableGaps = innerAODLimit;
                }
                if(fillableGaps > aodTotalLimit)
                {
                    fillableGaps = aodTotalLimit;
                }
                unsigned int correctedSites = fillableGaps;
                if(currentExcessInternalAtoms > currentEmptyCompZoneLocations)
                {
                    correctedSites += currentEmptyCompZoneLocations;
                }
                else
                {
                    correctedSites += currentExcessInternalAtoms;
                }
                
                // 2 * half diagonal step cost
                double minCost = Config::getInstance().moveCostOffset + costPerSubMove(outerDist) + costPerSubMove((correctedSites - 1) / 2) + 2 * (Config::getInstance().moveCostOffsetSubmove + 
                        Config::getInstance().moveCostScalingSqrt * M_4TH_ROOT_1_2 + Config::getInstance().moveCostScalingLinear * M_SQRT1_2);
                if(fillableGaps > 0 && (bestMoves.empty() || minCost / (double)correctedSites <= moveCostCutoffThreshold))
                {
                    // Always use all internal excess atoms as they essentially count twice
                    unsigned int atomsFromOutside = 0;
                    if(fillableGaps > currentExcessInternalAtoms)
                    {
                        atomsFromOutside = fillableGaps - currentExcessInternalAtoms;
                    }
                    unsigned int minFromLeft = atomsFromOutside < currentBorderAtomsRight ? 0 : (atomsFromOutside - currentBorderAtomsRight);
                    unsigned int maxFromLeft = currentBorderAtomsLeft < atomsFromOutside ? currentBorderAtomsLeft : atomsFromOutside;

                    std::optional<ParallelMove> bestMoveInRow = std::nullopt;
                    unsigned int bestDistInRow = 0;

                    unsigned int *distances = new (std::nothrow) unsigned int[fillableGaps];
                    if(distances == 0)
                    {
                        logger->error("Could not allocate memory");
                        continue;
                    }

                    for(unsigned int atomsFromLeft = minFromLeft; atomsFromLeft <= maxFromLeft; atomsFromLeft++)
                    {
                        ParallelMove::Step start;
                        ParallelMove::Step end;
                        if(rowFirst)
                        {
                            start.colSelection.resize(fillableGaps);
                            start.rowSelection.push_back(iBorder);
                            end.colSelection.resize(fillableGaps);
                            end.rowSelection.push_back(iTarget);
                        }
                        else
                        {
                            start.rowSelection.resize(fillableGaps);
                            start.colSelection.push_back(iBorder);
                            end.rowSelection.resize(fillableGaps);
                            end.colSelection.push_back(iTarget);
                        }

                        auto& outerDimStartVector = rowFirst ? start.colSelection : start.rowSelection;
                        auto& outerDimEndVector = rowFirst ? end.colSelection : end.rowSelection;

                        // Fill distances array first with distances from border atoms to compZone[2]
                        unsigned int positionsFound = 0;
                        int currentPosition = innerDimCompZone[0] - 1;
                        for(;currentPosition >= 0 && positionsFound < atomsFromLeft; currentPosition--)
                        {
                            if(accessArrayDim(stateArray, iBorder, currentPosition, rowFirst))
                            {
                                positionsFound++;
                                distances[atomsFromLeft - positionsFound] = innerDimCompZone[0] - currentPosition;
                                outerDimStartVector[atomsFromLeft - positionsFound] = (double)currentPosition;
                            }
                        }
                        positionsFound = 0;
                        currentPosition = innerDimCompZone[0];
                        for(;currentPosition < (int)innerDimCompZone[1] && positionsFound < currentExcessInternalAtoms && 
                            positionsFound < fillableGaps; currentPosition++)
                        {
                            if(accessArrayDim(stateArray, iBorder, currentPosition, rowFirst) && 
                                !accessArrayDim(targetGeometry, iBorder - outerDimCompZone[0], currentPosition - innerDimCompZone[0], rowFirst))
                            {
                                distances[atomsFromLeft + positionsFound] = currentPosition;
                                outerDimStartVector[atomsFromLeft + positionsFound] = (double)currentPosition;
                                positionsFound++;
                            }
                        }
                        positionsFound = 0;
                        currentPosition = innerDimCompZone[1];
                        for(;currentPosition < (int)innerSize && positionsFound < atomsFromOutside - atomsFromLeft; currentPosition++)
                        {
                            if(accessArrayDim(stateArray, iBorder, currentPosition, rowFirst))
                            {
                                distances[atomsFromLeft + currentExcessInternalAtoms + positionsFound] = 
                                    currentPosition - (innerDimCompZone[1] - 1);
                                outerDimStartVector[atomsFromLeft + currentExcessInternalAtoms + positionsFound] = (double)currentPosition;
                                positionsFound++;
                            }
                        }
                        // Now add distances to gaps (from compZone[2])
                        positionsFound = 0;
                        currentPosition = innerDimCompZone[0];
                        for(;currentPosition < (int)innerDimCompZone[1] && positionsFound < atomsFromLeft; currentPosition++)
                        {
                            if(!accessArrayDim(stateArray, iTarget, currentPosition, rowFirst) && 
                                accessArrayDim(targetGeometry, iTarget - outerDimCompZone[0], currentPosition - innerDimCompZone[0], rowFirst))
                            {
                                distances[positionsFound] += currentPosition - innerDimCompZone[0];
                                outerDimEndVector[positionsFound] = (double)currentPosition;
                                positionsFound++;
                            }
                        }
                        positionsFound = 0;
                        for(;currentPosition < (int)innerDimCompZone[1] && positionsFound < currentExcessInternalAtoms && 
                            positionsFound < fillableGaps; currentPosition++)
                        {
                            if(!accessArrayDim(stateArray, iTarget, currentPosition, rowFirst) && 
                                accessArrayDim(targetGeometry, iTarget - outerDimCompZone[0], currentPosition - innerDimCompZone[0], rowFirst))
                            {
                                distances[atomsFromLeft + positionsFound] = abs(currentPosition - (int)distances[positionsFound]);
                                outerDimEndVector[atomsFromLeft + positionsFound] = (double)currentPosition;
                                positionsFound++;
                            }
                        }
                        positionsFound = 0;
                        currentPosition = innerDimCompZone[1] - 1;
                        for(;currentPosition >= (int)innerDimCompZone[0] && positionsFound < atomsFromOutside - atomsFromLeft; currentPosition--)
                        {
                            if(!accessArrayDim(stateArray, iTarget, currentPosition, rowFirst) && 
                                accessArrayDim(targetGeometry, iTarget - outerDimCompZone[0], currentPosition - innerDimCompZone[0], rowFirst))
                            {
                                distances[fillableGaps - positionsFound - 1] += innerDimCompZone[1] - 1 - currentPosition;
                                outerDimEndVector[fillableGaps - positionsFound - 1] = (double)currentPosition;
                                positionsFound++;
                            }
                        }

                        unsigned int dist = *std::max_element(distances, distances + fillableGaps);

                        if(!bestMoveInRow.has_value() || dist < bestDistInRow)
                        {
                            bestMoveInRow = ParallelMove::fromStartAndEnd(stateArray, start, end, logger);
                            bestDistInRow = dist;
                        }
                    }

                    if(bestMoveInRow.has_value())
                    {
                        double costPerCorrectedSite = bestMoveInRow.value().cost() / (double)correctedSites;
                        if(bestMoves.empty() || costPerCorrectedSite <= moveCostCutoffThreshold)
                        {
                            #pragma omp critical
                            {
                                if(bestMoves.empty() || costPerCorrectedSite <= moveCostCutoffThreshold)
                                {
                                    moveMoveToSortedMoveListIfUseful(bestMoves, moveRanking, bestMoveInRow.value(), correctedSites, costPerCorrectedSite);
                                    moveCostCutoffThreshold = std::get<2>(bestMoves[0]) / BENEFIT_FRACTION_TO_ALSO_EXECUTE;
                                }
                            }
                        }
                    }

                    delete[] distances;
                }
            }
        }
        delete[] borderAtomsLeft;
        delete[] borderAtomsRight;
        delete[] excessInternalAtoms;
        delete[] emptyCompZoneLocations;
    }

    logger->info("Best subspace linear move function found {} moves", bestMoves.size());
    if(!bestMoves.empty())
    {
        logger->info("Best subspace linear move can fill {} at a cost of {} per gap", std::get<1>(bestMoves[0]), std::get<2>(bestMoves[0]));
    }
    return bestMoves;
}

std::vector<std::tuple<ParallelMove,int,double>> fillRowSidesDirectly(ArrayAccessor& stateArray, 
    size_t compZone[4], std::shared_ptr<spdlog::logger> logger,
    std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> alreadyMoved, 
    ArrayAccessor& targetGeometry)
{
    std::vector<std::tuple<ParallelMove,int,double>> bestMoves;
    std::vector<size_t> moveRanking;
    double moveCostCutoffThreshold = DBL_MAX;

    for(bool rowFirst : {true, false})
    {
        size_t outerDimCompZone[2];
        size_t innerDimCompZone[2];
        size_t outerSize;
        size_t innerSize;
        unsigned int innerAODLimit = 0;
        unsigned int outerAODLimit = 0;
        fillDimensionDependantData(stateArray, compZone, rowFirst, outerDimCompZone, 
            innerDimCompZone, outerSize, innerSize, outerAODLimit, innerAODLimit);

        #pragma omp parallel for schedule(dynamic,1)
        for(size_t i = outerDimCompZone[0]; i < outerDimCompZone[1]; i++)
        {
            unsigned int excessAtomsLeft = 0, excessAtomsRight = 0;
            for(size_t j = 0; j < innerDimCompZone[0]; j++)
            {
                if(accessArrayDim(stateArray, i, j, rowFirst))
                {
                    excessAtomsLeft++;
                }
            }
            for(size_t j = innerDimCompZone[1]; j < innerSize; j++)
            {
                if(accessArrayDim(stateArray, i, j, rowFirst))
                {
                    excessAtomsRight++;
                }
            }

            size_t leftPosition = innerDimCompZone[0], rightPosition = innerDimCompZone[1] - 1;
            unsigned int aodsUsedLeft = 0, aodsUsedRight = 0, correctedSites = 0;
            unsigned int atomsFromLeft = 0;
            unsigned int atomsFromRight = 0;
            while(leftPosition <= rightPosition && (excessAtomsLeft > 0 || excessAtomsRight > 0))
            {
                std::optional<unsigned int> lowestDistToNextGap = std::nullopt;
                bool fromLeft = true;
                size_t newPosition = 0;
                unsigned int requiredTones = 0;
                unsigned int additionalAtomsLeft = 0, additionalAtomsRight = 0;
                if(excessAtomsLeft > 0)
                {
                    for(size_t j = leftPosition; j <= rightPosition; j++)
                    {
                        if(accessArrayDim(targetGeometry, i - outerDimCompZone[0], j - innerDimCompZone[0], rowFirst))
                        {
                            requiredTones++;
                            if(!accessArrayDim(stateArray, i, j, rowFirst))
                            {
                                lowestDistToNextGap = j - leftPosition + 1;
                                newPosition = j + 1;
                                break;
                            }
                        }
                        else if(accessArrayDim(stateArray, i, j, rowFirst))
                        {
                            additionalAtomsLeft++;
                        }
                    }
                }
                if(excessAtomsRight > 0)
                {
                    unsigned int requiredTonesRight = 0;
                    for(size_t j = rightPosition; j >= leftPosition; j--)
                    {
                        if(accessArrayDim(targetGeometry, i - outerDimCompZone[0], j - innerDimCompZone[0], rowFirst))
                        {
                            requiredTonesRight++;
                            if(!accessArrayDim(stateArray, i, j, rowFirst))
                            {
                                if(!lowestDistToNextGap.has_value() || (rightPosition - j < lowestDistToNextGap.value()))
                                {
                                    lowestDistToNextGap = rightPosition - j + 1;
                                    fromLeft = false;
                                    newPosition = j - 1;
                                    requiredTones = requiredTonesRight;
                                }
                                break;
                            }
                        }
                        else if(accessArrayDim(stateArray, i, j, rowFirst))
                        {
                            additionalAtomsRight++;
                        }
                    }
                }

                if(!lowestDistToNextGap.has_value() || (aodsUsedLeft + aodsUsedRight + requiredTones > Config::getInstance().aodTotalLimit) || 
                    (aodsUsedLeft + aodsUsedRight + requiredTones > innerAODLimit))
                {
                    break;
                }
                else
                {
                    correctedSites++;
                    if(fromLeft)
                    {
                        leftPosition = newPosition;
                        excessAtomsLeft--;
                        excessAtomsLeft += additionalAtomsLeft;
                        atomsFromLeft++;
                        aodsUsedLeft += requiredTones;
                        correctedSites += additionalAtomsLeft;
                    }
                    else
                    {
                        rightPosition = newPosition;
                        excessAtomsRight--;
                        excessAtomsRight += additionalAtomsRight;
                        atomsFromRight++;
                        aodsUsedRight += requiredTones;
                        correctedSites += additionalAtomsRight;
                    }
                }
            }
            double minCost = Config::getInstance().moveCostOffset + costPerSubMove(1);
            if(bestMoves.empty() || (minCost / (double)correctedSites <= moveCostCutoffThreshold))
            {
                ParallelMove::Step start;
                ParallelMove::Step end;

                if(rowFirst)
                {
                    start.colSelection.resize(aodsUsedLeft + aodsUsedRight);
                    start.rowSelection.push_back(i);
                    end.colSelection.resize(aodsUsedLeft + aodsUsedRight);
                    end.rowSelection.push_back(i);
                }
                else
                {
                    start.rowSelection.resize(aodsUsedLeft + aodsUsedRight);
                    start.colSelection.push_back(i);
                    end.rowSelection.resize(aodsUsedLeft + aodsUsedRight);
                    end.colSelection.push_back(i);
                }

                auto& outerDimStartVector = rowFirst ? start.colSelection : start.rowSelection;
                auto& outerDimEndVector = rowFirst ? end.colSelection : end.rowSelection;

                int actuallyCorrectedSites = 0;
                int j = leftPosition - 1;
                unsigned int startTonesRemaining = aodsUsedLeft;
                unsigned int endTonesRemaining = aodsUsedLeft;
                bool moveAllowed = true;
                for(; j >= 0 && (startTonesRemaining > 0 || endTonesRemaining > 0); j--)
                {
                    if(j >= (int)innerDimCompZone[0] &&
                        accessArrayDim(targetGeometry, i - outerDimCompZone[0], j - innerDimCompZone[0], rowFirst))
                    {
                        if(!accessArrayDim(stateArray, i, j, rowFirst))
                        {
                            actuallyCorrectedSites++;
                        }
                        if(endTonesRemaining > 0)
                        {
                            outerDimEndVector[--endTonesRemaining] = j;
                        }
                        else
                        {
                            logger->warn("Too many target positions for fillRowSidesDirectly to handle. Skipping index");
                            moveAllowed = false;
                            break;
                        }
                    }
                    if(accessArrayDim(stateArray, i, j, rowFirst))
                    {
                        if(j >= (int)innerDimCompZone[0] &&
                            !accessArrayDim(targetGeometry, i - outerDimCompZone[0], j - innerDimCompZone[0], rowFirst))
                        {
                            actuallyCorrectedSites++;
                        }
                        if(startTonesRemaining > 0)
                        {
                            startTonesRemaining--;
                            outerDimStartVector[startTonesRemaining] = j;
                        }
                        else
                        {
                            logger->info("Too many atoms in section for fillRowSidesDirectly to handle. Skipping index");
                            moveAllowed = false;
                            break;
                        }
                    }
                }
                if(!moveAllowed)
                {
                    continue;
                }
                if(startTonesRemaining > 0 || endTonesRemaining > 0)
                {
                    logger->error("Not all tones could be filled for move from low indices. This point in fillRowSidesDirectly should never be reached!");
                    continue;
                }
                j = rightPosition + 1;
                startTonesRemaining = aodsUsedRight;
                endTonesRemaining = aodsUsedRight;
                for(; j < (int)innerSize && (startTonesRemaining > 0 || endTonesRemaining > 0); j++)
                {
                    if(j < (int)innerDimCompZone[1] &&
                        accessArrayDim(targetGeometry, i - outerDimCompZone[0], j - innerDimCompZone[0], rowFirst))
                    {
                        if(!accessArrayDim(stateArray, i, j, rowFirst))
                        {
                            actuallyCorrectedSites++;
                        }
                        if(endTonesRemaining > 0)
                        {
                            outerDimEndVector[aodsUsedLeft + aodsUsedRight - endTonesRemaining] = j;
                            endTonesRemaining--;
                        }
                        else
                        {
                            logger->warn("Too many target positions for fillRowSidesDirectly to handle. Skipping index");
                            moveAllowed = false;
                            break;
                        }
                    }
                    if(accessArrayDim(stateArray, i, j, rowFirst))
                    {
                        if(j < (int)innerDimCompZone[1] &&
                            !accessArrayDim(targetGeometry, i - outerDimCompZone[0], j - innerDimCompZone[0], rowFirst))
                        {
                            actuallyCorrectedSites++;
                        }
                        if(startTonesRemaining > 0)
                        {
                            outerDimStartVector[aodsUsedLeft + aodsUsedRight - startTonesRemaining] = j;
                            startTonesRemaining--;
                        }
                        else
                        {
                            logger->info("Too many atoms in section for fillRowSidesDirectly to handle. Skipping index");
                            moveAllowed = false;
                            break;
                        }
                    }
                }
                if(!moveAllowed)
                {
                    continue;
                }
                if(startTonesRemaining > 0 || endTonesRemaining > 0)
                {
                    logger->warn("Not all tones could be filled for move from high indices. This point in fillRowSidesDirectly should never be reached!");
                    continue;
                }

                if(!Config::getInstance().allowMultipleMovesPerAtom && alreadyMoved.has_value())
                {
                    for(size_t row : start.rowSelection)
                    {
                        for(size_t col : start.colSelection)
                        {
                            if(alreadyMoved.value()(row,col))
                            {
                                moveAllowed = false;
                                break;
                            }
                        }
                        if(!moveAllowed)
                        {
                            break;
                        }
                    }
                    if(!moveAllowed)
                    {
                        continue;
                    }
                }
                
                ParallelMove move;
                move.steps.push_back(std::move(start));
                move.steps.push_back(std::move(end));
                double costPerCorrectedSite = move.cost() / (double)actuallyCorrectedSites;

                if(bestMoves.empty() || costPerCorrectedSite <= moveCostCutoffThreshold)
                {
                    #pragma omp critical
                    {
                        if(bestMoves.empty() || costPerCorrectedSite <= moveCostCutoffThreshold)
                        {
                            moveMoveToSortedMoveListIfUseful(bestMoves, moveRanking, move, actuallyCorrectedSites, costPerCorrectedSite);
                            moveCostCutoffThreshold = std::get<2>(bestMoves[0]) / BENEFIT_FRACTION_TO_ALSO_EXECUTE;
                        }
                    }
                }
            }
        }
    }

    logger->info("Best direct move function found {} moves", bestMoves.size());
    if(!bestMoves.empty())
    {
        logger->info("Best direct move can fill {} at a cost of {} per gap", std::get<1>(bestMoves[0]), std::get<2>(bestMoves[0]));
    }
    return bestMoves;
}

bool analyzeArray(ArrayAccessor& stateArray, size_t compZone[4], int& incorrectTargetSites, 
    ArrayAccessor& targetGeometry, std::shared_ptr<spdlog::logger> logger)
{
    incorrectTargetSites = 0;
    for(size_t row = 0; row < (size_t)stateArray.rows(); row++)
    {
        for(size_t col = 0; col < (size_t)stateArray.cols(); col++)
        {
            if(row >= compZone[0] && row < compZone[1] && col >= compZone[2] && col < compZone[3] && 
                (stateArray(row,col) != targetGeometry(row - compZone[0], col - compZone[2])))
            {
                incorrectTargetSites++;
            }
        }
    }
    logger->debug("{} incorrect target sites (analyzeArray)", incorrectTargetSites);
    return true;
}

bool findNextMove(ArrayAccessor& stateArray, size_t compZone[4], std::vector<ParallelMove>& moves, 
    bool& sorted, int& incorrectTargetSites, ArrayAccessor& targetGeometry, std::shared_ptr<spdlog::logger> logger,
    std::optional<py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>> alreadyMoved,
    std::vector<std::chrono::nanoseconds>& timePerMoveFunction)
{
    logger->debug("Finding next best move");
    std::vector<std::tuple<ParallelMove,int,double>> bestMoves;
    std::vector<size_t> moveRanking;
    auto startTime = std::chrono::steady_clock::now();

    for(size_t i = 0; const auto& function : {fillRowSidesDirectly, fillRowThroughSubspace,
        moveSeveralRowsAndCols, removeUnwantedAtoms, moveSingleIndex})
    {
        for(auto [move, correctedTargetSites, costPerCorrectedTargetSite] : 
            function(stateArray, compZone, logger, alreadyMoved, targetGeometry))
        {
            if(correctedTargetSites > 0)
            {
                for(const auto& improvingFunction : {improveComplexMove, improveMoveByAddingIndependentAtom})
                {
                    auto [improvedMove, improvedCorrectedTargetSites, improvedCostPerCorrectedTargetSite] = improvingFunction(stateArray, 
                        compZone, logger, move, move.cost(), correctedTargetSites, alreadyMoved, targetGeometry);
                    if(improvedMove.has_value() && improvedCostPerCorrectedTargetSite < costPerCorrectedTargetSite)
                    {
                        move = improvedMove.value();
                        correctedTargetSites = improvedCorrectedTargetSites;
                        costPerCorrectedTargetSite = improvedCostPerCorrectedTargetSite;
                    }
                }
                moveMoveToSortedMoveListIfUseful(bestMoves, moveRanking, move, correctedTargetSites, costPerCorrectedTargetSite);
            }
        }
        auto duration = std::chrono::steady_clock::now() - startTime;
        startTime = std::chrono::steady_clock::now();
        if(timePerMoveFunction.size() <= i)
        {
            timePerMoveFunction.push_back(std::chrono::nanoseconds::zero());
        }
        timePerMoveFunction[i++] += duration;
    }

    if(bestMoves.empty())
    {
        logger->error("Couldn't find another move");
        return false;
    }
    else
    {
        auto& [move, bestCorrectedTargetSites, bestCostPerCorrectedTargetSite] = bestMoves[moveRanking[0]];
        if(!move.execute(stateArray, logger, alreadyMoved))
        {
            logger->error("Error when executing move");
            return false;
        }
        incorrectTargetSites -= bestCorrectedTargetSites;
        moves.push_back(move);

        for(size_t i = 1; i < bestMoves.size(); i++)
        {
            auto [nextBestMove, nextBestCorrectedTargetSites, nextBestCostPerCorrectedTargetSite] = bestMoves[moveRanking[i]];
            std::optional<int> updatedCorrectedTargetSite = calcCorrectedTargetSite(nextBestMove, stateArray,
                compZone, logger, alreadyMoved, targetGeometry);
            if(!updatedCorrectedTargetSite.has_value() || updatedCorrectedTargetSite.value() <= 0)
            {
                logger->info("Next best move skipped as it could not be confirmed");
                continue;
            }
            else if(updatedCorrectedTargetSite.value() <= 0)
            {
                logger->info("Next best move skipped as it corrects {} target sites", 
                    updatedCorrectedTargetSite.value());
            }
            else
            {
                nextBestCorrectedTargetSites = updatedCorrectedTargetSite.value();
                logger->info("Investigated next-best move has cost per corrected site {}, threshold: {}", 
                    nextBestCostPerCorrectedTargetSite, bestCostPerCorrectedTargetSite / BENEFIT_FRACTION_TO_ALSO_EXECUTE);
                if(nextBestCostPerCorrectedTargetSite <= bestCostPerCorrectedTargetSite / BENEFIT_FRACTION_TO_ALSO_EXECUTE)
                {
                    if(!nextBestMove.execute(stateArray, logger, alreadyMoved))
                    {
                        logger->error("Error when executing move");
                        return false;
                    }
                    else
                    {
                        // Save move but don't save to list of last costs as to not deteriorate performance too much
                        incorrectTargetSites -= nextBestCorrectedTargetSites;
                        moves.push_back(nextBestMove);
                    }
                }
                else
                {
                    break;
                }
            }
        }

        if(logger->level() <= spdlog::level::info)
        {
            std::stringstream strstream;
            strstream << "State after move execution: \n";
            for(size_t r = 0; r < (size_t)stateArray.rows(); r++)
            {
                for(size_t c = 0; c < (size_t)stateArray.cols(); c++)
                {
                    if(r >= compZone[0] && r < compZone[1] && c >= compZone[2] && c < compZone[3])
                    {
                        if(targetGeometry(r - compZone[0], c - compZone[2]))
                        {
                            strstream << (stateArray(r,c) != 0 ? "█" : "▒");
                        }
                        else
                        {
                            strstream << (stateArray(r,c) != 0 ? "X" : " ");
                        }
                    }
                    else
                    {
                        strstream << (stateArray(r,c) != 0 ? "█" : " ");
                    }
                }
                strstream << "\n";
            }
            logger->info(strstream.str());
        }
        logger->debug("{} incorrect target sites", incorrectTargetSites);
    }
    int incorrectTargetSitesAA;
    analyzeArray(stateArray, compZone, incorrectTargetSitesAA, targetGeometry, logger);
    if(incorrectTargetSites != incorrectTargetSitesAA)
    {
        return false;
    }

    if(incorrectTargetSites == 0)
    {
        sorted = true;
        logger->debug("Sorting done");
    }
    else if(incorrectTargetSites < 0)
    {
        logger->error("Too many sites corrected. This point should never be reached.");
        return false;
    }
    return true;
}

std::optional<std::vector<ParallelMove>> sortParallelInternal(
    ArrayAccessor& stateArray, size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    ArrayAccessor& targetGeometry, std::shared_ptr<spdlog::logger> logger)
{
    std::optional<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> alreadyMoved = std::nullopt;
    if(!Config::getInstance().allowMultipleMovesPerAtom)
    {
        alreadyMoved = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Constant(
            stateArray.rows(), stateArray.cols(), false);
    }

    size_t compZone[4] = {compZoneRowStart, compZoneRowEnd, compZoneColStart, compZoneColEnd};
    std::vector<ParallelMove> moves;
    if(compZoneRowStart >= compZoneRowEnd || compZoneRowEnd > (size_t)stateArray.rows() || 
        compZoneColStart >= compZoneColEnd || compZoneColEnd > (size_t)stateArray.cols())
    {
        logger->error("No suitable arguments");
        return std::nullopt;
    }
    int incorrectTargetSites = 0;
    analyzeArray(stateArray, compZone, incorrectTargetSites, targetGeometry, logger);
    bool sorted = false;

    std::vector<std::chrono::nanoseconds> timePerMoveFunction;

    while(!sorted)
    {
        if(alreadyMoved.has_value())
        {
            if(!findNextMove(stateArray, compZone, moves, sorted, incorrectTargetSites, 
                targetGeometry, logger, alreadyMoved.value(), timePerMoveFunction))
            {
                break;
            }
        }
        else
        {
            if(!findNextMove(stateArray, compZone, moves, sorted, incorrectTargetSites, 
                targetGeometry, logger, std::nullopt, timePerMoveFunction))
            {
                break;
            }
        }
    }

    for(int i = 0; auto duration : timePerMoveFunction)
    {
        logger->info("Total time taken by move function {}: {}ns", i++, duration.count());
    }

    if(!sorted)
    {
        logger->error("No move could be found");
        analyzeArray(stateArray, compZone, incorrectTargetSites, targetGeometry, logger);
        return std::nullopt;
    }
    if(Config::getInstance().alwaysGenerateAllAODTones)
    {
        for(auto& move : moves)
        {
            move.extendToUseAllTones(stateArray.rows(), stateArray.cols(), logger, false);
        }
    }

    return moves;
}

std::optional<std::vector<ParallelMove>> sortParallel(
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& stateArray, 
    size_t compZoneRowStart, size_t compZoneRowEnd, size_t compZoneColStart, size_t compZoneColEnd, 
    py::EigenDRef<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &targetGeometry)
{
    std::shared_ptr<spdlog::logger> logger = Config::getInstance().getParallelLogger();
    omp_set_num_threads(NUM_THREADS);

    EigenArrayAccessor stateArrayAccessor(stateArray);
    EigenArrayAccessor targetGeometryArrayAccessor(targetGeometry);
    return sortParallelInternal(stateArrayAccessor, compZoneRowStart, 
        compZoneRowEnd, compZoneColStart, compZoneColEnd, targetGeometryArrayAccessor, logger);
}
