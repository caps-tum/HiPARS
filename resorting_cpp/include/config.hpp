#pragma once

#include <fstream>
#include <ranges>
#include <string>
#include <spdlog/spdlog.h>
#include <stdexcept>

#include "spdlog/sinks/basic_file_sink.h"

#define DOUBLE_EQUIVALENCE_THRESHOLD 0.00001
#define M_4TH_ROOT_2 1.1892071150027210667
#define M_4TH_ROOT_1_2 1 / M_4TH_ROOT_2

class Config
{
    public:
        static Config& getInstance()
        {
            static Config instance;
            return instance;
        }
    private:
        Config() : sequentialLogger(nullptr), parallelLogger(nullptr), greedyLatticeLogger(nullptr), latticeByRowLogger(nullptr), 
            logFileName("sorting.log"), sequentialLoggerName("sequentialSortingLogger"), parallelLoggerName("parallelSortingLogger"),
            greedyLatticeLoggerName("greedyLatticeSortingLogger"), latticeByRowLoggerName("latticeByRowSortingLogger"), logLevel("info"),
            rowSpacing(1), columnSpacing(1), minAodSpacing(0), allowMovingEmptyTrapOntoOccupied(true), allowDiagonalMovement(true), 
            allowMovesBetweenRows(true), allowMovesBetweenCols(true), allowMultipleMovesPerAtom(false), alwaysGenerateAllAODTones(false), 
            verticalSortingChannel(std::nullopt), aodTotalLimit(256), aodRowLimit(16), aodColLimit(16), moveCostOffset(150), 
            moveCostOffsetSubmove(0), moveCostScalingSqrt(0), moveCostScalingLinear(0.1),
            recommendedDistFromOccSites(1), recommendedDistFromEmptySites(0.1), minDistFromOccSites(1), maxSubmoveDistInPenalizedArea(1.5) {}

        std::shared_ptr<spdlog::logger> sequentialLogger, parallelLogger, greedyLatticeLogger, latticeByRowLogger;
    public:
        Config(Config const&) = delete;
        void operator=(Config const&) = delete;
        void flushLogs()
        {
            if(this->sequentialLogger != nullptr)
            {
                this->sequentialLogger->flush();
            }
            if(this->parallelLogger != nullptr)
            {
                this->parallelLogger->flush();
            }
            if(this->greedyLatticeLogger != nullptr)
            {
                this->greedyLatticeLogger->flush();
            }
            if(this->latticeByRowLogger != nullptr)
            {
                this->latticeByRowLogger->flush();
            }
        };
        bool setProperty(std::string key, std::string val)
        {
            if(key.compare("logFileName") == 0)
            {
                this->logFileName = val;
                std::shared_ptr<spdlog::logger> logger;
                this->sequentialLogger = nullptr;
                this->parallelLogger = nullptr;
                this->greedyLatticeLogger = nullptr;
                this->latticeByRowLogger = nullptr;
            }
            else if(key.compare("sequentialLoggerName") == 0)
            {
                this->sequentialLoggerName = val;
                this->sequentialLogger = nullptr;
            }
            else if(key.compare("parallelLoggerName") == 0)
            {
                this->parallelLoggerName = val;
                this->parallelLogger = nullptr;
            }
            else if(key.compare("greedyLatticeLoggerName") == 0)
            {
                this->greedyLatticeLoggerName = val;
                this->greedyLatticeLogger = nullptr;
            }
            else if(key.compare("latticeByRowLoggerName") == 0)
            {
                this->latticeByRowLoggerName = val;
                this->latticeByRowLogger = nullptr;
            }
            else if(key.compare("logLevel") == 0)
            {
                this->logLevel = val;
                if(this->sequentialLogger != nullptr)
                {
                    this->sequentialLogger->set_level(spdlog::level::from_str(this->logLevel));
                }
                if(this->parallelLogger != nullptr)
                {
                    this->parallelLogger->set_level(spdlog::level::from_str(this->logLevel));
                }
                if(this->greedyLatticeLogger != nullptr)
                {
                    this->greedyLatticeLogger->set_level(spdlog::level::from_str(this->logLevel));
                }
                if(this->latticeByRowLogger != nullptr)
                {
                    this->latticeByRowLogger->set_level(spdlog::level::from_str(this->logLevel));
                }
            }
            else if(key.compare("rowSpacing") == 0)
            {
                this->rowSpacing = std::stod(val);
            }
            else if(key.compare("columnSpacing") == 0)
            {
                this->columnSpacing = std::stod(val);
            }
            else if(key.compare("minAodSpacing") == 0)
            {
                this->minAodSpacing = std::stod(val);
            }
            else if(key.compare("allowMovingEmptyTrapOntoOccupied") == 0)
            {
                std::transform(val.begin(), val.end(), val.begin(),
                    [](unsigned char c){ return std::tolower(c); });
                this->allowMovingEmptyTrapOntoOccupied = val.compare("true") == 0;
            }
            else if(key.compare("allowDiagonalMovement") == 0)
            {
                std::transform(val.begin(), val.end(), val.begin(),
                    [](unsigned char c){ return std::tolower(c); });
                this->allowDiagonalMovement = val.compare("true") == 0;
            }
            else if(key.compare("allowMovesBetweenRows") == 0)
            {
                std::transform(val.begin(), val.end(), val.begin(),
                    [](unsigned char c){ return std::tolower(c); });
                this->allowMovesBetweenRows = val.compare("true") == 0;
            }
            else if(key.compare("allowMovesBetweenCols") == 0)
            {
                std::transform(val.begin(), val.end(), val.begin(),
                    [](unsigned char c){ return std::tolower(c); });
                this->allowMovesBetweenCols = val.compare("true") == 0;
            }
            else if(key.compare("allowMultipleMovesPerAtom") == 0)
            {
                std::transform(val.begin(), val.end(), val.begin(),
                    [](unsigned char c){ return std::tolower(c); });
                this->allowMultipleMovesPerAtom = val.compare("true") == 0;
            }
            else if(key.compare("alwaysGenerateAllAODTones") == 0)
            {
                std::transform(val.begin(), val.end(), val.begin(),
                    [](unsigned char c){ return std::tolower(c); });
                this->alwaysGenerateAllAODTones = val.compare("true") == 0;
            }
            else if(key.compare("verticalSortingChannel") == 0)
            {
                std::transform(val.begin(), val.end(), val.begin(),
                    [](unsigned char c){ return std::tolower(c); });
                this->verticalSortingChannel = val.compare("true") == 0;
            }
            else if(key.compare("aodTotalLimit") == 0)
            {
                this->aodTotalLimit = std::stoi(val);
            }
            else if(key.compare("aodRowLimit") == 0)
            {
                this->aodRowLimit = std::stoi(val);
            }
            else if(key.compare("aodColLimit") == 0)
            {
                this->aodColLimit = std::stoi(val);
            }
            else if(key.compare("moveCostOffset") == 0)
            {
                this->moveCostOffset = stod(val);
            }
            else if(key.compare("moveCostOffsetSubmove") == 0)
            {
                this->moveCostOffsetSubmove = stod(val);
            }
            else if(key.compare("moveCostScalingSqrt") == 0)
            {
                this->moveCostScalingSqrt = stod(val);
            }
            else if(key.compare("moveCostScalingLinear") == 0)
            {
                this->moveCostScalingLinear = stod(val);
            }
            else if(key.compare("recommendedDistFromOccSites") == 0)
            {
                this->recommendedDistFromOccSites = stod(val);
            }
            else if(key.compare("recommendedDistFromEmptySites") == 0)
            {
                this->recommendedDistFromEmptySites = stod(val);
            }
            else if(key.compare("minDistFromOccSites") == 0)
            {
                this->minDistFromOccSites = stod(val);
            }
            else if(key.compare("maxSubmoveDistInPenalizedArea") == 0)
            {
                this->maxSubmoveDistInPenalizedArea = stod(val);
            }
            else
            {
                return false;
            }
            return true;
        }
        bool readConfig(std::string filePath)
        {
            std::ifstream configFile(filePath, std::ios_base::in);
            std::string line;
            
            if(!configFile.is_open())
            {
                return false;
            }

            // Set all given values, ignore all other lines
            while(std::getline(configFile, line))
            {
                std::istringstream configLine(line);
                auto delim = line.find('=');
                if(delim != std::string::npos)
                {
                    std::string key = line.substr(0, delim);
                    std::string::iterator end_pos = std::remove(key.begin(), key.end(), ' ');
                    key.erase(end_pos, key.end());
                    std::string val = line.substr(delim + 1);
                    end_pos = std::remove(val.begin(), val.end(), ' ');
                    val.erase(end_pos, val.end());
                    this->setProperty(key, val);
                }
            }
            return true;
        }
        std::shared_ptr<spdlog::logger> getSequentialLogger()
        {
            if(this->sequentialLogger == nullptr)
            {
                if((this->sequentialLogger = spdlog::get(this->sequentialLoggerName)) == nullptr)
                {
                    this->sequentialLogger = spdlog::basic_logger_mt(this->sequentialLoggerName, this->logFileName);
                }
                this->sequentialLogger->set_level(spdlog::level::from_str(Config::getInstance().logLevel));
            }
            return this->sequentialLogger;
        }
        std::shared_ptr<spdlog::logger> getParallelLogger()
        {
            if(this->parallelLogger == nullptr)
            {
                if((this->parallelLogger = spdlog::get(this->parallelLoggerName)) == nullptr)
                {
                    this->parallelLogger = spdlog::basic_logger_mt(this->parallelLoggerName, this->logFileName);
                }
                this->parallelLogger->set_level(spdlog::level::from_str(Config::getInstance().logLevel));
            }
            return this->parallelLogger;
        }
        std::shared_ptr<spdlog::logger> getGreedyLatticeLogger()
        {
            if(this->greedyLatticeLogger == nullptr)
            {
                if((this->greedyLatticeLogger = spdlog::get(this->greedyLatticeLoggerName)) == nullptr)
                {
                    this->greedyLatticeLogger = spdlog::basic_logger_mt(this->greedyLatticeLoggerName, this->logFileName);
                }
                this->greedyLatticeLogger->set_level(spdlog::level::from_str(Config::getInstance().logLevel));
            }
            return this->greedyLatticeLogger;
        }
        std::shared_ptr<spdlog::logger> getLatticeByRowLogger()
        {
            if(this->latticeByRowLogger == nullptr)
            {
                if((this->latticeByRowLogger = spdlog::get(this->latticeByRowLoggerName)) == nullptr)
                {
                    this->latticeByRowLogger = spdlog::basic_logger_mt(this->latticeByRowLoggerName, this->logFileName);
                }
                this->latticeByRowLogger->set_level(spdlog::level::from_str(Config::getInstance().logLevel));
            }
            return this->latticeByRowLogger;
        }

        std::string logFileName;
        std::string sequentialLoggerName;
        std::string parallelLoggerName;
        std::string greedyLatticeLoggerName;
        std::string latticeByRowLoggerName;
        std::string logLevel;
        double rowSpacing, columnSpacing, minAodSpacing;
        bool allowMovingEmptyTrapOntoOccupied, allowDiagonalMovement, allowMovesBetweenRows, allowMovesBetweenCols, 
            allowMultipleMovesPerAtom, alwaysGenerateAllAODTones;
        std::optional<bool> verticalSortingChannel;
        unsigned int aodTotalLimit, aodRowLimit, aodColLimit;
        double moveCostOffset, moveCostOffsetSubmove, moveCostScalingSqrt, moveCostScalingLinear;
        double recommendedDistFromOccSites, recommendedDistFromEmptySites, minDistFromOccSites, maxSubmoveDistInPenalizedArea;
};