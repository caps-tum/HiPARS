CC=c++
INCLUDE=-I../include -I../../extern/eigen -I../../extern/spdlog/include
CXXFLAGS=-O3 -Wall -std=c++20 -fPIC -fopenmp -D COMPILED_AS_EXECUTABLE $(INCLUDE)

SRC_DIR	:= ../src
OBJ_DIR	:= obj
BIN_DIR	:= lib

SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
SOURCES := $(filter-out $(SRC_DIR)/bind.cpp,$(SOURCES))
SOURCES := $(filter-out $(SRC_DIR)/fixLatticeByRow.cpp,$(SOURCES))
SOURCES := $(filter-out $(SRC_DIR)/sortLatticeByRow.cpp,$(SOURCES))
SOURCES := $(filter-out $(SRC_DIR)/sortLatticeGreedy.cpp,$(SOURCES))
SOURCES := $(filter-out $(SRC_DIR)/latticeHelpers.cpp,$(SOURCES))
SOURCES := $(filter-out $(SRC_DIR)/sortSequentiallyByRow.cpp,$(SOURCES))

OBJECTS	:= $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

all: testBinary
fresh:
	-rm $(OBJECTS)
	-rm $(OBJ_DIR)/main.o
	-rm $(BIN_DIR)/resorting_cpp
	make testBinary

testBinary: 
	make $(BIN_DIR)/resorting_cpp
$(BIN_DIR)/resorting_cpp: $(OBJECTS) $(OBJ_DIR)/main.o | $(BIN_DIR)
	$(CC) -o $@ $^ $(CXXFLAGS)

$(OBJ_DIR)/main.o: main.cpp | $(OBJ_DIR)
	$(CC) -c $< -o $@ $(CXXFLAGS)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) -c $< -o $@ $(CXXFLAGS)
$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@