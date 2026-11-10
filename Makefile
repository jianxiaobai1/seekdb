CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -DNDEBUG -Wall -Wextra

all: minivectordb

minivectordb: src/main.cpp src/engine.hpp src/sql.hpp src/nlohmann/json.hpp
	$(CXX) $(CXXFLAGS) -o $@ src/main.cpp

clean:
	rm -f minivectordb

.PHONY: all clean
