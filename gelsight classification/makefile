# CC = g++
# SRCS = src/tracking_class.cpp

# UNAME_S := $(shell uname -s)
# ifeq ($(UNAME_S),Linux)
#     CFLAGS = -g -Wall -std=c++11 -O3 -Wall -fPIC -shared -std=c++11 `python3 -m pybind11 --includes` -I ./
#     PROG = find_marker.so
# endif

# ifeq ($(UNAME_S),Darwin)
#     CFLAGS = -g -Wall -std=c++11 -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` -I ./
#     PROG = src/lib/find_marker`python3-config --extension-suffix`
# endif

# $(PROG):$(SRCS)
# 	$(CC) $(CFLAGS) -o $(PROG) $(SRCS)

# clean:
# 	rm -rf src/lib


# Makefile — build find_marker.so for your classification project

CC           := g++
SRCS         := src/tracking_class.cpp
PROG         := find_marker.so

# pick up the Conda env’s python
PYTHON       := $(shell which python)
# pybind11 include flags
PYBIND11_INC := $(shell $(PYTHON) -m pybind11 --includes)
# python linker flags (ensure python3-config exists in your env)
PY_LDFLAGS   := $(shell $(PYTHON)-config --ldflags)

# C++ flags: shared, position-independent, optimized
CXXFLAGS     := -g -Wall -std=c++11 -O3 -fPIC -shared

# Default target
all: $(PROG)

# build rule
$(PROG): $(SRCS)
	@echo "Building $(PROG) with:"
	@echo "  Python:    $(PYTHON)"
	@echo "  Includes:  $(PYBIND11_INC)"
	@echo "  LDFLAGS:   $(PY_LDFLAGS)"
	$(CC) $(CXXFLAGS) $(PYBIND11_INC) -I./ -o $(PROG) $(SRCS) $(PY_LDFLAGS)

# remove old build
clean:
	@echo "Cleaning old $(PROG)..."
	rm -f $(PROG)
