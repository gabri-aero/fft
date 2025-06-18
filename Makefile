# Define compiler and flags
CXX = g++
CXX_FLAGS = -O2 -ffast-math -march=native -Wall -fPIC
LD_FLAGS = -shared

# Define directories
INCLUDE_DIR = include
BUILD_DIR = build
TEST_DIR = tests

# Define header files and object files
HEADERS = $(wildcard $(INCLUDE_DIR)/*.hpp)
TEST_SOURCES = $(HEADERS:$(INCLUDE_DIR)/%.hpp=$(TEST_DIR)/Test%.cpp)
TEST_EXES = $(TEST_SOURCES:$(TEST_DIR)/%.cpp=$(BUILD_DIR)/%.exe)

# Define GTest libraries
GTEST_LIBS = -lgtest -lgtest_main -pthread
GTEST_DIR = /usr/include/gtest

# Build tests
test: $(TEST_EXES)
	echo $(HEADERS)
	echo $(TEST_SOURCES)
	echo $(TEST_EXES)
	@for test_exe in $(TEST_EXES); do \
		./$$test_exe || exit 1; \
	done

# Create build directory if it does not exist
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Test targets
$(TEST_EXES): $(TEST_SOURCES) $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE_DIR) -I$(GTEST_DIR) $< -o $@ $(GTEST_LIBS)

# Clean build files
clean:
	rm -rf $(BUILD_DIR)
