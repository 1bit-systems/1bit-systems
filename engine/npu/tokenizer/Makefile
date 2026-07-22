CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
TARGET = tokenize
SRC = tokenize.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean
