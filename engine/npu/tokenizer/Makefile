CC = gcc
CFLAGS = -O3 -Wall -Wextra
TARGET = tokenize

all: $(TARGET)

$(TARGET): tokenize.c
	$(CC) $(CFLAGS) -o $(TARGET) tokenize.c -lm

clean:
	rm -f $(TARGET)

.PHONY: all clean
