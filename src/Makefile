CC=g++

prefix := /usr/local

CFLAGS = -I. -Wall -O2
LINKFLAGS = -lm -static

TEST_GMIX = test/test-gmix

tests: $(TEST_GMIX)

$(TEST_GMIX): gmix.h test/test-gmix.cc
	$(CC) -o test/test-gmix test/test-gmix.cc $(CFLAGS) $(LINKFLAGS)
