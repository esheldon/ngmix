CC=g++

prefix := /usr/local

CFLAGS = -I. -Wall -O2
LINKFLAGS = -lm -static

TEST_GMIX = test/test-gmix
TEST_FMATH = test/test-fmath
TEST_FMATH_SPEED = test/test-fmath-speed
TEST_IMAGE = test/test-image

tests: $(TEST_GMIX) $(TEST_FMATH) $(TEST_FMATH_SPEED) $(TEST_IMAGE)

$(TEST_GMIX): gmix.h test/test-gmix.cc
	$(CC) -o test/test-gmix test/test-gmix.cc $(CFLAGS) $(LINKFLAGS)

$(TEST_FMATH): fmath.h test/test-fmath.cc
	$(CC) -o test/test-fmath test/test-fmath.cc $(CFLAGS) $(LINKFLAGS)

$(TEST_FMATH_SPEED): fmath.h test/test-fmath-speed.cc
	$(CC) -o test/test-fmath-speed test/test-fmath-speed.cc $(CFLAGS) $(LINKFLAGS)

$(TEST_IMAGE): image.h MtRng.h test/test-image.cc
	$(CC) -o test/test-image test/test-image.cc $(CFLAGS) $(LINKFLAGS)
