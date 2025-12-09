all: pi.out

pi.out: pi.cpp
	g++ pi.cpp -o pi.out -lpthread -O3

.PHONY: all clean
clean:
	rm -f pi.out
