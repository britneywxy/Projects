all: t9

t9: trie.o tnine.o
	gcc -Wall -std=c11 -g -o t9 trie.o tnine.o

trie.o: trie.c trienode.h
	gcc -Wall -std=c11 -g -c trie.c

tnine.o: tnine.c trienode.h
	gcc -Wall -std=c11 -g -c tnine.c

clean:
	rm -f .o t9~
