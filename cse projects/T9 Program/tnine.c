/* tnine is a program that drives a trie / t9 program.  This code
   will build a trie, according to trienode.  It will also run
   an interactive session where the user can retrieve words using
   t9 sequences.
   CSE374, HW5, 22wi
   Xiyi Wang
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "trienode.h"

// run_session run the on-going decoding interaction with the
// user.  It requires a previously build wordTrie to work.
void run_session(trieNode *wordTrie);

int main(int argc, char **argv) {
  FILE *dictionary = NULL;
  trieNode *wordTrie = NULL;

  if (argc < 2) {
    fprintf(stderr, "Usage: %s [DICTIONARY]\n", argv[0]);
    return EXIT_FAILURE;
  } else {
    dictionary = fopen(argv[1], "r");
    if (!dictionary) {
      fprintf(stderr, "Error: Cannot open %s\n", argv[1]);
      return EXIT_FAILURE;
    }
  }

  // build the trie
  wordTrie = build_tree(dictionary);

  // run interactive session
  run_session(wordTrie);

  // clean up
  free_tree(wordTrie);
  fclose(dictionary);

  return(EXIT_SUCCESS);
}

// run_session run the on-going decoding interaction with the
// user. It requires a previously build wordTrie to work.
void run_session(trieNode *wordTrie) {
  char pattern[MAXLEN];
  char prevPattern[MAXLEN];
  printf("Enter \"exit\" to quit. \n");
  while (1) {
    printf("Enter Key Sequence (or \"#\" for next word):\n");
    scanf("%s", pattern);
    if (strncmp(pattern, "exit", 4) == 0) {
      break;
    } else if (strncmp(pattern, "#", 1) == 0) {
      prevPattern[strlen(prevPattern) + 1] = '\0';
      prevPattern[strlen(prevPattern)] = '#';
    } else {
      strncpy(prevPattern, pattern, strlen(pattern) + 1);
    }
    printf("%s\n", get_word(wordTrie, prevPattern));
  }
}



