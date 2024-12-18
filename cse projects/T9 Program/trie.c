/* trie implements a trie, made of trieNodes. This includes
   code to build, search, and delete a trie
   Xiyi Wang
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "trienode.h"

/* helper functions */
// Takes in a lowercase letter c and returns its corresponding
// T9 number (minus 1). Assumes input is a valid ASCII character.
// Returns 0 for non-letter inputs.
char charToNum(char c);
// Takes in a pointer to a character array representing a word
// and returns an array of integers containing the T9 number
// patterns for the characters in the input word.
int* createNums(char *c);
// Creates a new trieNode with 9 branches
// and returns the empty node
trieNode* make_trienode();
// This function basically inserts a word into the T9 trie. It creates a branch for each
// T9 value of the characters in the word, and stores the word at the end of the branch.
// The function returns a pointer to the root node of the trie.
trieNode* insert_trie(trieNode* root, char* word);

char charToNum(char c) {
  if ((c == 97) || (c == 98) || (c == 99)) { // a,b,c
    return 1;
  } else if ((c == 100) || (c == 101) || (c == 102)) { //d,e,f
    return 2;
  } else if ((c == 103) || (c == 104) || (c == 105)) {
    return 3;
  } else if ((c == 106) || (c == 107) || (c == 108)) {
    return 4;
  } else if ((c == 109) || (c == 110) || (c == 111)) {
    return 5;
  } else if ((c == 112) || (c == 113) || (c == 114)
    || (c == 115)) {
    return 6;
  } else if ((c == 116) || (c == 117) || (c == 118)) {
    return 7;
  } else if ((c == 119) || (c == 120) || (c == 121)
    || (c == 122)) {
    return 8;
  } else {
    return 0;
  }
}

int* createNums(char *word) {
    int word_len = strlen(word);
    int* t9_numbers = malloc(word_len * sizeof(int));
    if (t9_numbers == NULL) {
      fprintf(stderr, "Error: Unable to allocate memory\n");
      exit(EXIT_FAILURE);
    }
    for (int i = 0; i < word_len; i++) {
        t9_numbers[i] = charToNum(word[i]);
    }
    return t9_numbers;
}

trieNode* make_trienode() {
  trieNode* node_ptr = (trieNode*)malloc(sizeof(trieNode));
  for (int i = 0; i < BRANCHES; i++) {
    node_ptr->branches[i] = NULL;
  }
  // Set the word field to NULL
  node_ptr->word = NULL;
  // Return a pointer to the newly created node
  return node_ptr;
}

trieNode* insert_trie(trieNode* root, char* word) {
  trieNode* temp = root;
  int *wordNums;
  wordNums = createNums(word);
  for (int i=0; word[i] != '\0'; i++) {
    int index = wordNums[i];
    if (temp->branches[index] == NULL) {
      temp->branches[index] = make_trienode();
    }
    temp = temp->branches[index];
    }
  while (temp->word != NULL) {
    if (temp->branches[0] == NULL) {
      temp->branches[0] = make_trienode();
    }
    temp = temp->branches[0];
  }
  temp->word  = (char*)malloc((strlen(word)+1));
  strncpy(temp->word, word, strlen(word)+1);
  free(wordNums);
  return root;
}

char* get_word(trieNode* root, char* pattern) {
  // Initialize the current node to the root of the trie
  trieNode* current = root;

  // Loop through each character in the pattern
  for (int i = 0; pattern[i] != '\0'; i++) {
    // If the current character is a hash symbol, follow the 0 branch
    if (pattern[i] == '#') {
      if (current->branches[0] != NULL) {
        current = current->branches[0];
      } else {
        return "There are no more T9onyms";
      }

      // If this is the last character in the pattern, return the word if it exists
      if (i == strlen(pattern) - 1) {
        if (current->word != NULL) {
          return current->word;
        } else {
          return "Not found in current dictionary";
        }
      }
    }
    // If the current character is not a digit or is out of range, return an error message
    else if (!(pattern[i] >= '2' && pattern[i] <= '9')) {
      return "Not found in current dictionary";
    } else {
      // Calculate the index of the branch based on the current digit
      // int index = pattern[i] - '2';
      int index = (pattern[i] - '0');
      index = index - 1;

      // Follow the branch if it exists
      if (current->branches[index] != NULL) {
        current = current->branches[index];

        // If this is the last character in the pattern, return the word if it exists
        if (i == strlen(pattern) - 1) {
          if (current->word != NULL) {
            return current->word;
          } else {
            return "Not found in current dictionary";
          }
        }
      }
    }
  }

  // If the loop completes without finding a word, return an error message
  return "Not found in current dictionary";
}

trieNode* build_tree(FILE *dict_file) {
  char word[MAXLEN];
  trieNode* root = make_trienode();
  while (fgets(word, MAXLEN, dict_file) != NULL) {
    if ((word[strlen(word) - 1]) == '\n') {
      word[strlen(word) - 1] = '\0';
    }
    root = insert_trie(root, word);
  }
  return root;
}

void free_tree(trieNode* root) {
  if (root == NULL) {
    return;
  }
  for (int i = 0; i < BRANCHES; i++) {
    free_tree(root->branches[i]);
    root->branches[i] = NULL;
  }
  if (root->word != NULL) {
    free(root->word);
    root->word = NULL;
  }
  free(root);
}