// Xiyi Wang

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_LINE_LENGTH 500

void checkArguments(int num_args);
int count_lines(FILE *file);
int count_words(FILE *file);
int count_chars(FILE *file);

int main(int argc, char *argv[]) {
    checkArguments(argc);
    int lines = 0, words = 0, chars = 0;
    int opt_lines = 0, opt_words = 0, opt_chars = 0;
    int first_file_index = 1;

    if (argv[1][0] == '-') {
        if (strncmp(argv[1], "-l", 2) == 0) {
            opt_lines = 1;
        } else if (strncmp(argv[1], "-w", 2) == 0) {
            opt_words = 1;
        } else if (strncmp(argv[1], "-c", 2) == 0) {
            opt_chars = 1;
        } else {
            first_file_index = 1;
        }
        first_file_index = 2;
    }

    // Check if the file does exist
    if (first_file_index >= argc) {
        fprintf(stderr, "Error: no input file. \n");
        exit(EXIT_FAILURE);
    }

    for (int i = first_file_index; i < argc; i++) {
        FILE *file = fopen(argv[i], "r");
        if (!file) {
            fprintf(stderr, "Error: could not open file %s\n", argv[i]);
            continue;
        }

        int file_lines = count_lines(file);
        int file_words = count_words(file);
        int file_chars = count_chars(file);
        fclose(file);

        lines += file_lines;
        words += file_words;
        chars += file_chars;

        if (!opt_lines && !opt_words && !opt_chars) {
            printf("%d %d %d %s\n", file_lines, file_words, file_chars, argv[i]);
        } else if (opt_lines) {
            printf("%d %s\n", file_lines, argv[i]);
                } else if (opt_words) {
            printf("%d %s\n", file_words, argv[i]);
        } else if (opt_chars) {
            printf("%d %s\n", file_chars, argv[i]);
        }
    }

    if (!opt_lines && !opt_words && !opt_chars) {
        printf("%d %d %d total\n", lines, words, chars);
    }
    exit(EXIT_SUCCESS);
}

/*
 * Check the number of arguments that were passed
 * int num_args, the number of arguments supplied
 * Exit with error if no input 
 */
void checkArguments(int num_args) {
  if (num_args < 2) {
    fprintf(stderr, "Too few arguments.\n");
    exit(EXIT_FAILURE);
  }
}

/*
 * Count the number of lines in a given file
 * IN: a pointer to the filename
 * OUT: a pointer to an int array containing the counts
 */
int count_lines(FILE *file) {
    int lines = 0;
    char line[MAX_LINE_LENGTH];
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        lines++;
    }
    rewind(file);
    return lines;
}

/*
 * Count the number of words in a given file
 * IN: a pointer to the filename
 * OUT: a pointer to an int array containing the counts
 */
int count_words(FILE *file) {
    int words = 0;
    int in_word = 0;
    char c;
    while ((c = fgetc(file)) != EOF) {
        if (c == ' ' || c == '\n' || c == '\t') {
            if (in_word) {
                words++;
                in_word = 0;
            }
        } else {
            in_word = 1;
        }
    }
    rewind(file);
    return words;
}

/*
 * Count the number of characters in a given file
 * IN: a pointer to the filename
 * OUT: a pointer to an int array containing the counts
 */
int count_chars(FILE *file) {
    int chars = 0;
    char c;
    while ((c = fgetc(file)) != EOF) {
        chars++;
    }
    rewind(file);
    return chars;
}
