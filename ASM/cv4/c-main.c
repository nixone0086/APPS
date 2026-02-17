#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void jen_prvocisla( long *pole, int N );
void bubble( int *pole, int N, int asc_desc );
long numbers[] = {2, 3, 4, 5, 6, 7, 8 ,9 ,10 ,11, 12, 13, 97, 100};
int size = sizeof(numbers) / sizeof(numbers[0]);

int main() {
//task 1
    printf("Before: ");
    for(int i =0 ; i <size; i++){
        printf ("%ld ",numbers[i]);
    }
    printf ("\n");
    printf ("Jen prvocisla\n");
   

    jen_prvocisla(numbers , size);
    printf("After: ");

    for(int i =0 ; i <size; i++){
        printf ("%ld ",numbers[i]);
    }
    printf ("\n");
//------------------------------------------
//task 2 :(


   return 0;
}
