#include <stdio.h>
#include <string.h>
void long2strbin(long t_cislo, char *t_str);
int suma_cislic(char *t_str);
int vetsi_nez_2N(long *t_pole, int t_N);


int main() {
    char binary_str[65];  
    long number = 7;
    //1
    long2strbin(number, binary_str);
    printf("Binary representation of %ld: %s\n", number, binary_str);
    //2
    char test_str[] = "a123b";
    int digit_sum = suma_cislic(test_str);
    printf("Sum of digits in '%s': %d\n", test_str, digit_sum);
     //3
   
}
