#include <stdio.h>
#include <stdint.h>

long g_long_array[5] = { 1, 1594, 111, 45678, 359 };
int g_ip_address = 0x0afe0a05;
int g_ip_mask = 0xffff0000;
char g_encoded[] = "onafxn";

int g_counter = 0; //promneny pro output
long g_output = 0;
int g_net_address = 0;

char g_ip_str[20] = {0};  
char g_net_str[20] = {0};  

void count_odd_numbers();
void process_even_numbers();
void decode_rot13();

int main() {
    count_odd_numbers();
    process_even_numbers();
    decode_rot13();

    printf("Ukol 1: Pocet lichych cisel (negovanych): %d\n", g_counter);
    printf("Ukol 2: Soucet sudych cisel deleny 7: %ld\n", g_output);
    printf("Ukol 3a: Sitova adresa: 0x%08x\n", g_net_address);
    printf("Original IP: %s\n", g_ip_str);
    printf("Sitova adresa: %s\n", g_net_str);
    printf("Ukol 3b: Dekodovany retezec: %s\n", g_encoded);
}
