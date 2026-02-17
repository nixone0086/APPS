#include <stdio.h>
#include <stdint.h>

void merge_and_extend();
void decode();

char g_char2[ 2 ] = { 0xCA, 0xFE };
short g_short_merged;
int g_extended;

char g_decoded[5]= {0};
int enc_string =0x53415050; //nebo hned zmenit na 0x53505041 :)

int main() {

merge_and_extend();
printf( "Variables g_short_merged=%d, g_extended=%#010x\n", g_short_merged, g_extended );

decode();
printf( "String = %s\n", g_decoded);
}
