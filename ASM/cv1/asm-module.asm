bits 64

section .data

   extern g_char2
   extern g_short_merged
   extern g_extended
   extern g_decoded
   extern enc_string

section .text

global merge_and_extend
    merge_and_extend:
    mov ah , byte[g_char2]
    mov al , byte[g_char2+1]
    mov [g_short_merged], ax
    mov al,[g_short_merged]
    mov [g_extended] ,ax
    ret

global decode  
    decode: 
    mov eax , [enc_string]
    mov dword[g_decoded], eax
    mov byte[g_decoded],'A'
    mov byte[g_decoded+2],'P'
    ret
