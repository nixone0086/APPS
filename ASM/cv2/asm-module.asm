bits 64
section .text
global count_odd_numbers
global process_even_numbers
global decode_rot13

extern g_long_array
extern g_counter
extern g_output
extern g_ip_address
extern g_ip_mask
extern g_net_address
extern g_encoded
extern g_ip_str
extern g_net_str

count_odd_numbers:;1 funkce    
    push rbx
    push rcx
    mov rbx, g_long_array  
    xor rcx, rcx          
    mov r8, 5             
.loop:
    mov rax, [rbx]    ;aktualni element
    test rax, 1          
    jz .even            
    inc rcx              
.even:
    add rbx, 8           
    dec r8                
    jnz .loop             
    mov [g_counter], ecx  
    not dword [g_counter] 
    pop rcx
    pop rbx
    ret
process_even_numbers: ;2 funkce
    push rbx
    push rcx
    
    mov rbx, g_long_array  ;
    xor rcx, rcx          
    mov r8, 5             ;array velikost
    
.loop:
    mov rax, [rbx]        ;aktualni element
    test rax, 1        
    jnz .odd             
    add rcx, rax          
.odd:
    add rbx, 8           
    dec r8              
    jnz .loop             
    mov rax, rcx          
    mov rcx, 7           
    cqo                   ;rozsireni  RAX v RDX:RAX
    idiv rcx              
    mov [g_output], rax   
    pop rcx
    pop rbx
    ret
decode_rot13: ;3 funkce + decode ROT13
    push rbx
    push rcx
    push rdx
    push rsi
    push rdi
    mov eax, [g_ip_address]
    and eax, [g_ip_mask]
    mov [g_net_address], eax
    mov eax, [g_ip_address] ;original IP address
    mov rdi, g_ip_str
    call format_ip
    ; Format network address
    mov eax, [g_net_address]
    mov rdi, g_net_str
    call format_ip
    mov rbx, g_encoded ;decode ROT13
.decode_loop:
    movzx ecx, byte [rbx]  ;aktualni znak
    test ecx, ecx          ;kontrola pokud je to null terminator
    jz .done               
    cmp ecx, 'a' ;kontrola male pismeny
    jl .next_char
    cmp ecx, 'z'
    jg .next_char
    cmp ecx, 'm' ;ROT13 
    jle .add_13
    sub ecx, 13 ;sub 13 for n-z
    jmp .store_char
.add_13:
    add ecx, 13  ; add 13 pro a-m
.store_char:
    mov byte [rbx], cl    
.next_char:
    inc rbx                ; nasledujii symbol
    jmp .decode_loop
.done:
    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    ret
format_ip: ;pomocna funkce pro format ip adresy
    push rax
    push rbx
    push rcx
    push rdx
    push rsi
    
    mov edx, eax ;4 byte left
    shr edx, 24
    and edx, 0xFF
    call append_number
    mov byte [rdi], '.'
    inc rdi
    mov edx, eax  ; 3rd byte
    shr edx, 16
    and edx, 0xFF
    call append_number
    mov byte [rdi], '.'
    inc rdi
    mov edx, eax ; 2nd byte
    shr edx, 8
    and edx, 0xFF
    call append_number
    mov byte [rdi], '.'
    inc rdi
    mov edx, eax ;prvni byte
    and edx, 0xFF
    call append_number
    mov byte [rdi], 0  ; null-terminate 
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret
append_number: ; Připojení desetinného čísla k řetězci
    push rax
    push rbx
    push rcx
    push rdx
    push rsi
    mov eax, edx
    xor esi, esi    ; pocet digitu
    test eax, eax
    jnz .not_zero
    mov byte [rdi], '0'
    inc rdi
    jmp .done
.not_zero:
    mov ebx, 10
    mov ecx, eax    
.count_digits:
    xor edx, edx
    div ebx         ; eax = eax / 10
    inc esi         
    test eax, eax
    jnz .count_digits
    mov eax, ecx    
    sub rsp, 16     
    mov rbx, rsp    
    mov ecx, 10
.convert:
    xor edx, edx
    div ecx       ; eax = eax / 10
    add dl, '0'   ; convertace to ASCII
    mov [rbx], dl   
    inc rbx         
    test eax, eax
    jnz .convert
    dec rbx         ; posledni digit
.copy_loop:
    mov al, [rbx]
    mov [rdi], al
    inc rdi
    dec rbx
    dec esi
    jnz .copy_loop
    add rsp, 16 ; сistka staku
.done:
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret
