bits 64
section .text 
    global long2strbin
    global suma_cislic
    global vetsi_nez_2N
   
;1 ukol

long2strbin:
    mov rax ,rdi
    mov rcx, 64
    mov rdx, rsi

convert_loop:
    shl rax, 1
    mov byte [rdx],0
    adc byte [rdx],'0'
    inc rdx
    loop convert_loop

    mov byte [rdx], 0
    ret


;2 ukol

suma_cislic:
    push rbx
    xor rbx, rbx
    xor rax, rax
.loop:
    mov al,[rdi]
    test al,al
    jz .done

    cmp al,'0'
    jl .next
    cmp al,'9'
    jg .next

    sub al, '0'
    add ebx,eax
.next:
    inc rdi
    jmp .loop
.done 
    mov eax,ebx
    pop rbx
    ret
