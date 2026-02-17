
section .text 
global jen_prvocisla
global bubble

            ;1
jen_prvocisla:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14

    mov r12, rdi
    mov r13, rsi
    xor r14 , r14

array_loop:
    cmp r14, r13
    jge end_array_loop

    mov rbx , [r12 + r14*8]

    mov rdi , rbx
    call is_prime

    test rax, rax
    jnz skip_zero

    mov qword [r12 + r14*8],0

skip_zero:
    inc r14
    jmp array_loop

end_array_loop:
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp

    ret

is_prime:
    push rbp
    mov rbp, rsp
    push rbx

    cmp rdi ,1 
    jle not_prime

    cmp rdi , 2
    je is_prime_true

    test rdi , 1
    jz not_prime

    mov rbx , 3 

prime_loop:
    mov rax , rbx
    imul rax , rbx
    cmp rax, rdi
    jg is_prime_true

    mov rax, rdi
    xor rdx,rdx
    idiv rbx

    test rdx ,rdx
    jz not_prime

    add rbx ,2 
    jmp prime_loop

is_prime_true:
    mov rax , 1
    jmp end_is_prime

not_prime:
    xor rax,rax

end_is_prime:
    pop rbx
    pop rbp
    ret
   

            ;2
