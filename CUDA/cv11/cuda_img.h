#pragma once

struct CudaImg
{
    uint3 m_size;
    union {
        void *m_p_void;
        uchar1 *m_p_uchar1;
        uchar3 *m_p_uchar3;
        uchar4 *m_p_uchar4;
    };
};
