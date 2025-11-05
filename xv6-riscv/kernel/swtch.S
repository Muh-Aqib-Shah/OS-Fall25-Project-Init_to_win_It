# Context switch
#
#   void swtch(struct context *old, struct context *new);
# 
# Save current registers in old. Load from new.	


.globl swtch
swtch:
        # Save general purpose registers
        sd ra, 0(a0)
        sd sp, 8(a0)
        sd s0, 16(a0)
        sd s1, 24(a0)
        sd s2, 32(a0)
        sd s3, 40(a0)
        sd s4, 48(a0)
        sd s5, 56(a0)
        sd s6, 64(a0)
        sd s7, 72(a0)
        sd s8, 80(a0)
        sd s9, 88(a0)
        sd s10, 96(a0)
        sd s11, 104(a0)

        # Save fpu registers f0–f31
        fsd f0, 112(a0)
        fsd f1, 120(a0)
        fsd f2, 128(a0)
        fsd f3, 136(a0)
        fsd f4, 144(a0)
        fsd f5, 152(a0)
        fsd f6, 160(a0)
        fsd f7, 168(a0)
        fsd f8, 176(a0)
        fsd f9, 184(a0)
        fsd f10, 192(a0)
        fsd f11, 200(a0)
        fsd f12, 208(a0)
        fsd f13, 216(a0)
        fsd f14, 224(a0)
        fsd f15, 232(a0)
        fsd f16, 240(a0)
        fsd f17, 248(a0)
        fsd f18, 256(a0)
        fsd f19, 264(a0)
        fsd f20, 272(a0)
        fsd f21, 280(a0)
        fsd f22, 288(a0)
        fsd f23, 296(a0)
        fsd f24, 304(a0)
        fsd f25, 312(a0)
        fsd f26, 320(a0)
        fsd f27, 328(a0)
        fsd f28, 336(a0)
        fsd f29, 344(a0)
        fsd f30, 352(a0)
        fsd f31, 360(a0)

        # Save fcsr
        frcsr t0
        sd t0, 368(a0)

        # Switch to new context
        ld ra, 0(a1)
        ld sp, 8(a1)
        ld s0, 16(a1)
        ld s1, 24(a1)
        ld s2, 32(a1)
        ld s3, 40(a1)
        ld s4, 48(a1)
        ld s5, 56(a1)
        ld s6, 64(a1)
        ld s7, 72(a1)
        ld s8, 80(a1)
        ld s9, 88(a1)
        ld s10, 96(a1)
        ld s11, 104(a1)

        # Restore fpu registers
        fld f0, 112(a1)
        fld f1, 120(a1)
        fld f2, 128(a1)
        fld f3, 136(a1)
        fld f4, 144(a1)
        fld f5, 152(a1)
        fld f6, 160(a1)
        fld f7, 168(a1)
        fld f8, 176(a1)
        fld f9, 184(a1)
        fld f10, 192(a1)
        fld f11, 200(a1)
        fld f12, 208(a1)
        fld f13, 216(a1)
        fld f14, 224(a1)
        fld f15, 232(a1)
        fld f16, 240(a1)
        fld f17, 248(a1)
        fld f18, 256(a1)
        fld f19, 264(a1)
        fld f20, 272(a1)
        fld f21, 280(a1)
        fld f22, 288(a1)
        fld f23, 296(a1)
        fld f24, 304(a1)
        fld f25, 312(a1)
        fld f26, 320(a1)
        fld f27, 328(a1)
        fld f28, 336(a1)
        fld f29, 344(a1)
        fld f30, 352(a1)
        fld f31, 360(a1)

        # Restore fcsr
        ld t0, 368(a1)
        fscsr t0

        ret
	
