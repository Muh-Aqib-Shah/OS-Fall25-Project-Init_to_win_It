# gen_maths_tables.py  (run locally: python3 gen_maths_tables.py > maths_testdata.h)
import math
import random
random.seed(0)

def f(x): return float(x)

def emit_array(name, arr, per_line=6):
    print(f"static const float {name}[] = {{")
    for i in range(0, len(arr), per_line):
        chunk = ", ".join(f"{v:.8f}f" for v in arr[i:i+per_line])
        print("  " + chunk + ",")
    print("};")
    print()

def main():
    SQRT_N=60; EXP_N=60; POW_N=60; SIN_N=60; COS_N=60; FABS_N=60
    print("#ifndef XV6M_MATHS_TESTDATA_H")
    print("#define XV6M_MATHS_TESTDATA_H")
    # sqrt: 0, small, perfect squares, large, denorm-ish
    sqrt_in=[]; sqrt_out=[]
    seeds = [0.0,1e-8,1e-4,0.25,1.0,2.0,4.0,9.0,1e3,1e6]
    while len(sqrt_in)<SQRT_N:
        x = random.choice(seeds) if random.random()<0.5 else 10**random.uniform(-6,6)
        sqrt_in.append(f(x))
        sqrt_out.append(f(math.sqrt(x)))
    print(f"#define SQRT_N {SQRT_N}")
    emit_array("sqrt_in", sqrt_in)
    emit_array("sqrt_out", sqrt_out)

    # exp: negatives, positives, large/small
    exp_in=[]; exp_out=[]
    while len(exp_in)<EXP_N:
        x = random.uniform(-12,12)
        exp_in.append(f(x)); exp_out.append(f(math.exp(x)))
    print(f"#define EXP_N {EXP_N}")
    emit_array("exp_in", exp_in)
    emit_array("exp_out", exp_out)

    # pow: include ints, fracs, negatives, 0^0, x^0, 1^x
    pow_x=[]; pow_y=[]; pow_out=[]
    cases=[(0.0,0.0),(0.0,2.0),(2.0,0.0),(1.0,5.0),(-2.0,3.0),(-2.0,4.0)]
    for c in cases: 
        pow_x.append(f(c[0])); pow_y.append(f(c[1]))
        pow_out.append(f((math.nan if c==(0.0,0.0) else (math.pow(c[0],c[1])))))
    while len(pow_x)<POW_N:
        x = random.choice([-2.0,-1.5,-1.0,-0.5,0.5,0.75,1.0,1.5,2.0,3.0,10.0])
        y = random.choice([-5.0,-3.0,-2.0,-1.0,-0.5,0.5,1.0,2.0,3.0,5.0])
        # Avoid domain where Python yields complex (neg base, non-integer exp)
        if x<0 and abs(y-round(y))>1e-6: 
            continue
        pow_x.append(f(x)); pow_y.append(f(y)); pow_out.append(f(math.pow(x,y)))
    print(f"#define POW_N {POW_N}")
    emit_array("pow_x", pow_x)
    emit_array("pow_y", pow_y)
    emit_array("pow_out", pow_out)

    # sin, cos: include multiples of pi and large values
    def many_angles(n):
        arr=[]
        base=[0, math.pi/6, math.pi/4, math.pi/3, math.pi/2, math.pi, 2*math.pi]
        for b in base: arr.append(f(b)); arr.append(f(-b))
        while len(arr)<n:
            k = random.uniform(-100.0,100.0)
            arr.append(f(k*math.pi))
        return arr[:n]

    sin_in = many_angles(SIN_N)
    sin_out = [f(math.sin(v)) for v in sin_in]
    print(f"#define SIN_N {SIN_N}")
    emit_array("sin_in", sin_in)
    emit_array("sin_out", sin_out)

    cos_in = many_angles(COS_N)
    cos_out = [f(math.cos(v)) for v in cos_in]
    print(f"#define COS_N {COS_N}")
    emit_array("cos_in", cos_in)
    emit_array("cos_out", cos_out)

    fabs_in=[f(random.uniform(-1e6,1e6)) for _ in range(FABS_N)]
    fabs_out=[f(abs(v)) for v in fabs_in]
    print(f"#define FABS_N {FABS_N}")
    emit_array("fabs_in", fabs_in)
    emit_array("fabs_out", fabs_out)

    print("#endif // XV6M_MATHS_TESTDATA_H")

if __name__ == "__main__":
    main()

