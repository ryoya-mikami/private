import struct
import math

def half_16bit_add(a, b):
    """
    16ビット(half)として加算を行う。
    16ビットを超える（0xFFFFを超える）場合は下位16ビットのみを保持する。
    """
    return (a + b) & 0xFFFF

def half_16bit_sub(a, b):
    """
    16ビット(half)として減算を行う。
    結果が負になる場合は2の補数表現（下位16ビット保持）としてラップアラウンドする。
    """
    return (a - b) & 0xFFFF

def half_16bit_mul(a, b):
    """
    16ビット(half)として乗算を行う。
    16ビットを超える場合は下位16ビットのみを保持する。
    """
    return (a * b) & 0xFFFF

def float_to_hex_ieee754(f):
    """
    IEEE 754 単精度（32ビット）浮動小数点数を16進数文字列に変換する。
    """
    # 'f'は単精度、'I'は無符号整数（32ビット）
    packed = struct.pack('>f', f)
    unpacked = struct.unpack('>I', packed)[0]
    return f"0x{unpacked:08X}"

def hex_to_float_ieee754(h_str):
    """
    16進数文字列（または数値）を IEEE 754 単精度浮動小数点数に変換する。
    """
    if isinstance(h_str, str):
        h = int(h_str, 16)
    else:
        h = h_str
    packed = struct.pack('>I', h)
    return struct.unpack('>f', packed)[0]

def solve_quadratic_standard(a, b, c):
    """
    標準的な2次方程式の解の公式 (1): x = (-b ± sqrt(b^2 - 4ac)) / 2a
    """
    d = math.sqrt(b**2 - 4*a*c)
    x1 = (-b + d) / (2*a)
    x2 = (-b - d) / (2*a)
    return x1, x2

def solve_quadratic_refined(a, b, c):
    """
    桁落ちを回避する改良された2次方程式の解の公式 (2)。
    q = -0.5 * (b + sgn(b) * sqrt(b^2 - 4ac))
    x1 = q / a, x2 = c / q
    """
    sgn_b = 1 if b >= 0 else -1
    q = -0.5 * (b + sgn_b * math.sqrt(b**2 - 4*a*c))
    x1 = q / a
    x2 = c / q
    return x1, x2

def main():
    print("========================================")
    print(" 第11回 数値計算アルゴリズム 課題コード")
    print("========================================\n")

    print("--- 課題 (1) 16ビット整数の演算 ---")
    results_1 = [
        ("(a) 0x356F + 0x6212", half_16bit_add(0x356F, 0x6212)),
        ("(b) 0x5863 - 0x8457", half_16bit_sub(0x5863, 0x8457)),
        ("(c) 0x0025 * 0x0012", half_16bit_mul(0x0025, 0x0012))
    ]
    for label, res in results_1:
        print(f"{label} = 0x{res:04X} (dec: {res})")

    print("\n--- 課題 (2) 浮動小数点数変換 (IEEE 754) ---")
    val_a = 2.75
    hex_a = float_to_hex_ieee754(val_a)
    print(f"(a) {val_a} -> IEEE 754 Hex: {hex_a}")

    hex_b = "0xC1280000"
    val_b = hex_to_float_ieee754(hex_b)
    print(f"(b) {hex_b} -> Float Value: {val_b}")

    print("\n--- 課題 (3) 2次方程式と桁落ち ---")
    print("条件: a=1, b=10^10, c=1")
    a, b, c = 1, 10**10, 1
    
    x1_std, x2_std = solve_quadratic_standard(a, b, c)
    x1_ref, x2_ref = solve_quadratic_refined(a, b, c)

    print(f"公式 (1) [標準]: x1 = {x1_std}, x2 = {x2_std}")
    print(f"公式 (2) [改良]: x1 = {x1_ref}, x2 = {x2_ref}")
    
    print("\n[考察]")
    print("公式(1)では x1 を求める際の分子 -b + sqrt(b^2-4ac) で桁落ちが発生します。")
    print("bに対し4acが極端に小さいため、sqrt(b^2-4ac)がほぼbになり、")
    print("『ほぼ等しい数の引き算』が行われることで精度が失われ、x1 は 0.0 になります。")

if __name__ == "__main__":
    main()
