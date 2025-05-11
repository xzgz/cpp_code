import custom_converter

def test_str2float():
    float_info_list = [
        ["0x3F800000", 8, 23],
        ["0x80000000", 8, 23],
        ["0x5de8", 5, 10],
        ["0x43bd", 8, 7],
    ]
    for float_info in float_info_list:
        hex_str, exponent_width, significand_with = float_info
        bin_str = custom_converter.hex_to_binary(hex_str)
        fpx = custom_converter.custom_to_double(bin_str, exponent_width, significand_with)
        diff_with_nearest = custom_converter.calculate_diff_with_nearest(bin_str, exponent_width, significand_with)
        print(f"hex_input: {hex_str} binary_input: {bin_str} fp_value: {fpx:.17e} diff_with_nearest: {diff_with_nearest:.17e}")

test_str2float()
