#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <limits>
#include <bitset>
#include <iostream>
#include <iomanip>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace std;

string binaryStringToHex(const string& s) {
    if (s.empty()) {
        return "";
    }

    // 补前导零，使长度是4的倍数
    int pad = (4 - (s.size() % 4)) % 4;
    string padded = string(pad, '0') + s;

    string hex;
    for (size_t i = 0; i < padded.size(); i += 4) {
        string group = padded.substr(i, 4);
        bitset<4> bits(group);
        int val = bits.to_ulong();
        // 转换为十六进制字符
        if (val < 10) {
            hex += '0' + val;
        } else {
            hex += 'A' + (val - 10);
        }
    }
    hex = "0x" + hex;

    return hex;
}

string hexToBinary(const string& hexStr) {
    string binStr;
    size_t start = 0;

    // 去除0x/0X前缀
    if (hexStr.size() >= 2 && hexStr[0] == '0' && (hexStr[1] == 'x' || hexStr[1] == 'X')) {
        start = 2;
    }

    // 二进制结果预分配内存
    binStr.reserve((hexStr.size() - start) * 4);

    for (size_t i = start; i < hexStr.size(); ++i) {
        char c = hexStr[i];
        int val = 0;

        // 字符有效性检查和转换
        if (c >= '0' && c <= '9') {
            val = c - '0';
        } else if (c >= 'a' && c <= 'f') {
            val = 10 + (c - 'a');
        } else if (c >= 'A' && c <= 'F') {
            val = 10 + (c - 'A');
        } else {
            throw invalid_argument("无效的十六进制字符: " + string(1, c));
        }

        // 直接拼接bitset转换的二进制字符串
        binStr += bitset<4>(val).to_string();
    }

    return binStr;
}

uint64_t binary_str_to_uint64(const string& s, int start, int len) {
    uint64_t val = 0;
    for (int i = 0; i < len; ++i) {
        val = (val << 1) | (s[start + i] - '0');
    }
    return val;
}

double custom_to_double(const string& s, int w, int t) {
    double fp_val = 0.0;
    const uint64_t sign_bit = binary_str_to_uint64(s, 0, 1);
    const uint64_t exp_raw = binary_str_to_uint64(s, 1, w);
    const uint64_t mantissa = binary_str_to_uint64(s, 1 + w, t);
    const int bias_custom = (1 << (w - 1)) - 1;
    const uint64_t max_exp = (1ULL << w) - 1;

    // 构造 double 的位模式（原值）
    uint64_t double_bits = sign_bit << 63;
    if (exp_raw == max_exp) {    // NaN 或无穷大
        double_bits |= 0x7FF0000000000000;
        if (mantissa == 0) {
            // 无穷大
            fp_val = *reinterpret_cast<double*>(&double_bits);
        } else {
            // NaN: 保留尾数高位
            uint64_t payload = (mantissa << (52 - t)) & 0x000FFFFFFFFFFFFF;
            double_bits |= payload | 0x7FF8000000000000;
            fp_val = *reinterpret_cast<double*>(&double_bits);
        }
    } else {
        // 处理正常数和次正规数
        int exp_val = (exp_raw != 0) ? (exp_raw - bias_custom) : (1 - bias_custom);
        int exp_double = exp_val + 1023;    // double 的指数偏置为 1023

        if (exp_double >= 0x7FF) {    // 溢出到无穷大
            double_bits |= 0x7FF0000000000000;
            fp_val = *reinterpret_cast<double*>(&double_bits);
        } else {    // 正常数
            double_bits |= static_cast<uint64_t>(exp_double) << 52;
            double_bits |= (mantissa << (52 - t)) & 0x000FFFFFFFFFFFFF;
            fp_val = *reinterpret_cast<double*>(&double_bits);
        }
        if (exp_raw == 0) {
            uint64_t double_base_bits = (sign_bit << 63) | (static_cast<uint64_t>(exp_double) << 52);
            double fp_base_val = *reinterpret_cast<double*>(&double_base_bits);
            fp_val -= fp_base_val;
        }
    }
    cout << std::fixed << std::setprecision(17) << scientific << "binary_input: " << s << "\nfp_value: " << fp_val
         << endl;

    return fp_val;
}

double calculate_diff_with_nearest(const string& s, int w, int t) {
    double diff = 0.0;
    const uint64_t sign_bit = binary_str_to_uint64(s, 0, 1);
    const uint64_t exp_raw = binary_str_to_uint64(s, 1, w);
    const uint64_t mantissa = binary_str_to_uint64(s, 1 + w, t);
    const int bias_custom = (1 << (w - 1)) - 1;
    const uint64_t max_exp = (1ULL << w) - 1;

    // 处理特殊值
    if (exp_raw == max_exp) {
        if (mantissa == 0) {    // 无穷大
            diff = numeric_limits<double>::infinity();
        } else {    // NaN
            diff = numeric_limits<double>::quiet_NaN();
        }
    } else {
        // 构造 diff 的位模式
        uint64_t diff_bits = sign_bit << 63;
        uint64_t diff_base_bits = 0;
        int exp_val = (exp_raw != 0) ? (exp_raw - bias_custom) : (1 - bias_custom);
        int exp_diff = exp_val + 1023;    // 保持指数不变
        diff_bits |= static_cast<uint64_t>(exp_diff) << 52;
        diff_base_bits = diff_bits;
        diff_bits |= 1ULL << (52 - t);
        double diff_base = *reinterpret_cast<double*>(&diff_base_bits);
        double diff_val = *reinterpret_cast<double*>(&diff_bits);
        // double diff = diff_base - diff_val;
        diff = std::abs(diff_val - diff_base);
        // cout << std::scientific << std::setprecision(17) << diff_base << endl;
        // cout << std::scientific << std::setprecision(17) << diff_val << endl;
    }
    cout << std::fixed << std::setprecision(17) << scientific << "diff_with_nearest: " << diff << endl;

    return diff;
}

// Pybind11 包装代码
PYBIND11_MODULE(custom_converter, m) {
    m.doc() = "Python binding for custom_to_double function";
    m.def("custom_to_double",
          &custom_to_double,
          py::arg("s"),
          py::arg("w"),
          py::arg("t"),
          "Convert string to double with custom parameters");
    m.def("calculate_diff_with_nearest",
          &calculate_diff_with_nearest,
          py::arg("s"),
          py::arg("w"),
          py::arg("t"),
          "Convert string to double with custom parameters");

    m.def("hex_to_binary",
          &hexToBinary,    // 新增绑定
          py::arg("hex_str"),
          "Convert hex string to binary string");
    m.def("binary_to_hex",
          &binaryStringToHex,    // 可选绑定
          py::arg("binary_str"),
          "Convert binary string to hex string");
}
