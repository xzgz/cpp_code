from flask import Flask, render_template_string, request, jsonify
import custom_converter

app = Flask(__name__)

state = {
    "E": 8,
    "M": 23,
    "bits": [0] * 32,
    "binary_str": "",
    "hex_value": "0x3F800000",
    "fp_value": "",
    "diff_with_nearest": "",
    "exp_bias": 0,
    "exp_max": 0,
    "exp_val": 0,
}

def binary_str_to_uint64(s: str, start: int, length: int) -> int:
    if not all(c in '01' for c in s[start:start+length]):
        raise ValueError("Input string must contain only '0' and '1'")
    return int(s[start:start+length], 2) & 0xFFFFFFFFFFFFFFFF
    # val = 0
    # for i in range(length):
    #     val = (val << 1) | (ord(s[start + i]) - ord('0'))
    # return val

exp_bias = 0
exp_max = 0
exp_val = 0
binary_str = ""
fp_value = 0
diff_with_nearest = 0

@app.route('/', methods=['GET', 'POST'])
def index():
    global exp_bias
    global exp_max
    global exp_val
    global binary_str
    global fp_value
    global diff_with_nearest
    if request.method == 'POST':
        # print(f"binary_str={binary_str}")
        if 'set_params' in request.form:
            state['E'] = int(request.form.get('E', 0))
            state['M'] = int(request.form.get('M', 0))
            total_bits = state['E'] + state['M'] + 1
            state['bits'] = [0] * total_bits

        elif 'convert_hex' in request.form:
            hex_str = request.form.get('hex_value', '').strip()
            state['hex_value'] = hex_str
            total_bits = state['E'] + state['M'] + 1

            try:
                num = int(hex_str, 16)
                # print(f"total_bits={total_bits}")
                # print(f"num={num}")
                # print(f"bin(num)={bin(num)}")
                # print(f"bin(num)[2:]={bin(num)[2:]}")
                binary_str = bin(num)[2:].zfill(total_bits)
                # print(f"binary_str={binary_str}")
                # print(f"len_binary_str1={len(binary_str)}")
                binary_str = binary_str[-total_bits:]  # Ensure correct length
                # print(f"len_binary_str2={len(binary_str)}")

                exp_raw_val = binary_str_to_uint64(binary_str, 1, state['E'])
                exp_val = exp_raw_val - exp_bias
                if exp_raw_val == 0:
                    exp_val = 1 - exp_bias
            except ValueError:
                binary_str = '0' * total_bits

            exponent_width = state['E']
            significand_with = state['M']
            fp_value = custom_converter.custom_to_double(binary_str, exponent_width, significand_with)
            diff_with_nearest = custom_converter.calculate_diff_with_nearest(binary_str, exponent_width, significand_with)
            state['bits'] = [int(bit) for bit in binary_str]

        elif 'convert' in request.form:
            binary_str = ''.join(map(str, state['bits']))
            exponent_width = state['E']
            significand_with = state['M']
            fp_value = custom_converter.custom_to_double(binary_str, exponent_width, significand_with)
            diff_with_nearest = custom_converter.calculate_diff_with_nearest(binary_str, exponent_width, significand_with)

            exp_raw_val = binary_str_to_uint64(binary_str, 1, state['E'])
            exp_val = exp_raw_val - exp_bias
            if exp_raw_val == 0:
                exp_val = 1 - exp_bias

        state['binary_str'] = binary_str
        state['fp_value'] = f"{fp_value:.17e}"
        state['diff_with_nearest'] = f"{diff_with_nearest:.17e}"

        exp_bias = (1 << (state['E'] - 1)) - 1
        exp_max = (1 << state['E']) - 1
        state['exp_bias'] = exp_bias
        state['exp_max'] = exp_max
        state['exp_val'] = exp_val
        # print(f"exp_bias={exp_bias}")
        # print(f"exp_max={exp_max}")
        # print(f"exp_val={exp_val}")

    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hex to Float Converter</title>
        <style>
            .bit-btn {
                width: 20px;
                height: 20px;
                margin: 2px;
                border: 1px solid #333;
                font-weight: bold;
                background-color: #f0f0f0;
            }
            button[name="set_params"] {
                width: 120px;          
                height: 40px;          
                min-width: 100px;      
            }
            button[name="convert_hex"] {
                width: 120px;         
                height: 40px;          
                min-width: 100px;      
            }
            button[name="convert"] {
                width: 120px;          
                height: 40px;          
                min-width: 100px;    
            }
            .container { margin: 20px; font-family: Arial, sans-serif; }
            .input-group {
                border: 2px solid #666;
                padding: 20px;
                margin: 15px 0;
                background: #f8f8f8;
            }
            .bit-group {
                display: inline-block;
                padding: 8px;
                margin: 4px;
            }
            .sign-bit { background: #ff4444; }    /* 红色符号位 */
            .exp-bits { background: #44ff44; }    /* 绿色指数位 */
            .frac-bits { background: #4444ff; }   /* 蓝色尾数位 */
            .output-box {
                border: 2px solid #666;
                padding: 10px;
                margin: 15px 0;
                line-height: 2.5;
                background: #f0f0f0;
                font-size: 16px;
            }
            input[type="number"] {
                width: 58px;
                padding: 10px;
                margin-right: 15px;
            }
            input[type="text"] {
                width: 120px;
                padding: 10px;
                margin-right: 15px;
            }
            input[name="binary_str"] {
                width: 400px;
                padding: 10px;
                margin-left: 45px;
            }
            input[name="fp_value"] {
                width: 400px;
                padding: 10px;
                margin-left: 70px;
            }
            input[name="diff_with_nearest"] {
                width: 400px;
                padding: 10px;
                margin-left: 10px;
            }
            input[name="exp_bias"] {
                width: 400px;
                padding: 10px;
                margin-left: 10px;
            }
            input[name="exp_max"] {
                width: 400px;
                padding: 10px;
                margin-left: 10px;
            }
            input[name="exp_val"] {
                width: 400px;
                padding: 10px;
                margin-left: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <form method="post">
                <!-- 参数输入行 -->
                <div class="input-group">
                    E <input type="number" name="E" value="{{ state.E }}">
                    M <input type="number" name="M" value="{{ state.M }}">
                    <button type="submit" name="set_params" style="background: #e0e0e0;">Expand</button>
                </div>

                <!-- Hex输入行 -->
                <div class="input-group">
                    <label>hex_value</label>
                    <input type="text" name="hex_value" value="{{ state.hex_value }}">
                    <button type="submit" name="convert_hex" style="background: #e0e0e0;">Convert to float</button>
                </div>

                <!-- 二进制位分组显示 -->
                {% if state.bits %}
                <div class="input-group">
                    <div class="bit-group sign-bit">
                        <button class="bit-btn" onclick="toggleBit(0)">{{ state.bits[0] }}</button>
                    </div>
                    <div class="bit-group exp-bits">
                        {% for i in range(1, state.E+1) %}
                        <button class="bit-btn" onclick="toggleBit({{ i }})">{{ state.bits[i] }}</button>
                        {% endfor %}
                    </div>
                    <div class="bit-group frac-bits">
                        {% for i in range(state.E+1, state.bits|length) %}
                        <button class="bit-btn" onclick="toggleBit({{ i }})">{{ state.bits[i] }}</button>
                        {% endfor %}
                    </div>
                    <button type="submit" name="convert" style="height:40px; background: #e0e0e0;">Update values</button>
                </div>
                {% endif %}
            </form>

            <!-- 输出结果区 -->
            <div class="output-box">
                <label>binary_input</label>
                <input type="text" name="binary_str" value="{{ state.binary_str }}">
            </div>
            <div class="output-box">
                <label>fp_value</label>
                <input type="text" name="fp_value" value="{{ state.fp_value }}">
            </div>
            <div class="output-box">
                <label>diff_with_nearest</label>
                <input type="text" name="diff_with_nearest" value="{{ state.diff_with_nearest }}">
            </div>
            <div class="output-box">
                <label>exp_bias</label>
                <input type="text" name="exp_bias" value="{{ state.exp_bias }}">
            </div>
            <div class="output-box">
                <label>exp_max</label>
                <input type="text" name="exp_max" value="{{ state.exp_max }}">
            </div>
            <div class="output-box">
                <label>exp_val</label>
                <input type="text" name="exp_val" value="{{ state.exp_val }}">
            </div>
        </div>

        <script>
            function toggleBit(index) {
                fetch('/toggle', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({index: index})
                }).then(response => response.json())
                  .then(data => {
                    document.querySelectorAll('.bit-btn')[index].textContent = data.new_value;
                });
            }
        </script>
    </body>
    </html>
    ''', state=state)

@app.route('/toggle', methods=['POST'])
def toggle_bit():
    data = request.get_json()
    index = data['index']
    if 0 <= index < len(state['bits']):
        state['bits'][index] = 1 - state['bits'][index]
    return jsonify(new_value=state['bits'][index])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
