1. 小图使用 32bit 以节省内存并提高速度。
    > DGL can use either 32- or 64-bit integers to store the node and edge IDs. The data types for the node and edge IDs should be the same. By using 64 bits, DGL can handle graphs with up to 263−1 nodes or edges. However, if a graph contains less than 231−1 nodes or edges, one should use 32-bit integers as it leads to better speed and requires less memory. DGL provides methods for making such conversions. See below for an example.

2. `pytorch ` 版本需要小于 1.8, 因为 `FFT` 功能有较大变动。