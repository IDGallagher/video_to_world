#include "ispc_texcomp.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{

bool ReadFileBytes(const char* path, std::vector<std::uint8_t>& out_bytes)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
    {
        return false;
    }

    const std::streamsize size = input.tellg();
    if (size < 0)
    {
        return false;
    }

    out_bytes.resize(static_cast<std::size_t>(size));
    input.seekg(0, std::ios::beg);
    return input.read(reinterpret_cast<char*>(out_bytes.data()), size).good();
}

bool WriteFileBytes(const char* path, const std::vector<std::uint8_t>& bytes)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        return false;
    }

    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return output.good();
}

std::size_t ComputeBC7Size(int width, int height)
{
    const int block_width = width / 4;
    const int block_height = height / 4;
    return static_cast<std::size_t>(block_width) * static_cast<std::size_t>(block_height) * 16u;
}

int CompressFile(const char* input_path, const char* output_path, int width, int height, int stride)
{
    if (width <= 0 || height <= 0 || stride < width * 4)
    {
        std::cerr << "Invalid dimensions or stride." << std::endl;
        return 1;
    }
    if ((width % 4) != 0 || (height % 4) != 0)
    {
        std::cerr << "Width and height must be multiples of 4 for BC7 input." << std::endl;
        return 1;
    }

    std::vector<std::uint8_t> input_bytes;
    if (!ReadFileBytes(input_path, input_bytes))
    {
        std::cerr << "Failed to read input file: " << input_path << std::endl;
        return 1;
    }

    const std::size_t minimum_size = static_cast<std::size_t>(stride) * static_cast<std::size_t>(height);
    if (input_bytes.size() < minimum_size)
    {
        std::cerr << "Input file is too small for the provided dimensions." << std::endl;
        return 1;
    }

    rgba_surface surface{};
    surface.ptr = input_bytes.data();
    surface.width = width;
    surface.height = height;
    surface.stride = stride;

    bc7_enc_settings settings{};
    GetProfile_basic(&settings);

    std::vector<std::uint8_t> output_bytes(ComputeBC7Size(width, height));
    CompressBlocksBC7(&surface, output_bytes.data(), &settings);

    if (!WriteFileBytes(output_path, output_bytes))
    {
        std::cerr << "Failed to write output file: " << output_path << std::endl;
        return 1;
    }

    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 7)
    {
        std::cerr << "Usage: divstream_bc7_cli compress <input_file> <output_file> <width> <height> <stride>" << std::endl;
        return 1;
    }

    const std::string command = argv[1];
    if (command != "compress")
    {
        std::cerr << "Unsupported command: " << command << std::endl;
        return 1;
    }

    const int width = std::atoi(argv[4]);
    const int height = std::atoi(argv[5]);
    const int stride = std::atoi(argv[6]);
    return CompressFile(argv[2], argv[3], width, height, stride);
}
